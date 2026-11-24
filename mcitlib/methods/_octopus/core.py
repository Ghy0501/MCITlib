# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Numerical operations needed by the Octopus LLaVA port.

These functions do not register an executable method.  In particular, collecting
GPWC still requires a full-model backward pass on the current task.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch


def sample_indices(size: int, count: int, seed: int = 42) -> list[int]:
    """Match Swift's repeat-plus-permutation dataset sampling, including repeats."""
    if size <= 0 or count <= 0:
        raise ValueError("size and sample count must be positive")
    repetitions, remainder = divmod(count, size)
    indices = np.tile(np.arange(size), repetitions)
    if remainder:
        tail = np.random.RandomState(seed).permutation(size)[:remainder]
        indices = np.concatenate((indices, tail))
    return indices.tolist()


class GradientMean:
    """Per-batch running GPWC mean, retaining the original tensor dtypes."""

    def __init__(self) -> None:
        self.count = 0
        self.values: dict[str, torch.Tensor] = {}

    def update(self, gradients: Mapping[str, torch.Tensor]) -> None:
        if not gradients:
            raise ValueError("GPWC batch has no gradients")
        gradients = {key: value.detach().cpu().clone() for key, value in gradients.items()}
        if self.count and gradients.keys() != self.values.keys():
            raise ValueError("GPWC parameter set changed between batches")
        if not self.count:
            self.values = gradients
        else:
            for key, average in self.values.items():
                if average.shape != gradients[key].shape or average.dtype != gradients[key].dtype:
                    raise ValueError(f"GPWC shape or dtype changed: {key}")
                self.values[key] = average + (gradients[key] - average) / (self.count + 1)
        self.count += 1

    def state_dict(self) -> dict[str, Any]:
        return {"count": self.count, "values": {k: v.clone() for k, v in self.values.items()}}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        count, values = state["count"], state["values"]
        if not isinstance(count, int) or count < 0 or bool(values) != bool(count):
            raise ValueError("invalid GPWC running-mean state")
        self.count = count
        self.values = {k: v.detach().cpu().clone() for k, v in values.items()}


def sum_history(gradients: Sequence[Mapping[str, torch.Tensor]], *, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    """Sum each historical model's GPWC in the release's stack/sum order."""
    if not gradients:
        raise ValueError("at least one historical GPWC is required")
    keys = gradients[0].keys()
    if any(entry.keys() != keys for entry in gradients[1:]):
        raise ValueError("historical GPWC keys differ")
    return {key: torch.stack([entry[key].to(device) for entry in gradients]).sum(dim=0).detach().cpu()
            for key in keys}


def regularization(
    pairs: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    lambda_1: float,
    lambda_2: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Released Stage-2 loss for (LoRA A, LoRA B, summed full-weight GPWC).

    GPWC and A's contribution to its projection are detached, as in the release.
    The result is added once per backward call; the training loop is responsible
    for reproducing its placement relative to gradient accumulation scaling.
    """
    if not pairs:
        raise ValueError("at least one LoRA A/B pair is required")
    orth = pairs[0][1].new_zeros(())
    magnitude = pairs[0][1].new_zeros(())
    for a, b, gradient in pairs:
        if a.ndim != 2 or b.ndim != 2 or gradient.shape != (b.shape[0], a.shape[1]) or b.shape[1] != a.shape[0]:
            raise ValueError("incompatible A, B, and GPWC shapes")
        gradient = gradient.to(device=b.device, dtype=b.dtype).detach()
        projected = (a @ gradient.T).detach()
        b_normalized = b / (torch.norm(b) + 1e-8)
        orth = orth + torch.trace(b_normalized @ (projected / (torch.norm(projected) + 1e-8))).abs()
        magnitude = magnitude + torch.norm(b)
    return orth * lambda_1 + magnitude * lambda_2, orth, magnitude


@torch.no_grad()
def merge_adapters_into_weights(
    weights: Mapping[str, torch.Tensor],
    adapters: Sequence[Mapping[str, tuple[torch.Tensor, torch.Tensor, float]]],
) -> None:
    """Merge each incremental adapter in order, preserving intermediate rounding.

    Each adapter maps a base-weight key to its A, B, and alpha/r scale.  These
    are separate in-place additions because concatenating factors changes bf16
    rounding from the released sequential merge path.
    """
    for adapter in adapters:
        for name, (a, b, scale) in adapter.items():
            if name not in weights:
                raise ValueError(f"missing weight for adapter: {name}")
            weight = weights[name]
            a, b = a.to(device=weight.device, dtype=weight.dtype), b.to(device=weight.device, dtype=weight.dtype)
            if a.ndim != 2 or b.ndim != 2 or (b.shape[0], a.shape[1]) != tuple(weight.shape) or b.shape[1] != a.shape[0]:
                raise ValueError(f"adapter shape mismatch: {name}")
            weight.add_((b @ a) * scale)
