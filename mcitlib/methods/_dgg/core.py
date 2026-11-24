# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""DGG's per-backward parameter guidance and released replay selection.

The hook must be installed between the DeepSpeed optimizer backward and the
engine all-reduce epilogue. Calling it as a loss term changes the algorithm.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from random import Random
from typing import Any

import numpy as np
import torch


def replay_rows(
    current: Sequence[dict[str, Any]], previous: Sequence[Sequence[dict[str, Any]]], *,
    samples_per_task: int = 2000, seed: int = 1993,
) -> list[dict[str, Any]]:
    """Current rows then one independently seeded sample per previous task."""
    if samples_per_task < 0:
        raise ValueError("negative replay size")
    rows = deepcopy(list(current))
    for task in previous:
        # The release invokes a separately seeded sampler for every input file.
        rows.extend(deepcopy(Random(seed).sample(list(task), min(samples_per_task, len(task)))))
    return rows


def image_paths_absolute(rows: Sequence[dict[str, Any]], image_root: str | Path) -> list[dict[str, Any]]:
    """Resolve relative images before joining records from distinct task roots."""
    root = Path(image_root).resolve()
    out = deepcopy(list(rows))
    for row in out:
        image = row.get("image")
        if image is not None:
            def absolute(path):
                if not isinstance(path, str):
                    raise ValueError("image must be a path string")
                return str(root / path) if not Path(path).is_absolute() else path
            if isinstance(image, list):
                row["image"] = [absolute(path) for path in image]
            else:
                row["image"] = absolute(image)
    return out


class GradientGuidance:
    """A previous-task anchor with one NumPy Bernoulli draw per backward."""

    def __init__(self, history: Mapping[str, torch.Tensor], probability: float):
        if not history or not 0 <= probability <= 1:
            raise ValueError("DGG requires nonempty history and a probability in [0, 1]")
        self.history = {name: value.detach().cpu().clone() for name, value in history.items()}
        self.probability = probability
        self.last_seen = self.last_nonzero = self.last_changed = 0

    def apply(self, named_parameters, *, rng=np.random) -> bool:
        self.last_seen = self.last_nonzero = self.last_changed = 0
        if rng.binomial(1, self.probability) < 1:
            return False
        with torch.no_grad():
            for name, parameter in named_parameters:
                grad = parameter.grad
                if grad is None:
                    continue
                self.last_seen += 1
                if name not in self.history:
                    raise KeyError(f"DGG history is missing trainable parameter {name}")
                grad_norm = grad.norm()
                if grad_norm == 0:
                    continue
                self.last_nonzero += 1
                anchor = self.history[name]
                if anchor.shape != parameter.shape:
                    raise ValueError(f"DGG history shape mismatch: {name}")
                if anchor.device != parameter.device or anchor.dtype != parameter.dtype:
                    anchor = anchor.to(device=parameter.device, dtype=parameter.dtype)
                    self.history[name] = anchor
                delta = (parameter - anchor).detach()
                delta_norm = delta.norm()
                if delta_norm != 0:
                    if delta_norm > grad_norm:
                        delta = delta / delta_norm * grad_norm
                    grad.add_(delta)
                    self.last_changed += 1
        return True

    def state_dict(self) -> dict[str, Any]:
        return {"probability": self.probability, "history": {k: v.clone() for k, v in self.history.items()}}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "GradientGuidance":
        return cls(state["history"], state["probability"])
