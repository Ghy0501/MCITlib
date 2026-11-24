# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Frozen historical LoRA branches for Octopus stage two.

History is intentionally external to the active LoRA state dict. A stage-two
checkpoint must record its ordered history paths and reattach them on reload.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MethodType

import torch
import torch.nn.functional as F


def _forward_with_history(layer, x):
    adapter = layer.active_adapter
    if adapter not in layer.lora_A or layer.disable_adapters:
        if layer.disable_adapters and layer.merged:
            layer.unmerge()
        return F.linear(x, layer.weight.T if layer.fan_in_fan_out else layer.weight, layer.bias)
    if layer.merged:
        return F.linear(x, layer.weight.T if layer.fan_in_fan_out else layer.weight, layer.bias)
    result = F.linear(x, layer.weight.T if layer.fan_in_fan_out else layer.weight, layer.bias)
    result_dtype = result.dtype
    dropped = layer.lora_dropout[adapter](x.to(layer.lora_A[adapter].weight.dtype))
    scale = layer.scaling[adapter]
    result = result + layer.lora_B[adapter](layer.lora_A[adapter](dropped)) * scale
    for index in range(layer._octopus_history_count):
        a = getattr(layer, f"_octopus_history_a_{index}")
        b = getattr(layer, f"_octopus_history_b_{index}")
        result = result + F.linear(F.linear(dropped, a), b) * scale
    return result.to(result_dtype)


def attach_history(
    model, adapters: Sequence[Mapping[str, torch.Tensor]], scales: Sequence[float],
) -> None:
    """Attach ordered frozen A/B tensors to every default-adapter linear layer.

    Names use PEFT's serialized keys with `.lora_A.weight` and
    `.lora_B.weight` suffixes. All historical snapshots must cover all active
    layers and have equal rank and scaling to the current adapter.
    """
    if not adapters or len(scales) != len(adapters):
        raise ValueError("Octopus stage two requires a scale for every historical adapter")
    layers = []
    for name, layer in model.named_modules():
        if not hasattr(layer, "lora_A") or not hasattr(layer, "lora_B"):
            continue
        if getattr(layer, "active_adapter", None) != "default" or "default" not in layer.lora_A:
            raise ValueError(f"unsupported active LoRA adapter at {name}")
        if getattr(layer, "_octopus_history_count", None) is not None:
            raise ValueError(f"history is already attached at {name}")
        current_a, current_b = layer.lora_A["default"].weight, layer.lora_B["default"].weight
        if any(scale != layer.scaling["default"] for scale in scales):
            raise ValueError(f"history scaling mismatch at {name}")
        pairs = []
        for adapter in adapters:
            akey, bkey = f"{name}.lora_A.weight", f"{name}.lora_B.weight"
            if akey not in adapter or bkey not in adapter:
                raise ValueError(f"history does not cover {name}")
            a, b = adapter[akey], adapter[bkey]
            if a.shape != current_a.shape or b.shape != current_b.shape:
                raise ValueError(f"history shape mismatch at {name}")
            pairs.append((a.detach().clone().to(current_a), b.detach().clone().to(current_b)))
        layers.append((layer, pairs))
    if not layers:
        raise ValueError("no compatible LoRA layers found")
    for layer, pairs in layers:
        layer._octopus_had_instance_forward = "forward" in vars(layer)
        layer._octopus_original_forward = vars(layer).get("forward")
        for index, (a, b) in enumerate(pairs):
            layer.register_buffer(f"_octopus_history_a_{index}", a, persistent=False)
            layer.register_buffer(f"_octopus_history_b_{index}", b, persistent=False)
        layer._octopus_history_count = len(pairs)
        layer.forward = MethodType(_forward_with_history, layer)


def detach_history(model) -> None:
    """Restore original forward methods and remove transient history buffers."""
    for layer in model.modules():
        count = getattr(layer, "_octopus_history_count", None)
        if count is None:
            continue
        if layer._octopus_had_instance_forward:
            layer.forward = layer._octopus_original_forward
        else:
            del layer.forward
        for index in range(count):
            delattr(layer, f"_octopus_history_a_{index}")
            delattr(layer, f"_octopus_history_b_{index}")
        del layer._octopus_history_count
        del layer._octopus_had_instance_forward
        del layer._octopus_original_forward


def pairs_for_regularization(model, gpwc: Mapping[str, torch.Tensor]):
    """Bind LoRA A/B parameters to their original unwrapped base-weight gradients."""
    pairs = []
    for name, layer in model.named_modules():
        if not hasattr(layer, "lora_A") or not hasattr(layer, "lora_B"):
            continue
        if not name.startswith("base_model.model."):
            raise ValueError(f"unexpected PEFT module prefix: {name}")
        key = name[len("base_model.model."):] + ".weight"
        if key not in gpwc:
            raise KeyError(f"GPWC missing base-weight gradient {key}")
        pairs.append((layer.lora_A["default"].weight, layer.lora_B["default"].weight, gpwc[key]))
    if not pairs:
        raise ValueError("no LoRA layers matched GPWC")
    return pairs
