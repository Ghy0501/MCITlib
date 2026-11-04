# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""KeepLoRA's SVD-based A initialization and A-only freezing."""
from __future__ import annotations

import torch


def freeze_lora_a_matrices(model, local_rank=-1):
    """Freeze every adapter A matrix and return its frozen parameter count."""
    from peft.tuners.lora import LoraLayer

    frozen_params = 0
    for _, module in model.named_modules():
        if isinstance(module, LoraLayer) and hasattr(module, "lora_A"):
            for lora_a in module.lora_A.values():
                if hasattr(lora_a, "weight"):
                    lora_a.weight.requires_grad = False
                    frozen_params += lora_a.weight.numel()
    return frozen_params


def initialize_lora_weights_custom(model, gradient_dict, lora_r, local_rank=-1):
    """Copy precomputed SVD directions into A, including ZeRO-gathered weights."""
    import deepspeed
    from peft.tuners.lora import LoraLayer

    is_deepspeed = hasattr(model, "module")
    target_model = model.module if is_deepspeed else model
    initialized_count = 0
    for name, module in target_model.named_modules():
        if not isinstance(module, LoraLayer) or not hasattr(module, "lora_A"):
            continue
        for lora_a in module.lora_A.values():
            if not hasattr(lora_a, "weight"):
                continue
            param = lora_a.weight
            svd_weight = None
            for grad_name, grad_svd in gradient_dict.items():
                if grad_name.replace(".weight", "") in name:
                    svd_weight = grad_svd
                    break
            if svd_weight is None:
                continue
            if svd_weight.shape != param.data.shape:
                raise ValueError(
                    f"Shape mismatch for {name}: SVD weight shape {svd_weight.shape} "
                    f"!= param shape {param.data.shape}"
                )
            if is_deepspeed and hasattr(param, "ds_id"):
                with deepspeed.zero.GatheredParameters([param], modifier_rank=0):
                    if local_rank in [0, -1]:
                        with torch.no_grad():
                            param.data.copy_(svd_weight.to(param.data.device))
                            initialized_count += 1
            else:
                with torch.no_grad():
                    param.data.copy_(svd_weight.to(param.data.device))
                    initialized_count += 1

    if local_rank in [0, -1]:
        print(f"Successfully initialized {initialized_count} LoRA matrices with pre-computed SVD")
        if initialized_count == 0:
            raise ValueError("No LoRA matrices were initialized! Check gradient file and parameter names.")
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
