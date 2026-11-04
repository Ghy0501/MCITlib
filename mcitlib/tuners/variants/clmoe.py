# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""CL-MoE's base-router computation with optional context-scoped diagnostics."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from peft.utils import PeftType

from ..mcit_expert_lora import MCITExpertLoraConfig, MCITExpertLoraLinear, MCITExpertLoraModel, transpose


@dataclass
class MCITClMoELoraConfig(MCITExpertLoraConfig):
    def __post_init__(self):
        self.peft_type = PeftType.MOE_LORA_CLMoE


class MCITClMoELoraLinear(MCITExpertLoraLinear):
    def __init__(self, *args, router_stats=None, **kwargs):
        self.router_stats = router_stats
        self.noisy_gating = True
        self.topk = 2
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, **kwargs):
        previous_dtype = x.dtype
        if self.active_adapter not in self.lora_A.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r[self.active_adapter] > 0:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            x = x.to(self.lora_A[self.active_adapter].loraA[0].weight.dtype)
            self.lora_router = self.lora_router.to(x.device)
            router = self.lora_router[self.active_adapter](x)
            router = torch.softmax(router, dim=-1)
            if self.router_stats is not None:
                self.router_stats.record(router)
            for i in range(self.expert_num):
                result += (
                    self.lora_B[self.active_adapter].loraB[i](
                        self.lora_A[self.active_adapter].loraA[i](self.lora_dropout[self.active_adapter](x)),
                    )
                    * self.scaling[self.active_adapter]
                    * router[:, :, i].unsqueeze(-1)
                )
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        return result.to(previous_dtype)


class MCITClMoELoraModel(MCITExpertLoraModel):
    def _extra_module_kwargs(self, lora_config):
        return {"router_stats": getattr(lora_config, "_mcit_router_stats", None)}

    def _linear_class(self):
        return MCITClMoELoraLinear
