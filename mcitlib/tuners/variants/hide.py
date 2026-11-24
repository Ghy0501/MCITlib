# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""HiDe exact expert fusion and last-layer sample-weight strategy."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.utils import PeftType

from ..inference_cache import (
    InferenceBankInvalidation,
    clear_expert_inference_cache,
    expert_weight,
    observe_active_adapter,
)
from ..mcit_expert_lora import (
    MCITExpertBankA,
    MCITExpertBankB,
    MCITExpertLoraConfig,
    MCITExpertLoraLayer,
    MCITExpertLoraLinear,
    MCITExpertLoraModel,
    transpose,
)


@dataclass
class MCITHiDeLoraConfig(MCITExpertLoraConfig):
    cur_task: int = field(default=4)

    def __post_init__(self):
        self.peft_type = PeftType.MOE_LORA_HiDe


class MCITHiDeBankA(InferenceBankInvalidation, MCITExpertBankA):
    def __init__(self, in_features, out_features, expert_num, cur_task, training, layer, weight):
        super().__init__(in_features, out_features, expert_num)
        self.cur_task, self.training, self.layer, self.expert_weight = cur_task, training, layer, weight

    def forward(self, x):
        if self.training:
            clear_expert_inference_cache()
            assert 0 <= self.cur_task < self.expert_num
            return self.loraA[self.cur_task](x)
        if int(self.layer) != 31:
            if not torch.is_grad_enabled():
                parameters = [self.loraA[i].weight for i in range(self.cur_task + 1)]

                def build():
                    fused = torch.zeros((self.r, self.in_features), device=x.device)
                    for weight in parameters:
                        fused += weight
                    return fused

                return F.linear(x, expert_weight(self, parameters, x,
                                extra=('hide-a', self.cur_task, self.expert_num, self.layer), build=build))
            clear_expert_inference_cache()
            temp = nn.Linear(self.in_features, self.r, bias=False).to(x.device)
            fused = torch.zeros((self.r, self.in_features), device=x.device)
            for i in range(self.cur_task + 1):
                fused += self.loraA[i].weight
            with torch.no_grad():
                temp.weight.copy_(fused)
            return temp(x)
        return [self.loraA[i](x) for i in range(self.expert_num)]


class MCITHiDeBankB(InferenceBankInvalidation, MCITExpertBankB):
    def __init__(self, in_features, out_features, expert_num, cur_task, training, layer, weight):
        super().__init__(in_features, out_features, expert_num)
        self.cur_task, self.training, self.layer, self.expert_weight = cur_task, training, layer, weight

    def forward(self, x):
        if self.training:
            clear_expert_inference_cache()
            assert 0 <= self.cur_task < self.expert_num
            return self.loraB[self.cur_task](x)
        if int(self.layer) != 31:
            if not torch.is_grad_enabled():
                parameters = [self.loraB[i].weight for i in range(self.cur_task + 1)]

                def build():
                    fused = torch.zeros((self.out_features, self.r), device=x.device)
                    for weight in parameters:
                        fused += weight
                    return fused

                return F.linear(x, expert_weight(self, parameters, x,
                                extra=('hide-b', self.cur_task, self.expert_num, self.layer), build=build))
            clear_expert_inference_cache()
            temp = nn.Linear(self.r, self.out_features, bias=False).to(x.device)
            fused = torch.zeros((self.out_features, self.r), device=x.device)
            for i in range(self.cur_task + 1):
                fused += self.loraB[i].weight
            with torch.no_grad():
                temp.weight.copy_(fused)
            return temp(x)
        return [self.loraB[i](x[i]) for i in range(self.expert_num)]


class MCITHiDeLoraLayer(MCITExpertLoraLayer):
    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        self.lora_dropout.update(nn.ModuleDict({adapter_name: dropout}))
        if r > 0:
            self.lora_A.update(nn.ModuleDict({adapter_name: MCITHiDeBankA(
                self.in_features, r, self.expert_num, self.cur_task, self.training, self.layer, self.expert_weight
            )}))
            self.lora_B.update(nn.ModuleDict({adapter_name: MCITHiDeBankB(
                r, self.out_features, self.expert_num, self.cur_task, self.training, self.layer, self.expert_weight
            )}))
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)


class MCITHiDeLoraLinear(MCITExpertLoraLinear, MCITHiDeLoraLayer):
    def __init__(self, adapter_name, in_features, out_features, r=0, lora_alpha=1,
                 lora_dropout=0.0, fan_in_fan_out=False, train_signal=False, layer=0,
                 expert_weight=None, cur_task=0, **kwargs):
        self.cur_task, self.training = cur_task, train_signal
        self.layer, self.expert_weight = layer, expert_weight if expert_weight is not None else []
        super().__init__(adapter_name, in_features, out_features, r=r, lora_alpha=lora_alpha,
                         lora_dropout=lora_dropout, fan_in_fan_out=fan_in_fan_out, **kwargs)
        self.cur_task, self.training = cur_task, train_signal
        self.layer, self.expert_weight = layer, self.expert_weight

    def forward(self, x: torch.Tensor, **kwargs):
        if self.training or torch.is_grad_enabled():
            clear_expert_inference_cache()
        else:
            observe_active_adapter(self)
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
            if self.training:
                a = self.lora_A[self.active_adapter].loraA[self.cur_task](self.lora_dropout[self.active_adapter](x))
                b = self.lora_B[self.active_adapter].loraB[self.cur_task](a)
                result += b * self.scaling[self.active_adapter]
            elif int(self.layer) != 31:
                a = self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                b = self.lora_B[self.active_adapter](a)
                result += b * self.scaling[self.active_adapter]
            else:
                for i in range(len(self.expert_weight)):
                    result += self.lora_B[self.active_adapter].loraB[i](
                        self.lora_A[self.active_adapter].loraA[i](self.lora_dropout[self.active_adapter](x))
                    ) * self.scaling[self.active_adapter] * self.expert_weight[i]
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        return result.to(previous_dtype)


class MCITHiDeLoraModel(MCITExpertLoraModel):
    def __init__(self, model, config, adapter_name):
        super().__init__(model, config, adapter_name)
        self.expert_weight = [0.0] * 8

    def _module_context(self, key):
        return {"layer": key.split(".")[2]}

    def _extra_module_kwargs(self, lora_config):
        return {
            "cur_task": lora_config.cur_task,
            "train_signal": self.model.training,
            "expert_weight": self.expert_weight,
        }

    def _linear_class(self):
        return MCITHiDeLoraLinear
