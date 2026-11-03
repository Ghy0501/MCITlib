# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""O-LoRA expert selection and concatenated inference strategy."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.utils import PeftType

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
class MCITOrthogonalLoraConfig(MCITExpertLoraConfig):
    cur_task: int = field(default=4)

    def __post_init__(self):
        self.peft_type = PeftType.MOE_LORA_CoIN


class MCITOrthogonalBankA(MCITExpertBankA):
    def __init__(self, in_features, out_features, expert_num, cur_task, training):
        super().__init__(in_features, out_features, expert_num)
        self.cur_task = cur_task
        self.training = training

    def forward(self, x, cur_task):
        if self.training:
            assert 0 <= self.cur_task < self.expert_num, "Invalid current_task value"
            return self.loraA[self.cur_task](x)
        temp = nn.Linear(self.in_features, (cur_task + 1) * self.r, bias=False).to(x.device)
        with torch.no_grad():
            temp.weight.copy_(torch.cat([self.loraA[i].mlp.weight for i in range(cur_task + 1)], dim=0))
        return temp(x)


class MCITOrthogonalBankB(MCITExpertBankB):
    def __init__(self, in_features, out_features, expert_num, cur_task, training):
        super().__init__(in_features, out_features, expert_num)
        self.cur_task = cur_task
        self.training = training

    def forward(self, x, cur_task):
        if self.training:
            assert 0 <= self.cur_task < self.expert_num, "Invalid current_task value"
            return self.loraB[self.cur_task](x)
        temp = nn.Linear((cur_task + 1) * self.r, self.out_features, bias=False).to(x.device)
        with torch.no_grad():
            temp.weight.copy_(torch.cat([self.loraB[i].mlp.weight for i in range(cur_task + 1)], dim=1))
        return temp(x)


class MCITOrthogonalLoraLayer(MCITExpertLoraLayer):
    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        self.lora_dropout.update(nn.ModuleDict({adapter_name: dropout}))
        if r > 0:
            self.lora_A.update(nn.ModuleDict({adapter_name: MCITOrthogonalBankA(
                self.in_features, r, self.expert_num, self.cur_task, self.training
            )}))
            self.lora_B.update(nn.ModuleDict({adapter_name: MCITOrthogonalBankB(
                r, self.out_features, self.expert_num, self.cur_task, self.training
            )}))
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)


class MCITOrthogonalLoraLinear(MCITExpertLoraLinear, MCITOrthogonalLoraLayer):
    def __init__(self, adapter_name, in_features, out_features, r=0, lora_alpha=1,
                 lora_dropout=0.0, fan_in_fan_out=False, train_signal=False, cur_task=0, **kwargs):
        self.cur_task = cur_task
        self.training = train_signal
        super().__init__(adapter_name, in_features, out_features, r=r, lora_alpha=lora_alpha,
                         lora_dropout=lora_dropout, fan_in_fan_out=fan_in_fan_out, **kwargs)
        self.cur_task = cur_task
        self.training = train_signal

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
            if self.training:
                a = self.lora_A[self.active_adapter].loraA[self.cur_task](self.lora_dropout[self.active_adapter](x))
                b = self.lora_B[self.active_adapter].loraB[self.cur_task](a)
            else:
                a = self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x), self.cur_task)
                b = self.lora_B[self.active_adapter](a, self.cur_task)
            result += b * self.scaling[self.active_adapter]
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        return result.to(previous_dtype)


class MCITOrthogonalLoraModel(MCITExpertLoraModel):
    def _extra_module_kwargs(self, lora_config):
        return {"cur_task": lora_config.cur_task, "train_signal": self.model.training}

    def _linear_class(self):
        return MCITOrthogonalLoraLinear
