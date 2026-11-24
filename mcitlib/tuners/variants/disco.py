# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""DISCO task expert and block-diagonal expert-merging strategy."""
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
class MCITDiscoLoraConfig(MCITExpertLoraConfig):
    cur_task: int = field(default=0)

    def __post_init__(self):
        self.peft_type = PeftType.MOE_LORA_CoIN


class MCITDiscoBankA(InferenceBankInvalidation, MCITExpertBankA):
    def __init__(self, in_features, out_features, expert_num, cur_task, training):
        super().__init__(in_features, out_features, expert_num)
        self.cur_task = cur_task
        self.training = training

    def forward(self, x):
        if not self.training and not torch.is_grad_enabled():
            parameters = [self.loraA[i].mlp.weight for i in range(self.expert_num)]

            def build():
                # The legacy temporary Linear is initialized in the default
                # dtype before its concatenated expert weights are copied in.
                return torch.cat(parameters, dim=0).to(device=x.device, dtype=torch.get_default_dtype())

            return F.linear(x, expert_weight(self, parameters, x, extra=('disco-a', self.expert_num), build=build))
        clear_expert_inference_cache()
        temp = nn.Linear(self.in_features, self.expert_num * self.r, bias=False).to(x.device)
        weight = torch.cat([self.loraA[i].mlp.weight for i in range(self.expert_num)], dim=0)
        with torch.no_grad():
            temp.weight.copy_(weight)
        return temp(x)


class MCITDiscoBankB(InferenceBankInvalidation, MCITExpertBankB):
    def __init__(self, in_features, out_features, expert_num, cur_task, training):
        super().__init__(in_features, out_features, expert_num)
        self.cur_task = cur_task
        self.training = training

    def forward(self, x):
        if not self.training and not torch.is_grad_enabled():
            parameters = [self.loraB[i].mlp.weight for i in range(self.expert_num)]

            def build():
                return torch.cat(parameters, dim=1).to(device=x.device, dtype=torch.get_default_dtype())

            return F.linear(x, expert_weight(self, parameters, x, extra=('disco-b', self.expert_num), build=build))
        clear_expert_inference_cache()
        temp = nn.Linear(self.expert_num * self.r, self.out_features, bias=False).to(x.device)
        weight = torch.cat([self.loraB[i].mlp.weight for i in range(self.expert_num)], dim=1)
        with torch.no_grad():
            temp.weight.copy_(weight)
        return temp(x)


class MCITDiscoLoraLayer(MCITExpertLoraLayer):
    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        self.lora_dropout.update(nn.ModuleDict({adapter_name: dropout}))
        if r > 0:
            self.lora_A.update(nn.ModuleDict({adapter_name: MCITDiscoBankA(
                self.in_features, r, self.expert_num, self.cur_task, self.training
            )}))
            self.lora_B.update(nn.ModuleDict({adapter_name: MCITDiscoBankB(
                r, self.out_features, self.expert_num, self.cur_task, self.training
            )}))
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)


def apply_block_diagonal_expert_mask(weight: torch.Tensor, mask_signal, diag_size: int) -> None:
    """Apply DISCO's v3 block-diagonal mask without changing operation order."""
    for index, flag in enumerate(mask_signal):
        if flag != 0:
            start = index * diag_size
            end = start + diag_size
            if end > start and diag_size > 0:
                weight.data[start:end, start:end] = torch.eye(diag_size) * flag


class MCITDiscoLoraLinear(MCITExpertLoraLinear, MCITDiscoLoraLayer):
    def __init__(self, adapter_name, in_features, out_features, r=0, lora_alpha=1,
                 lora_dropout=0.0, fan_in_fan_out=False, train_signal=False, cur_task=0, **kwargs):
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self.expert_num = kwargs.pop("expert_num", True)
        self.te_dim = kwargs.pop("task_embedding_dim", True)
        self.cur_task = cur_task
        self.lora_id = cur_task
        self.mask_signal = [1.0, 0, 0, 0, 0, 0, 0, 0]
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        MCITDiscoLoraLayer.__init__(self, in_features=in_features, out_features=out_features, expert_num=self.expert_num)
        self.training = train_signal
        self.lora_AB = nn.ModuleDict({adapter_name: nn.Linear(r, r, bias=False)})
        self.lora_AB[adapter_name].weight.data.zero_()
        self.diag_size = r // self.expert_num
        self.weight.requires_grad = False
        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

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
            apply_block_diagonal_expert_mask(
                self.lora_AB[self.active_adapter].weight, self.mask_signal, self.diag_size
            )
            self.lora_AB = self.lora_AB.to(x.device)
            if self.training:
                a = self.lora_A[self.active_adapter].loraA[self.lora_id](self.lora_dropout[self.active_adapter](x))
                b = self.lora_B[self.active_adapter].loraB[self.lora_id](a)
            else:
                a = self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
                ab = self.lora_AB[self.active_adapter](a)
                b = self.lora_B[self.active_adapter](ab)
            result += b * self.scaling[self.active_adapter]
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        return result.to(previous_dtype)


class MCITDiscoLoraModel(MCITExpertLoraModel):
    def _extra_module_kwargs(self, lora_config):
        return {"cur_task": lora_config.cur_task, "train_signal": self.model.training}

    def _linear_class(self):
        return MCITDiscoLoraLinear
