# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 MCITlib authors
"""Original V4 implementation of SiGMA sign adaptation and inference merging.

The upstream reference has no declared license; no upstream source is vendored.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch
from peft.utils import PeftType
from torch import nn
from torch.nn import functional as F

from ..mcit_expert_lora import (
    MCITExpert,
    MCITExpertLoraConfig,
    MCITExpertLoraLinear,
    MCITExpertLoraModel,
    transpose,
)


@dataclass
class SiGMAConfig(MCITExpertLoraConfig):
    cur_task: int = 0
    specific_layers: list[int] = field(default_factory=lambda: [31])
    relaxation_ratio: float = 0.8

    def __post_init__(self):
        self.peft_type = PeftType.MOE_LORA_SIGMA


class _BankA(nn.Module):
    def __init__(self, in_features, rank, num):
        super().__init__()
        self.loraA = nn.ModuleList([MCITExpert(in_features, rank) for _ in range(num)])


class _BankB(nn.Module):
    def __init__(self, rank, out_features, num):
        super().__init__()
        self.loraB = nn.ModuleList([MCITExpert(rank, out_features) for _ in range(num)])


def inference_delta(deltas, relaxation_ratio=0.8):
    """Merge task deltas using signs, an overpower mask, and cosine scaling."""
    if not deltas:
        raise ValueError('SiGMA requires at least one task delta')
    first = deltas[0]
    merged = first.clone()
    for delta in deltas[1:]:
        same = first.sign() == delta.sign()
        opposite = ~same & (delta.abs() > relaxation_ratio * first.abs())
        reference = first * opposite
        conflicting = delta * opposite
        similarity = F.cosine_similarity(reference.reshape(-1), conflicting.reshape(-1), dim=0)
        merged = merged + delta * same + (1 - similarity) * conflicting
    return merged


class SiGMALinear(MCITExpertLoraLinear):
    def __init__(self, *args, cur_task=0, specific_layers=None, relaxation_ratio=0.8, layer_index=-1, **kwargs):
        self.cur_task = int(cur_task)
        self.specific_layers = list([31] if specific_layers is None else specific_layers)
        self.relaxation_ratio = float(relaxation_ratio)
        self.layer_index = int(layer_index)
        super().__init__(*args, **kwargs)
        if not 0 <= self.cur_task < self.expert_num:
            raise ValueError('SiGMA task index must address an expert')

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.lora_dropout[adapter_name] = nn.Dropout(lora_dropout) if lora_dropout else nn.Identity()
        if r <= 0:
            raise ValueError('SiGMA requires a positive per-task rank')
        self.lora_A[adapter_name] = _BankA(self.in_features, r, self.expert_num)
        self.lora_B[adapter_name] = _BankB(r, self.out_features, self.expert_num)
        self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            for a, b in zip(self.lora_A[adapter_name].loraA, self.lora_B[adapter_name].loraB):
                nn.init.kaiming_uniform_(a.weight, a=5 ** 0.5)
                nn.init.zeros_(b.weight)
        self.to(self.weight.device)

    def freeze_inactive(self):
        for i, (a, b) in enumerate(zip(self.lora_A[self.active_adapter].loraA, self.lora_B[self.active_adapter].loraB)):
            for p in (*a.parameters(), *b.parameters()):
                p.requires_grad_(i == self.cur_task)
        for p in self.lora_router.parameters():
            p.requires_grad_(False)

    def _deltas(self):
        a = self.lora_A[self.active_adapter].loraA
        b = self.lora_B[self.active_adapter].loraB
        return [b[i].weight @ a[i].weight * self.scaling[self.active_adapter] for i in range(self.cur_task + 1)]

    @torch.no_grad()
    def _add_merged_to(self, weight):
        # Match the release's accumulation order and destination dtype: each
        # same-sign and conflict term is rounded when added to the base, and
        # the anchor is added only after every later task.
        deltas = self._deltas()
        anchor = deltas[0]
        for delta in deltas[1:]:
            same = delta.sign() == anchor.sign()
            conflict = ~same & (delta.abs() > self.relaxation_ratio * anchor.abs())
            selected = delta * conflict
            reference = anchor * conflict.to(weight.dtype)
            scale = 1 - F.cosine_similarity(reference.reshape(-1), selected.reshape(-1), dim=0)
            weight.add_(transpose(delta * same, self.fan_in_fan_out))
            weight.add_(transpose(scale * selected, self.fan_in_fan_out))
        weight.add_(transpose(anchor, self.fan_in_fan_out))
        return weight

    @torch.no_grad()
    def merge(self):
        if not self.merged:
            self._add_merged_to(self.weight)
            self.merged = True

    def unmerge(self):
        if self.merged:
            # Floating point per-add rounding makes subtraction inexact. Avoid
            # retaining a full base-model copy during inference.
            raise RuntimeError('SiGMA inference merge cannot be reversed exactly; reload the base model')

    def forward(self, x, **kwargs):
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), self.bias)
        if self.disable_adapters or self.merged or self.active_adapter not in self.lora_A:
            return result
        if not self.training:
            # The ordinary builder calls merge_and_unload before generation.
            # Computing the same weight here also makes an unmerged PEFT model usable.
            merged_weight = self._add_merged_to(self.weight.detach().clone())
            return F.linear(x, transpose(merged_weight, self.fan_in_fan_out), self.bias)
        dtype = x.dtype
        a = self.lora_A[self.active_adapter].loraA
        b = self.lora_B[self.active_adapter].loraB
        x = self.lora_dropout[self.active_adapter](x.to(a[0].weight.dtype))
        current = b[self.cur_task](a[self.cur_task](x)) * self.scaling[self.active_adapter]
        if self.cur_task and self.layer_index not in self.specific_layers:
            with torch.no_grad():
                history = sum(e.weight for e in b[:self.cur_task]) @ sum(e.weight for e in a[:self.cur_task])
                present = b[self.cur_task].weight @ a[self.cur_task].weight
                guided = history * (history.sign() == present.sign())
            current = current + F.linear(x, guided)
        return (result + current).to(dtype)


class SiGMAModel(MCITExpertLoraModel):
    def _create_new_module(self, config, adapter_name, target, **context):
        if getattr(self.model, 'is_loaded_in_4bit', False) or getattr(self.model, 'is_loaded_in_8bit', False):
            raise ValueError('SIGMA currently supports only 16-bit unquantized LLaVA')
        if not isinstance(target, nn.Linear):
            raise ValueError('SIGMA requires a torch.nn.Linear target')
        return super()._create_new_module(config, adapter_name, target, **context)

    def _linear_class(self):
        return SiGMALinear

    def _module_context(self, key):
        match = re.search(r'(?:^|\.)layers\.(\d+)\.', key)
        return {'layer_index': int(match.group(1)) if match else -1}

    def _extra_module_kwargs(self, cfg):
        return {'cur_task': cfg.cur_task, 'specific_layers': cfg.specific_layers, 'relaxation_ratio': cfg.relaxation_ratio}

    def add_adapter(self, adapter_name, config=None):
        super().add_adapter(adapter_name, config)
        for layer in self.model.modules():
            if isinstance(layer, SiGMALinear) and not self.peft_config[adapter_name].inference_mode:
                layer.freeze_inactive()
