# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""O-LoRA method plugin and orthogonal regularization hook."""
from __future__ import annotations

import re
from typing import Any

import torch

from ..config.schema import DataSource, TaskSpec
from ..tuners._peft_registry import tuner_model_context
from ..tuners.variants.olora import MCITOrthogonalLoraConfig, MCITOrthogonalLoraModel
from .base import ContinualMethod


class OLoRAMethod(ContinualMethod):
    id = "olora"
    aliases = ("OLoRA", "O-LoRA")
    train_entry = "moe"

    def __init__(self, cfg: dict[str, Any] | None = None, ctx: Any = None):
        self.cfg = cfg or {}
        self.cur_task = int(self.cfg.get("cur_task", 0))

    def select_train_data(self, task: TaskSpec) -> DataSource:
        if task.data is None:
            raise ValueError("task has no data source")
        return task.data

    def build_trainable(self, model: Any, backbone: Any) -> Any:
        from peft import get_peft_model
        config = MCITOrthogonalLoraConfig(
            r=self.cfg.get("r", 16),
            lora_alpha=self.cfg.get("lora_alpha", 32),
            target_modules=self.cfg.get("target_modules"),
            lora_dropout=self.cfg.get("lora_dropout", 0.05),
            bias=self.cfg.get("bias", "none"),
            task_type="CAUSAL_LM",
            task_embedding_dim=self.cfg.get("task_embedding_dim", 64),
            expert_num=self.cfg.get("expert_num", 4),
            cur_task=self.cur_task,
        )
        with tuner_model_context("MOE_LORA_CoIN", MCITOrthogonalLoraModel):
            return get_peft_model(model, config)

    def training_overrides(self) -> dict[str, Any]:
        return {"gradient_checkpointing_kwargs": {"use_reentrant": False}}

    def before_task(self, ctx: Any) -> list:
        self.cur_task = int(self.cfg.get("cur_task", ctx.task_id - 1))
        return []

    def extra_loss(self, model: Any, inputs: Any, outputs: Any, base_loss: Any) -> Any:
        orthogonal_loss = base_loss.new_tensor(0.0)
        if self.cur_task < 1:
            return orthogonal_loss
        current, previous = {}, {}
        pattern = re.compile(r"default\.loraA\.(\d+)\.weight")
        for name, param in model.named_parameters():
            match = pattern.search(name)
            if match is None:
                continue
            task_id = int(match.group(1))
            prefix = name.split("loraA")[0]
            if task_id == self.cur_task:
                current[prefix] = param
            elif 0 <= task_id < self.cur_task:
                previous.setdefault(prefix, {})[task_id] = param
        for prefix, current_a in current.items():
            for prev_task, previous_a in previous.get(prefix, {}).items():
                orthogonal_loss += torch.abs(torch.mm(current_a, previous_a.detach().T)).sum()
        return 0.05 * orthogonal_loss
