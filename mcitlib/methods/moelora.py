# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""MoELoRA plugin backed by MCITlib's single expert-LoRA implementation."""
from __future__ import annotations

from typing import Any

from ..config.schema import DataSource, TaskSpec
from ..tuners.mcit_expert_lora import MCITExpertLoraConfig, MCITExpertLoraModel
from .base import ContinualMethod


class MoELoRAMethod(ContinualMethod):
    id = "moelora"
    aliases = ("MoELoRA", "moe-lora")
    train_entry = "moe"

    def __init__(self, cfg: dict[str, Any] | None = None, ctx: Any = None):
        self.cfg = cfg or {}

    def select_train_data(self, task: TaskSpec) -> DataSource:
        if task.data is None:
            raise ValueError("task has no data source")
        return task.data

    def build_trainable(self, model: Any, backbone: Any) -> Any:
        from peft import get_peft_model

        config = MCITExpertLoraConfig(
            r=self.cfg.get("r", 16),
            lora_alpha=self.cfg.get("lora_alpha", 32),
            target_modules=self.cfg.get("target_modules"),
            lora_dropout=self.cfg.get("lora_dropout", 0.05),
            bias=self.cfg.get("bias", "none"),
            task_type="CAUSAL_LM",
            task_embedding_dim=self.cfg.get("task_embedding_dim", 64),
            expert_num=self.cfg.get("expert_num", 4),
        )
        return get_peft_model(model, config)


__all__ = ["MCITExpertLoraModel", "MoELoRAMethod"]
