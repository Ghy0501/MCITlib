# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""DISCO method plugin: block-diagonal expert fusion and private LLaVA model."""
from __future__ import annotations

from typing import Any

from ..config.schema import DataSource, TaskSpec
from ..tuners._peft_registry import tuner_model_context
from ..tuners.variants.disco import MCITDiscoLoraConfig, MCITDiscoLoraModel
from ._disco.llava_overrides import LlavaLlamaForCausalLM
from .base import ContinualMethod


class DISCOMethod(ContinualMethod):
    id = "disco"
    aliases = ("DISCO",)
    train_entry = "moe"
    model_override = "disco"
    required_capabilities = frozenset({"text_tower", "clip_features", "layer_index"})

    def __init__(self, cfg: dict[str, Any] | None = None, ctx: Any = None):
        self.cfg = cfg or {}
        self.cur_task = int(self.cfg.get("cur_task", 0))

    def select_train_data(self, task: TaskSpec) -> DataSource:
        if task.data is None:
            raise ValueError("task has no data source")
        return task.data

    def model_class_override(self) -> type | None:
        return LlavaLlamaForCausalLM

    def build_trainable(self, model: Any, backbone: Any) -> Any:
        from peft import get_peft_model

        config = MCITDiscoLoraConfig(
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
        with tuner_model_context("MOE_LORA_CoIN", MCITDiscoLoraModel, MCITDiscoLoraConfig):
            return get_peft_model(model, config)

    def before_task(self, ctx: Any) -> list:
        self.cur_task = int(self.cfg.get("cur_task", ctx.task_id - 1))
        return []
