# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""KeepLoRA plugin with private t437 model and SVD-initialized LoRA."""
from __future__ import annotations

from typing import Any

import torch

from ..config.schema import DataSource, TaskSpec
from ._keeplora.lora import freeze_lora_a_matrices, initialize_lora_weights_custom
from .base import ContinualMethod


class KeepLoRAMethod(ContinualMethod):
    id = "keeplora"
    aliases = ("KeepLoRA", "keep-lora")
    env = "llava-t437"
    model_override = "keeplora"
    required_capabilities = frozenset({"model_class_override"})

    def __init__(self, cfg: dict[str, Any] | None = None, ctx: Any = None):
        self.cfg = cfg or {}

    def select_train_data(self, task: TaskSpec) -> DataSource:
        if task.data is None:
            raise ValueError("task has no data source")
        return task.data

    def model_class_override(self) -> type | None:
        from ._keeplora.llava_overrides import LlavaLlamaForCausalLM

        return LlavaLlamaForCausalLM

    def build_trainable(self, model: Any, backbone: Any) -> Any:
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=self.cfg.get("r", 32),
            lora_alpha=self.cfg.get("lora_alpha", 64),
            target_modules=self.cfg.get("target_modules"),
            lora_dropout=self.cfg.get("lora_dropout", 0.05),
            bias=self.cfg.get("bias", "none"),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        gradient_path = self.cfg.get("init_lora_from_gradients")
        if gradient_path:
            gradients = torch.load(gradient_path, map_location="cpu")["gradients"]
            initialize_lora_weights_custom(model, gradients, config.r)
        if self.cfg.get("freeze_lora_A", False):
            freeze_lora_a_matrices(model)
        return model
