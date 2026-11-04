# SPDX-License-Identifier: Apache-2.0 AND MIT
# Copyright (c) 2025 Jinpeng Chen
# Copyright (c) 2025-2026 MCITlib authors
"""SEFE plugin with t437 model override and key-element/merge steps."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config.schema import Command, DataSource, Step, TaskSpec
from .base import ContinualMethod


class SEFEMethod(ContinualMethod):
    id = "sefe"
    aliases = ("RegLoRA", "reglora", "SEFE")
    env = "llava-t437"
    model_override = "sefe"
    required_capabilities = frozenset({"model_class_override"})

    def __init__(self, cfg: dict[str, Any] | None = None, ctx: Any = None):
        self.cfg = cfg or {}

    def select_train_data(self, task: TaskSpec) -> DataSource:
        if task.data is None:
            raise ValueError("task has no data source")
        return task.data

    def model_class_override(self) -> type | None:
        from ._sefe.llava_overrides import LlavaLlamaForCausalLM
        return LlavaLlamaForCausalLM

    def build_trainable(self, model: Any, backbone: Any) -> Any:
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=self.cfg.get("r", 16),
            lora_alpha=self.cfg.get("lora_alpha", 32),
            target_modules=self.cfg.get("target_modules"),
            lora_dropout=self.cfg.get("lora_dropout", 0.05),
            bias=self.cfg.get("bias", "none"),
            task_type="CAUSAL_LM",
        )
        return get_peft_model(model, config)

    def before_task(self, ctx: Any) -> list[Step]:
        if ctx.task_id <= 1:
            return []
        train = self.cfg.get("train_config", self.cfg)
        argv = [
            "python", "mcitlib/methods/_sefe/key_elements.py",
            str(train.get("key_id", ctx.task_id - 2)),
            str(train["key_path"]),
            str(train["base_model"]),
        ]
        return [Step("key_elements", Command(argv=argv, kind="extra", task_id=ctx.task_id))]

    def after_task(self, ctx: Any) -> list[Step]:
        train = self.cfg.get("train_config", self.cfg)
        output_dir = Path(ctx.output_dir or train["output_dir"])
        previous = train.get("previous_model")
        if previous is None:
            previous = self.cfg["model_config"]["model_name"]
        argv = [
            "python", "mcitlib/methods/_sefe/merge.py",
            "--model-path", str(output_dir),
            "--model-base", str(previous),
            "--save-model-path", f"{output_dir}_merged",
        ]
        return [Step("merge", Command(argv=argv, kind="extra", task_id=ctx.task_id))]
