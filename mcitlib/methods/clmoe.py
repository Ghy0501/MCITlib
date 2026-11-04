# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""CL-MoE method plugin with task context and optional router diagnostics."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config.schema import DataSource, TaskSpec
from ..tuners._peft_registry import tuner_model_context
from ..tuners.variants.clmoe import MCITClMoELoraConfig, MCITClMoELoraModel
from ..utils.router_stats import RouterStats
from .base import ContinualMethod


class CLMoEMethod(ContinualMethod):
    id = "clmoe"
    aliases = ("CL-MoE", "cl-moe")
    train_entry = "moe"

    def __init__(self, cfg: dict[str, Any] | None = None, ctx: Any = None):
        self.cfg = cfg or {}
        task_id = ctx.task_id if ctx is not None and ctx.task_id is not None else self.cfg.get("task_id", 0)
        output_dir = self.cfg.get("diagnostics_dir")
        if output_dir is None and ctx is not None and ctx.output_dir is not None:
            output_dir = ctx.output_dir
        diagnostics = self.cfg.get("diagnostics", {})
        self.router_stats = RouterStats(
            task_id=int(task_id),
            output_dir=Path(output_dir or "."),
            enabled=bool(diagnostics.get("router_stats", False)),
        )

    def select_train_data(self, task: TaskSpec) -> DataSource:
        if task.data is None:
            raise ValueError("task has no data source")
        return task.data

    def build_trainable(self, model: Any, backbone: Any) -> Any:
        from peft import get_peft_model

        config = MCITClMoELoraConfig(
            r=self.cfg.get("r", 16),
            lora_alpha=self.cfg.get("lora_alpha", 32),
            target_modules=self.cfg.get("target_modules"),
            lora_dropout=self.cfg.get("lora_dropout", 0.05),
            bias=self.cfg.get("bias", "none"),
            task_type="CAUSAL_LM",
            task_embedding_dim=self.cfg.get("task_embedding_dim", 64),
            expert_num=self.cfg.get("expert_num", 4),
        )
        config._mcit_router_stats = self.router_stats
        with tuner_model_context("MOE_LORA_CLMoE", MCITClMoELoraModel):
            return get_peft_model(model, config)

    def before_task(self, ctx: Any) -> list:
        self.router_stats.task_id = int(ctx.task_id)
        return []
