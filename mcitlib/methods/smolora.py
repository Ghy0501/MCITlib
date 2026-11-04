# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""SMoLoRA plugin with explicit, hash-checked instruction embedding assets."""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any

from ..config.schema import DataSource, TaskSpec
from ..tuners._peft_registry import tuner_model_context
from ..tuners.mcit_separable_expert_lora import MCITSeparableExpertLoraConfig, MCITSeparableExpertLoraModel
from .base import ContinualMethod


def load_instruction_embeddings(path: str | Path, expected_sha256: str) -> list:
    """Load the v3 pickle only on explicit method invocation after verification."""
    asset = Path(path)
    actual = hashlib.sha256(asset.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"instruction embedding sha256 mismatch: {asset}: {actual} != {expected_sha256}")
    with asset.open("rb") as handle:
        embeddings = pickle.load(handle)
    return embeddings.tolist()


class SMoLoRAMethod(ContinualMethod):
    id = "smolora"
    aliases = ("SMoLoRA", "smo-lora")
    train_entry = "moe"

    def __init__(self, cfg: dict[str, Any] | None = None, ctx: Any = None):
        self.cfg = cfg or {}

    def select_train_data(self, task: TaskSpec) -> DataSource:
        if task.data is None:
            raise ValueError("task has no data source")
        return task.data

    def build_trainable(self, model: Any, backbone: Any) -> Any:
        from peft import get_peft_model

        embeddings = self.cfg.get("ins_emb")
        if isinstance(embeddings, (str, Path)):
            embeddings = load_instruction_embeddings(embeddings, self.cfg["ins_emb_sha256"])
        config = MCITSeparableExpertLoraConfig(
            r=self.cfg.get("r", 16),
            lora_alpha=self.cfg.get("lora_alpha", 32),
            target_modules=self.cfg.get("target_modules"),
            lora_dropout=self.cfg.get("lora_dropout", 0.05),
            bias=self.cfg.get("bias", "none"),
            task_type="CAUSAL_LM_SMoLoRA",
            expert_num=self.cfg.get("expert_num", 4),
            ins_type=self.cfg.get("ins_type", 0),
            ins_emb=embeddings,
        )
        with tuner_model_context("SMOLORA", MCITSeparableExpertLoraModel):
            return get_peft_model(model, config)
