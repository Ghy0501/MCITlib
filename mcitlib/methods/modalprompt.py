# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""ModalPrompt plugin with private prompt model and separate inference."""
from __future__ import annotations

from typing import Any

from ..config.schema import DataSource, TaskSpec
from ._modalprompt.llava_overrides import MCITModalPrompt
from .base import ContinualMethod


class ModalPromptMethod(ContinualMethod):
    id = "modalprompt"
    aliases = ("ModalPrompt",)
    train_entry = "standard"
    deepspeed_config = "zero3_offload"
    separate_train_and_infer = True
    model_override = "modalprompt"
    required_capabilities = frozenset({"text_tower", "clip_features"})

    def __init__(self, cfg: dict[str, Any] | None = None, ctx: Any = None):
        self.cfg = cfg or {}
        self.cur_task = int(self.cfg.get("cur_task", 1))

    def select_train_data(self, task: TaskSpec) -> DataSource:
        if task.data is None:
            raise ValueError("task has no data source")
        return task.data

    def model_class_override(self) -> type | None:
        return MCITModalPrompt

    def build_trainable(self, model: Any, backbone: Any) -> Any:
        if not isinstance(model, MCITModalPrompt):
            raise TypeError("ModalPrompt requires the MCITModalPrompt model override")
        # Prompt transforms are initialized by the training entry after the
        # tokenizer is available, matching v3's required operation order.
        return model

    def before_task(self, ctx: Any) -> list:
        self.cur_task = int(self.cfg.get("cur_task", ctx.task_id))
        return []

    def prepare_inference(self, model: Any, ctx: Any) -> Any:
        if hasattr(model, "set_cur_task"):
            model.set_cur_task(int(ctx.task_id))
        return model
