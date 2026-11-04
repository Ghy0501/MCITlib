# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""MR-LoRA plugin: standard LoRA training plus separate router stages."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..config.schema import DataSource, Stage, TaskSpec
from ._mrlora import build_router_user_message
from .base import ContinualMethod


class MRLoRAMethod(ContinualMethod):
    id = "mrlora"
    aliases = ("MR-LoRA", "mr-lora")
    train_entry = "standard"
    required_capabilities = frozenset({"router_data"})

    def __init__(self, cfg: dict[str, Any] | None = None, ctx: Any = None):
        self.cfg = cfg or {}

    def select_train_data(self, task: TaskSpec) -> DataSource:
        if task.data is None:
            raise ValueError("task has no data source")
        source = task.data
        if not task.router:
            return source
        if source.router_path is None or source.router_folder is None:
            raise ValueError(f"MR-LoRA router requires router_path and router_folder: {source}")
        return replace(source, train_path=source.router_path, train_folder=source.router_folder)

    def build_router_user_message(self, raw_question: str) -> str:
        return build_router_user_message(self.cfg["benchmark"], raw_question)

    def extra_stages(self) -> list[Stage]:
        return [Stage(name="router")]
