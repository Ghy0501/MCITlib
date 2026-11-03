# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Reference LoRA-FT method plugin."""
from __future__ import annotations

from ..config.schema import DataSource, TaskSpec
from .base import ContinualMethod


class LoraFTMethod(ContinualMethod):
    id = "lora_ft"
    aliases = ("LoRA-FT", "lora-ft")

    def select_train_data(self, task: TaskSpec) -> DataSource:
        if task.data is None:
            raise ValueError("task has no data source")
        return task.data
