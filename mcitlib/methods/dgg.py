# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Dynamic Gradient Guidance with task-specific replay preparation."""
from __future__ import annotations

from dataclasses import replace

from ..config.schema import DataSource, TaskSpec
from .base import ContinualMethod


class DGGMethod(ContinualMethod):
    id = "dgg"
    aliases = ("DGG",)
    separate_train_and_infer = True

    def select_train_data(self, task: TaskSpec) -> DataSource:
        if task.data is None:
            raise ValueError("DGG task has no data source")
        if task.task_id == 1:
            return task.data
        path = self.cfg["train_config"].get("dgg_replay_path")
        if not path:
            raise ValueError("DGG task requires a prepared replay path")
        return replace(task.data, train_path=path)
