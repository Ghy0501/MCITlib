# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Replay data-selection plugin; model and training code stay shared."""
from __future__ import annotations

from dataclasses import replace

from ..config.schema import DataSource, TaskSpec
from .base import ContinualMethod


class ReplayMethod(ContinualMethod):
    id = "replay"
    aliases = ("Replay",)

    def select_train_data(self, task: TaskSpec) -> DataSource:
        if task.data is None:
            raise ValueError("task has no data source")
        source = task.data
        if source.replay_path is None or source.router_folder is None:
            raise ValueError(f"Replay requires replay_path and router_folder: {source}")
        return replace(source, train_path=source.replay_path, train_folder=source.router_folder)
