# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 MCITlib authors
"""PCLR method registration; the stage implementation is isolated under _pclr."""
from __future__ import annotations

from ..config.schema import DataSource, TaskSpec
from .base import ContinualMethod


class PCLRMethod(ContinualMethod):
    id = 'pclr'
    aliases = ('PCLR',)

    def select_train_data(self, task: TaskSpec) -> DataSource:
        if task.data is None:
            raise ValueError('PCLR needs task data')
        return task.data
