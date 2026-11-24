# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 MCITlib authors
"""Octopus continual-learning method declaration."""
from __future__ import annotations

from ..config.schema import DataSource, TaskSpec
from .base import ContinualMethod


class OctopusMethod(ContinualMethod):
    id = 'octopus'
    aliases = ('Octopus',)
    model_override = 'octopus'

    def select_train_data(self, task: TaskSpec) -> DataSource:
        if task.data is None:
            raise ValueError('task has no training data')
        return task.data
