# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 MCITlib authors
"""SiGMA method plugin."""
from __future__ import annotations

from ..config.schema import DataSource, TaskSpec
from ..tuners._peft_registry import tuner_model_context
from ..tuners.variants.sigma import SiGMAConfig, SiGMAModel
from .base import ContinualMethod


class SiGMAMethod(ContinualMethod):
    id = 'sigma'
    aliases = ('SiGMA',)
    train_entry = 'moe'

    def select_train_data(self, task: TaskSpec) -> DataSource:
        if task.data is None:
            raise ValueError('task has no data source')
        return task.data

    def build_trainable(self, model, backbone):
        from peft import get_peft_model
        config = SiGMAConfig(
            r=self.cfg.get('r', 16), lora_alpha=self.cfg.get('lora_alpha', 32),
            target_modules=self.cfg.get('target_modules'), lora_dropout=self.cfg.get('lora_dropout', .05),
            bias='none', task_type='CAUSAL_LM', expert_num=self.cfg.get('expert_num', 6),
            cur_task=self.cfg.get('cur_task', 0),
        )
        with tuner_model_context('MOE_LORA_SIGMA', SiGMAModel, SiGMAConfig):
            return get_peft_model(model, config)
