# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 MCITlib authors
"""SAME method plugin."""
from __future__ import annotations

from ..config.schema import DataSource, TaskSpec
from ..tuners._peft_registry import tuner_model_context
from ..tuners.variants.same import SAMEConfig, SAMELinear, SAMEModel
from .base import ContinualMethod


class SAMEMethod(ContinualMethod):
    id = 'same'
    aliases = ('SAME',)
    train_entry = 'moe'

    def select_train_data(self, task: TaskSpec) -> DataSource:
        if task.data is None:
            raise ValueError('task has no data source')
        return task.data

    def build_trainable(self, model, backbone):
        from peft import get_peft_model
        config = SAMEConfig(
            r=self.cfg.get('r', 48), lora_alpha=self.cfg.get('lora_alpha', 96),
            target_modules=['gate_proj', 'up_proj', 'down_proj'],
            lora_dropout=self.cfg.get('lora_dropout', .05), bias='none', task_type='CAUSAL_LM',
            expert_num=self.cfg.get('expert_num', 6), cur_task=self.cfg.get('cur_task', 0),
        )
        with tuner_model_context('MOE_LORA_SAME', SAMEModel, SAMEConfig):
            return get_peft_model(model, config)

    def training_overrides(self):
        return {'gradient_checkpointing_kwargs': {'use_reentrant': False}}

    def load_previous(self, model, prev):
        layers = [m for m in model.modules() if isinstance(m, SAMELinear)]
        if not layers:
            raise RuntimeError('no SAME layers found after checkpoint loading')
        for layer in layers:
            layer.next_task()
        return model

    def finish_task(self, model):
        layers = [m for m in model.modules() if isinstance(m, SAMELinear)]
        if not layers:
            raise RuntimeError('no SAME layers found at task boundary')
        for layer in layers:
            layer.snapshot()
