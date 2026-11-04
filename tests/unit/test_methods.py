# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Method-plugin behavior tests independent of model downloads."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from mcitlib.config.schema import DataSource, TaskSpec
from mcitlib.methods.hide import HiDeMethod
from mcitlib.methods.moelora import MoELoRAMethod
from mcitlib.methods.olora import OLoRAMethod
from mcitlib.methods.replay import ReplayMethod


def task_with_data() -> TaskSpec:
    source = DataSource(
        train_path="train.json",
        replay_path="replay.json",
        router_path="router.json",
        test_path="test.json",
        train_folder="train-images",
        router_folder="router-images",
        test_folder="test-images",
    )
    return TaskSpec("ucit", 1, "ImageNet-R", "model", "data", "train", data=source)


def test_replay_only_changes_data_selection() -> None:
    selected = ReplayMethod().select_train_data(task_with_data())
    assert selected.train_path == "replay.json"
    assert selected.train_folder == "router-images"
    assert selected.test_path == "test.json"
    assert selected.test_folder == "test-images"


def test_olora_training_override_and_zero_loss() -> None:
    method = OLoRAMethod({"cur_task": 0})
    assert method.training_overrides() == {"gradient_checkpointing_kwargs": {"use_reentrant": False}}
    base = torch.tensor(2.0, requires_grad=True)
    assert method.extra_loss(None, None, None, base).item() == 0.0


def test_olora_orthogonal_loss_matches_formula() -> None:
    method = OLoRAMethod({"cur_task": 1})
    current = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    previous = torch.nn.Parameter(torch.tensor([[2.0, 0.0], [1.0, 1.0]]))
    model = SimpleNamespace(named_parameters=lambda: [
        ("default.loraA.1.weight", current),
        ("default.loraA.0.weight", previous),
    ])
    loss = method.extra_loss(model, None, None, torch.tensor(0.0))
    expected = 0.05 * torch.abs(torch.mm(current, previous.detach().T)).sum()
    assert torch.equal(loss, expected)


def test_hide_declares_private_model_override() -> None:
    method = HiDeMethod()
    assert method.model_override == "hide"
    assert method.model_class_override() is not None
    assert method.required_capabilities == frozenset({"text_tower", "clip_features", "layer_index"})


def test_moelora_plugin_builds_shared_expert_tuner() -> None:
    from unittest.mock import patch

    task = task_with_data()
    method = MoELoRAMethod({"r": 8, "lora_alpha": 16, "expert_num": 4, "target_modules": ["q_proj"]})
    assert method.train_entry == "moe"
    assert method.select_train_data(task) is task.data
    with patch("peft.get_peft_model", side_effect=lambda model, config: config) as get_model:
        config = method.build_trainable(SimpleNamespace(), None)
    assert config.r == 8
    assert config.lora_alpha == 16
    assert config.expert_num == 4
    assert config.peft_type == "MOE_LORA_CoIN"
    get_model.assert_called_once()
