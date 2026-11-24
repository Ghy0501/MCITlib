# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""CPU checks for ModalPrompt's original trainer contracts."""
from __future__ import annotations

import pytest
import torch
from torch import nn
from transformers import TrainingArguments

from mcitlib.methods._modalprompt import observe
from mcitlib.methods._modalprompt.trainer import (
    ModalPromptTrainer,
    capture_origin,
)

pytestmark = pytest.mark.env("llava-modalprompt-ds126")


class TinyModal(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 3)
        self.prompt_transform = nn.Linear(3, 3)
        self.mm_projector = nn.Linear(3, 3)
        self.other = nn.Linear(3, 1)
        self.embed_tokens_mask_1d = torch.tensor([0., 0., 0., 0., 0., 0., 1., 1.])
        self.cur_task = 2
        self.seen_at_forward = []
        self.grads = []
        self.embed_tokens.weight.register_hook(lambda grad: self.grads.append(grad.detach().clone()))

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_origin_embed_tokens(self):
        self.origin_embed_tokens_weights = self.embed_tokens.weight.detach()

    def forward(self, input_ids, labels=None):
        self.seen_at_forward.append(self.embed_tokens.weight.detach().clone())
        x = self.embed_tokens(input_ids)
        prompt = self.prompt_transform(x[:, -1, :])
        self.image_cos_sim_loss = (prompt - 0.1).square().mean()
        self.text_cos_sim_loss = (prompt + 0.2).square().mean()
        logits = self.other(self.mm_projector(x.mean(dim=1)))
        return {"loss": logits.square().mean(), "logits": logits}


def make_trainer(tmp_path, model, *, steps):
    args = TrainingArguments(output_dir=str(tmp_path), max_steps=steps, per_device_train_batch_size=1,
                             learning_rate=0.02, weight_decay=0, save_strategy="no", report_to=[],
                             disable_tqdm=True, logging_strategy="no", no_cuda=True)
    args.mm_projector_lr = 0.002
    args.group_by_modality_length = False
    data = [{"input_ids": torch.tensor([0, 1, 6]), "labels": torch.tensor(0)} for _ in range(steps)]
    origin = capture_origin(model)
    # DS frees the pre-partition storage by replacing Parameter.data. Keep a
    # full CPU parameter here, with the same storage separation for this test.
    model.embed_tokens.weight.data = model.embed_tokens.weight.data.clone()
    return ModalPromptTrainer(model=model, args=args, train_dataset=data, origin=origin,
                              allow_cpu_standin=True)


def test_real_trainer_loss_learning_rates_and_reset_order(tmp_path):
    torch.manual_seed(4)
    model = TinyModal()
    initial = model.embed_tokens.weight.detach().clone()
    trainer = make_trainer(tmp_path, model, steps=3)
    trainer.create_optimizer()
    for name, param in model.named_parameters():
        group = next(group for group in trainer.optimizer.param_groups if any(p is param for p in group["params"]))
        assert group["lr"] == (0.002 if any(s in name for s in ("prompt_transform", "embed_tokens", "mm_projector")) else 0.02)
    sample = {"input_ids": torch.tensor([[0, 1, 6]]), "labels": torch.tensor([0])}
    total, outputs = trainer.compute_loss(model, sample, return_outputs=True)
    assert torch.allclose(total, outputs["loss"] + model.image_cos_sim_loss + model.text_cos_sim_loss)
    total.backward()
    assert model.prompt_transform.weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    model.seen_at_forward.clear()
    model.grads.clear()
    trainer.train()
    assert len(model.seen_at_forward) == 3
    assert all(torch.equal(rows[:6], initial[:6]) for rows in model.seen_at_forward)
    assert not torch.equal(model.seen_at_forward[-1][6:], initial[6:])
    assert all(grad[:2].abs().sum() > 0 for grad in model.grads)
    # The v3 reset is before each training step; the final optimizer update
    # can therefore leave base rows changed in the saved checkpoint.
    assert not torch.equal(model.embed_tokens.weight[:2], initial[:2])


def test_capture_is_independent_after_storage_replacement(tmp_path):
    model = TinyModal()
    trainer = make_trainer(tmp_path, model, steps=1)
    with torch.no_grad():
        model.embed_tokens.weight[0].add_(3)
    assert not torch.equal(trainer._modal_origin[0], model.embed_tokens.weight[0])


def test_capture_already_partitioned_embedding_with_gather(monkeypatch):
    import deepspeed
    model = TinyModal()
    weight = model.embed_tokens.weight
    expected = weight.detach().clone()
    weight.ds_id = 1
    weight.data = torch.empty(0)
    calls = []

    class Gather:
        def __init__(self, params):
            assert params == [weight]

        def __enter__(self):
            calls.append("gather")
            weight.data = expected.clone()

        def __exit__(self, *args):
            weight.data = torch.empty(0)
            calls.append("partition")

    monkeypatch.setattr(deepspeed.zero, "GatheredParameters", Gather)
    captured = capture_origin(model)
    assert calls == ["gather", "partition"]
    assert weight.numel() == 0
    assert torch.equal(captured, expected)


def test_official_zero_api_changes_master_only_before_three_steps(tmp_path, monkeypatch):
    """Public DeepSpeed calls + synthetic master; not real ZeRO partitioning."""
    import deepspeed
    assert deepspeed.__version__ == "0.12.6"
    torch.manual_seed(21)
    model = TinyModal()
    trainer = make_trainer(tmp_path, model, steps=3)
    monkeypatch.setattr(observe, "repo_root", lambda: tmp_path)
    monkeypatch.setenv("MCITLIB_MODAL_OBSERVE_DIR", str(tmp_path / "logs/observed"))
    monkeypatch.setenv("RANK", "0")
    weight = model.embed_tokens.weight
    origin = trainer._modal_origin.clone()

    class SyntheticMaster:
        def __init__(self):
            from types import SimpleNamespace

            self.value = weight.detach().clone().float()
            self.setter_calls = 0
            self.optimizer = SimpleNamespace(param_groups=[{"lr": 2e-4}])
            self.fp16_groups = [list(model.parameters())]
            self.sub_group_to_group_id = {0: 0}

        def get_full_hp_param(self, param):
            assert param is weight
            return self.value.clone()

        def set_full_hp_param(self, value, param):
            assert param is weight
            self.value.copy_(value)
            self.setter_calls += 1

    master = SyntheticMaster()
    weight.ds_id = 1
    weight._z3_optimizer = master
    batch = {"input_ids": torch.tensor([[0, 1, 6]]), "labels": torch.tensor([0])}
    forwards = []
    for step in range(3):
        trainer.state.global_step = step
        before_lp = weight.detach().clone()
        before_hp = master.value.clone()
        model.zero_grad(set_to_none=True)
        trainer.training_step(model, batch)
        forwards.append(model.seen_at_forward[-1].clone())
        assert master.setter_calls == step + 1
        assert torch.equal(weight.detach(), before_lp)  # No low-precision write here.
        assert torch.equal(master.value[:6], origin[:6])
        assert torch.equal(master.value[6:], before_hp[6:])
        assert weight.grad[:2].abs().sum() > 0
        # Model the normal optimizer update/copy-back after the training step.
        with torch.no_grad():
            master.value.add_(weight.grad.float(), alpha=-0.01)
            weight.copy_(master.value)
    assert torch.equal(forwards[0][:6], origin[:6])
    assert not torch.equal(forwards[1][:2], origin[:2])  # v3 does not sync LP on reset.
    assert not torch.equal(weight[:2], origin[:2])  # No extra reset at final save.
    import json

    records = [json.loads(line) for line in (tmp_path / "logs/observed/task2/rank0.jsonl").read_text().splitlines()]
    assert [(row["event"], row["step"]) for row in records] == [
        (event, step) for step in range(3) for event in ("master_restore", "loss")
    ]
    for row in records[::2]:
        assert row["safe_get"].endswith("safe_get_full_fp32_param")
        assert row["safe_set"].endswith("safe_set_full_fp32_param")
        assert row["master_shape"] == row["origin_shape"] == [8, 3]
        assert row["active_rows"] == 2
        assert row["groups"][0]["lr"] == 2e-4
        assert any("embed_tokens" in name for name in row["groups"][0]["parameters"])
        assert any("prompt_transform" in name for name in row["groups"][0]["parameters"])
        assert any("mm_projector" in name for name in row["groups"][0]["parameters"])


def test_observation_rejects_output_outside_project_logs(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(observe, "repo_root", lambda: project)
    monkeypatch.setenv("MCITLIB_MODAL_OBSERVE_DIR", str(tmp_path / "outside"))
    with pytest.raises(ValueError, match="must be inside"):
        observe.write_event(1, 0, "loss")
    assert not (tmp_path / "outside").exists()


def test_deepspeed_config_bypasses_trainer_groups_and_uses_one_lr():
    """Exercise HF's actual DeepSpeed decision and the DS warmup schedule."""
    from types import SimpleNamespace

    from accelerate.utils import DummyOptim, DummyScheduler
    from deepspeed.runtime.lr_schedules import WarmupLR
    from transformers.deepspeed import deepspeed_optim_sched

    model = TinyModal()
    parameters = list(model.parameters())
    config = SimpleNamespace(config={"optimizer": {}, "scheduler": {}})

    def forbidden():
        raise AssertionError("configured DeepSpeed optimizer must bypass trainer.create_optimizer")

    trainer = SimpleNamespace(create_optimizer=forbidden)
    args = SimpleNamespace(adafactor=False)
    optimizer, scheduler = deepspeed_optim_sched(trainer, config, args, 3, parameters)
    assert isinstance(optimizer, DummyOptim)
    assert isinstance(scheduler, DummyScheduler)
    assert len(optimizer.params) == len(parameters)
    assert all(first is second for first, second in zip(optimizer.params, parameters))
    assert all(not isinstance(item, dict) for item in optimizer.params)

    # The actual fixed DS config passes one scalar max LR to WarmupLR.
    cpu_optimizer = torch.optim.AdamW(parameters, lr=2e-4)
    warmup = WarmupLR(cpu_optimizer, warmup_min_lr=0, warmup_max_lr=2e-4, warmup_num_steps=1)
    observed = []
    for _ in range(3):
        warmup.step()
        observed.append(cpu_optimizer.param_groups[0]["lr"])
    assert warmup.warmup_num_steps == 2
    assert observed == pytest.approx([0, 2e-4, 2e-4])
