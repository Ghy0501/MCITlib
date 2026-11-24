# SPDX-License-Identifier: Apache-2.0
"""Continuation of the HF-owned scheduler alongside a simulated ZeRO optimizer."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
get_cosine_schedule_with_warmup = pytest.importorskip("transformers.optimization").get_cosine_schedule_with_warmup

from mcitlib.methods._dgg.scheduler_checkpoint import SIDECAR, DGGSchedulerCheckpoint, capture, restore  # noqa: E402


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.projector = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
        self.bias = torch.nn.Parameter(torch.tensor([5.0, 6.0]))


def make_trainer(output, *, warmup=0, swap=False):
    model = Model()
    groups = [model.lora, model.projector, model.bias]
    if swap:
        groups[1], groups[2] = groups[2], groups[1]
    optimizer = torch.optim.AdamW([{'params': [groups[0]]}, {'params': []},
        {'params': [groups[1]], 'lr': 2e-5}, {'params': [groups[2]], 'lr': 2e-5}],
        lr=2e-4, weight_decay=0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup, num_training_steps=2)
    # The pinned engine removes empty groups after the Trainer creates LambdaLR.
    optimizer.param_groups[:] = [group for group in optimizer.param_groups if group['params']]
    zero = SimpleNamespace(optimizer=optimizer, bit16_groups=[[p] for p in groups])
    engine = SimpleNamespace(module=model, optimizer=zero, lr_scheduler=None, global_steps=0)
    args = SimpleNamespace(output_dir=str(output), lr_scheduler_type='cosine',
                           get_warmup_steps=lambda total: warmup)
    trainer = SimpleNamespace(model_wrapped=engine, lr_scheduler=scheduler, args=args,
                              _trial=None, _get_output_dir=lambda trial: str(output))
    state = SimpleNamespace(global_step=0, max_steps=2, is_world_process_zero=True)
    return trainer, state


def step(trainer, state, scale):
    for parameter in trainer.model_wrapped.module.parameters():
        parameter.grad = torch.full_like(parameter, scale)
    trainer.model_wrapped.optimizer.optimizer.step()
    trainer.lr_scheduler.step()
    trainer.model_wrapped.optimizer.optimizer.zero_grad()
    state.global_step += 1
    trainer.model_wrapped.global_steps += 1


def equal_nested(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_nested(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal_nested(x, y) for x, y in zip(a, b))
    return a == b


def test_real_multigroup_cosine_sidecar_continues_optimizer_and_scheduler(tmp_path):
    original, state = make_trainer(tmp_path)
    step(original, state, .01)
    checkpoint = tmp_path / 'checkpoint-1'
    checkpoint.mkdir()
    callback = DGGSchedulerCheckpoint(original)
    callback.on_save(original.args, state, object())
    assert (checkpoint / SIDECAR).is_file()
    saved = capture(original, state)
    assert saved['scheduler_state']['base_lrs'] == [2e-4, 2e-4, 2e-5, 2e-5]
    assert [g['options']['lr'] for g in saved['optimizer_groups']] == [1e-4, 1e-4, 1e-5]
    weights = deepcopy(original.model_wrapped.module.state_dict())
    optimizer = deepcopy(original.model_wrapped.optimizer.optimizer.state_dict())

    resumed, resumed_state = make_trainer(tmp_path)
    resumed.model_wrapped.module.load_state_dict(weights)
    resumed.model_wrapped.optimizer.optimizer.load_state_dict(optimizer)
    resumed_state.global_step = resumed.model_wrapped.global_steps = 1
    assert resumed.lr_scheduler.last_epoch == 0
    DGGSchedulerCheckpoint(resumed, str(checkpoint)).on_train_begin(resumed.args, resumed_state, object())
    assert resumed.lr_scheduler.last_epoch == 1
    assert capture(resumed, resumed_state) == saved

    step(original, state, .02)
    step(resumed, resumed_state, .02)
    assert equal_nested(original.model_wrapped.module.state_dict(), resumed.model_wrapped.module.state_dict())
    assert equal_nested(original.model_wrapped.optimizer.optimizer.state_dict(),
                        resumed.model_wrapped.optimizer.optimizer.state_dict())
    assert capture(original, state) == capture(resumed, resumed_state)
    assert resumed.lr_scheduler.get_last_lr() == [0.0, 0.0, 0.0]


def test_sidecar_missing_corrupt_and_no_unwritten_checkpoint(tmp_path):
    trainer, state = make_trainer(tmp_path)
    checkpoint = tmp_path / 'checkpoint-1'
    checkpoint.mkdir()
    with pytest.raises(FileNotFoundError, match='missing its scheduler sidecar'):
        DGGSchedulerCheckpoint(trainer, str(checkpoint)).on_train_begin(trainer.args, state, object())
    (checkpoint / SIDECAR).write_text('{bad json')
    with pytest.raises(ValueError, match='cannot be read'):
        DGGSchedulerCheckpoint(trainer, str(checkpoint)).on_train_begin(trainer.args, state, object())
    with pytest.raises(FileNotFoundError, match='native checkpoint was not saved'):
        DGGSchedulerCheckpoint(trainer).on_save(trainer.args, state, object())


def test_incompatible_version_groups_rates_steps_and_schedule_are_rejected(tmp_path):
    trainer, state = make_trainer(tmp_path)
    step(trainer, state, .01)
    saved = capture(trainer, state)
    for key, bad in [('version', 2), ('trainer_step', 2), ('deepspeed_step', 2),
                     ('optimizer_groups', []), ('signature', {})]:
        payload = deepcopy(saved)
        payload[key] = bad
        with pytest.raises(ValueError, match='DGG scheduler sidecar'):
            restore(trainer, state, payload)
    payload = deepcopy(saved)
    payload['scheduler_state'] = None
    with pytest.raises(ValueError, match='missing its scheduler state'):
        restore(trainer, state, payload)

    changed, changed_state = make_trainer(tmp_path, swap=True)
    step(changed, changed_state, .01)
    with pytest.raises(ValueError, match='optimizer_groups'):
        restore(changed, changed_state, saved)
    changed, changed_state = make_trainer(tmp_path, warmup=1)
    step(changed, changed_state, .01)
    with pytest.raises(ValueError, match='signature'):
        restore(changed, changed_state, saved)
    trainer.model_wrapped.optimizer.optimizer.param_groups[0]['lr'] = .7
    with pytest.raises(ValueError, match='optimizer_groups'):
        restore(trainer, state, saved)
