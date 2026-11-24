# SPDX-License-Identifier: Apache-2.0
import copy

import pytest
import torch
from torch import nn

from mcitlib.methods._pclr.core import PoolConfig, RankPool, distillation_loss, mean_query, retained_indices


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.proj(x)


def new_pool():
    torch.manual_seed(7)
    return RankPool(Tiny(), {"layer0": ["proj"]}, PoolConfig(4, 3, new_rank=4, top_rank=3,
                         alpha=6, share_rank=2, skip_interval=2))


def test_learning_routing_and_key_loss():
    m = new_pool()
    q = torch.tensor([[1., 0, 0, 0]])
    v = torch.tensor([[1., 0, 0]])
    m.init_new_keys(q[0], v[0])
    m.set_query(q, v)
    x = torch.randn(1, 2, 4)
    out = m.base(x)
    loss = out.square().mean() + m.alignment_loss()
    loss.backward()
    _, layer = next(m.layers())
    assert layer.base.weight.grad is None
    assert layer.new_b.grad is not None and layer.new_b.grad.abs().sum() > 0
    assert m.groups['g0'].new_text.grad is not None
    assert torch.allclose(mean_query(torch.tensor([[[1., 0], [100., 100.]]]), torch.tensor([[1, 0]])), torch.tensor([[1., 0]]))


def test_two_tasks_compress_distill_and_boundary_roundtrip(tmp_path):
    model = new_pool()
    base_weights = copy.deepcopy(model.base.proj.base.state_dict())
    q1, q2 = torch.tensor([1., 0, 0, 0]), torch.tensor([0., 1, 0, 0])
    v1, v2 = torch.tensor([1., 0, 0]), torch.tensor([0., 1, 0])
    model.init_new_keys(q1, v1)
    with torch.no_grad():
        next(model.layers())[1].new_b.normal_()
    model.append_task_query()
    first = tmp_path / 'first.pt'
    model.save_boundary(first, 'learned')
    base2 = Tiny()
    base2.proj.load_state_dict(base_weights)
    model, phase = RankPool.load_boundary(base2, first)
    assert phase == 'learned'
    model.begin_task(q2, v2)
    with torch.no_grad():
        next(model.layers())[1].new_b.normal_()
    model.append_task_query()
    model.merge_new()
    assert retained_indices(4, 2).tolist() == [0, 2]
    model.compress(split_at=4)
    assert model.groups['g0'].static_text.shape[0] == 4
    model.set_query(q2[None], v2[None], part='new')
    x = torch.randn(1, 5, 4)
    model.runtime.teacher = True
    with torch.no_grad():
        teacher = model.base(x)
    model.runtime.teacher = False
    model.runtime.part = 'static'
    student = model.base(x)
    loss = distillation_loss(teacher, student, top_k=4)
    loss.backward()
    layer = next(model.layers())[1]
    assert layer.static_b.grad is not None
    assert layer.shared_b.grad is not None
    assert torch.isfinite(loss)
    before = model.base(x).detach()
    path = tmp_path / 'compressed.pt'
    model.save_boundary(path, 'integrated')
    base3 = Tiny()
    base3.proj.load_state_dict(base_weights)
    restored, phase = RankPool.load_boundary(base3, path)
    restored.set_query(q2[None], v2[None], part='static')
    assert phase == 'integrated'
    assert torch.equal(restored.task_text, model.task_text)
    assert torch.allclose(before, restored.base(x))
    restored.begin_task(q1, v1)
    assert restored.groups['g0'].new_text.shape[0] == 4
    assert restored.groups['g0'].static_text.shape[0] == 4
    assert not next(restored.layers())[1].shared_a.requires_grad


def test_reject_invalid_compression_and_checkpoint(tmp_path):
    model = new_pool()
    model.merge_new()
    with pytest.raises(ValueError):
        model.compress(0)
    path = tmp_path / 'bad.pt'
    torch.save({'format': 'wrong'}, path)
    with pytest.raises(ValueError):
        RankPool.load_boundary(Tiny(), path)


def test_llava_bridge_backward_and_generation(tmp_path):
    from types import SimpleNamespace

    from mcitlib.methods._pclr.llava import RoutedLlava

    class Vision(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_size = 3
            self.select_feature = 'patch'
        def forward(self, images):
            return images[:, :3, 0, 0].unsqueeze(1)

    class Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(9, 4)
            self.proj = nn.Linear(4, 9, bias=False)
            self.vision = Vision()
            self.config = SimpleNamespace(hidden_size=4)
        def get_model(self):
            return self
        def get_vision_tower(self):
            return self.vision
        def forward(self, input_ids, labels=None, **kwargs):
            logits = self.proj(self.embed_tokens(input_ids))
            loss = logits.square().mean() if labels is not None else None
            return SimpleNamespace(loss=loss, logits=logits)
        def generate(self, inputs, images=None, **kwargs):
            return torch.cat([inputs, self.forward(inputs).logits[:, -1].argmax(-1, keepdim=True)], -1)

    base = Base()
    config = PoolConfig(4, 3, new_rank=4, top_rank=3, alpha=6, share_rank=2, skip_interval=2)
    pool = RankPool(base, {'layer0': ['proj']}, config)
    pool.init_new_keys(torch.tensor([1., 0, 0, 0]), torch.tensor([1., 0, 0]))
    model = RoutedLlava(pool)
    ids = torch.tensor([[1, 2]])
    images = torch.ones(1, 3, 2, 2)
    result = model(input_ids=ids, labels=ids, images=images, instruction_ids=ids,
                   instruction_mask=torch.ones_like(ids), has_image=[True])
    result['loss'].backward()
    assert next(pool.layers())[1].new_b.grad.abs().sum() > 0
    pool.append_task_query()
    generated = model.generate(ids, images=images, instruction_ids=ids)
    assert generated.shape == (1, 3)
    assert base.vision.select_feature == 'patch'
    pool.begin_task(torch.tensor([0., 1, 0, 0]), torch.tensor([0., 1, 0]))
    with torch.no_grad():
        next(pool.layers())[1].new_b.normal_()
    pool.append_task_query()
    pool.merge_new()
    pool.compress(4)
    integrator = RoutedLlava(pool, stage='integrate')
    result = integrator(input_ids=ids, labels=ids, images=images, instruction_ids=ids,
                        instruction_mask=torch.ones_like(ids), has_image=[True])
    result['loss'].backward()
    assert next(pool.layers())[1].static_b.grad is not None
    assert next(pool.layers())[1].shared_b.grad is not None
    assert not pool.runtime.teacher

    # Exercise the actual Transformers 4.31 Trainer and optimizer on CPU.
    from transformers import TrainingArguments

    from mcitlib.backbones.llava.trainer import LLaVATrainer
    args = TrainingArguments(output_dir=str(tmp_path), no_cuda=True, per_device_train_batch_size=1,
                             max_steps=1, save_strategy='no', report_to=[], remove_unused_columns=False)
    args.group_by_modality_length = False
    args.is_SAT = False
    args.mm_projector_lr = None
    row = {'input_ids': ids[0], 'labels': ids[0], 'images': images[0],
           'instruction_ids': ids[0], 'instruction_mask': torch.ones_like(ids[0]), 'has_image': True}
    trainer = LLaVATrainer(model=integrator, args=args, train_dataset=[row])
    trainer.train()
    assert trainer.state.global_step == 1


def test_v4_stage_plans_for_both_benchmarks():
    from mcitlib.runner import build_sequence_commands

    for benchmark in ('ucit', 'mllm_dcl'):
        commands = build_sequence_commands('configs', 'llava', 'pclr', benchmark, task_id=4, gpus='4,5')
        stages = [c for c in commands if c.kind == 'train']
        assert [s.metadata['pclr_phase'] for s in stages] == ['learning', 'compression_integration']
        learning, integration = stages
        assert learning.sequence == 'train' and integration.sequence == 'pclr_integration'
        assert learning.argv[learning.argv.index('--include') + 1] == 'localhost:4,5'
        assert integration.argv[integration.argv.index('--include') + 1] == 'localhost:4,5'
        assert integration.argv[integration.argv.index('--previous_task_model_path') + 1] == learning.argv[learning.argv.index('--output_dir') + 1]
        assert float(integration.argv[integration.argv.index('--learning_rate') + 1]) == 5e-5
        assert integration.argv[integration.argv.index('--num_train_epochs') + 1] == '1'
        assert any('mcitlib.methods._pclr.eval_runners' in ' '.join(c.argv) for c in commands)
    final = build_sequence_commands('configs', 'llava', 'pclr', 'ucit', task_id=6)
    consolidation = next(c for c in final if c.metadata.get('pclr_phase') == 'compression_integration')
    assert consolidation.metadata['skip_interval'] == 8


def test_doctor_checks_query_rows_and_missing_history(tmp_path):
    import json

    from mcitlib.config.schema import Command
    from mcitlib.doctor import Report
    from mcitlib.methods._pclr.doctor import check_pclr

    folder = tmp_path / 'images'
    folder.mkdir()
    data = tmp_path / 'train.json'
    data.write_text(json.dumps([{'conversations': [{'from': 'human', 'value': 'describe'}],
                                'image': 'missing.jpg'}]))
    command = Command(argv=['deepspeed', '--data_path', str(data), '--image_folder', str(folder),
                            '--previous_task_model_path', str(tmp_path / 'prior'),
                            '--output_dir', str(tmp_path / 'task2')], kind='train', task_id=2,
                      env={'MCITLIB_METHOD': 'pclr', 'MCITLIB_PCLR_STAGE': 'learn', 'MCITLIB_PCLR_TASK_ID': '2',
                           'MCITLIB_PCLR_OPTIONS': json.dumps({'new_rank': 64, 'top_rank': 64,
                               'alpha': 128, 'share_rank': 32, 'skip_interval': 4})})
    report = Report()
    check_pclr(report, [command])
    assert report.failed
    assert 'missing/unsupported image' in report.checks[0].detail
    assert 'missing prior PCLR boundary' in report.checks[0].detail


def test_sample_budget_preserves_integration(tmp_path):
    import hashlib
    import json
    from pathlib import Path

    from mcitlib.runner import build_sequence_commands
    from mcitlib.unified_training import apply_training_profile, load_training_profile

    root = Path(__file__).resolve().parents[2]
    source = tmp_path / 'sample.json'
    source.write_text(json.dumps([{'id': i} for i in range(3000)]))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    plan = build_sequence_commands(root / 'configs', 'llava', 'pclr', 'ucit', task_id=4, gpus='4,5')
    train = [c for c in plan if c.kind == 'train']
    for command in train:
        command.argv[command.argv.index('--data_path') + 1] = str(source)
    profile = load_training_profile(root / 'configs/training_profiles/llava_sample_unified.yaml', repo_root=root)
    adjusted, audit = apply_training_profile(train, profile=profile, world_size=2, expected_method='pclr',
                                             expected_task_ids=(4,), data_specs_by_index={
                                                 0: {'primary_path': str(source), 'primary_sha256': digest}})
    main, integration = adjusted
    assert main.argv[main.argv.index('--learning_rate') + 1] == '0.0002'
    assert main.argv[main.argv.index('--num_train_epochs') + 1] == '1'
    assert main.argv[main.argv.index('--per_device_train_batch_size') + 1] == '8'
    assert main.argv[main.argv.index('--gradient_accumulation_steps') + 1] == '4'
    assert integration.argv == train[1].argv
    assert [item['role'] for item in audit] == ['main_train', 'auxiliary_preserved']


def test_teacher_fills_activated_rank_from_old_pool_when_new_is_smaller():
    config = PoolConfig(4, 3, new_rank=2, top_rank=3, alpha=3, share_rank=0, skip_interval=2)
    model = RankPool(Tiny(), {'layer0': ['proj']}, config, static=4, train=2)
    layer = next(model.layers())[1]
    with torch.no_grad():
        layer.base.weight.zero_()
        layer.static_a.fill_(1)
        layer.static_b.fill_(1)
        layer.new_b.zero_()
    model.task_text = torch.tensor([[1., 0, 0, 0], [0., 1, 0, 0]])
    model.task_vision = torch.tensor([[1., 0, 0], [0., 1, 0]])
    model.merge_new()
    model.compress(split_at=4)
    model.set_query(model.task_text[1:2], model.task_vision[1:2], 'new')
    model.runtime.teacher = True
    # Two newest ranks contribute zero; the remaining selected old rank contributes four.
    assert torch.allclose(model.base(torch.ones(1, 1, 4)), torch.full((1, 1, 4), 4.0))


def test_real_clip_precision_for_query_training_and_raw_generation():
    # Other method tests install process-global LLaVA class overrides. The
    # production entry point starts in a fresh process, so exercise that same
    # boundary rather than accidentally reusing another method's model class.
    import json
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    process = subprocess.run([sys.executable, '-B', '-m', 'tools.pclr_precision_probe', '--device', 'cpu'],
                             cwd=root, capture_output=True, text=True, check=True)
    result = json.loads(process.stdout.splitlines()[-1])
    assert result['compute_dtype'] == 'torch.bfloat16'
    assert result['clip_norm_dtype'] == 'torch.float32'
    assert result['query_calls'] >= 1
    assert result['training_calls'] >= 2
    assert result['generation_calls'] >= 2
    assert result['projections_with_gradient'] > 0
