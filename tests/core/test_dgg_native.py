# SPDX-License-Identifier: Apache-2.0
"""Plan, replay, and state tests for the isolated DGG candidate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from mcitlib.config.schema import Command
from mcitlib.dataflow import find_gaps
from mcitlib.paths import PathResolver
from mcitlib.registry import compatibility, matrix
from mcitlib.runner import build_sequence_commands
from mcitlib.unified_training import apply_training_profile, load_training_profile

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from mcitlib.methods._dgg.replay import prepare  # noqa: E402


def value(argv, flag):
    return argv[argv.index(flag) + 1]


def resolver(tmp_path, repo_root):
    path = tmp_path / 'paths.yaml'
    path.write_text(yaml.safe_dump({name: str(tmp_path / name) for name in
                                   ('model_root', 'data_root', 'ckpt_root', 'results_root')}))
    return PathResolver.from_file(path, repo_root=repo_root)


@pytest.mark.parametrize('benchmark', ['ucit', 'mllm_dcl'])
def test_dgg_plan_orders_replay_and_never_loads_previous_as_initial_weights(repo_root, tmp_path, benchmark):
    paths = resolver(tmp_path, repo_root)
    commands = build_sequence_commands(repo_root / 'configs', 'llava', 'dgg', benchmark, paths=paths, gpus='4')
    trains = [c for c in commands if c.kind == 'train']
    assert len(trains) == (6 if benchmark == 'ucit' else 5)
    expected_epochs = ['1'] * 6 if benchmark == 'ucit' else ['1', '3', '1', '2', '1']
    assert [value(train.argv, '--num_train_epochs') for train in trains] == expected_epochs
    for index, train in enumerate(trains, 1):
        assert value(train.argv, '--include') == 'localhost:4'
        assert 'CUDA_VISIBLE_DEVICES' not in train.env
        assert value(train.argv, '--lora_r') == '48' and value(train.argv, '--lora_alpha') == '96'
        assert value(train.argv, '--save_strategy') == 'epoch'
        assert '--previous_task_model_path' not in train.argv
        if index == 1:
            assert value(train.argv, '--dgg_history_model_path') == 'none'
        else:
            preceding = commands[commands.index(train) - 1]
            assert 'mcitlib.methods._dgg.replay' in preceding.argv
            assert preceding.argv.count('--data_path') == index
            assert value(preceding.argv, '--output') == value(train.argv, '--data_path')
            assert value(train.argv, '--dgg_history_model_path') == value(trains[index - 2].argv, '--output_dir')
    gaps = find_gaps(commands, (str(tmp_path / 'model_root'), str(tmp_path / 'data_root')),
                     internal_roots=(str(tmp_path / 'ckpt_root'), str(tmp_path / 'results_root')))
    assert not gaps
    ev = build_sequence_commands(repo_root / 'configs', 'llava', 'dgg', benchmark, paths=paths,
                                 task_id=2, only_eval=True, gpus='4')
    chunks = [c for c in ev if '--chunk-idx' in c.argv]
    assert chunks and all(c.env['CUDA_VISIBLE_DEVICES'] == '4' and 'parallel_group' not in c.metadata for c in chunks)


def test_dgg_explicit_scope(monkeypatch):
    from dataclasses import replace

    from mcitlib import registry
    assert compatibility('dgg', 'llava')[0]
    assert not compatibility('dgg', 'internvl')[0]
    monkeypatch.setattr(registry, 'BACKBONES', tuple(replace(b, v4_supported=True) if b.id == 'internvl' else b
                                                      for b in registry.BACKBONES))
    assert not compatibility('dgg', 'internvl')[0]  # remains unsupported if that backbone is later enabled
    assert all(not row['supported'] for row in matrix() if row['method'] == 'dgg' and row['benchmark'] == 'mllm_acl')


def test_dgg_replay_execution_and_sample_profile(repo_root, tmp_path):
    image_root = tmp_path / 'images'
    image_root.mkdir()
    (image_root / 'one.jpg').write_bytes(b'1')
    (image_root / 'two.jpg').write_bytes(b'2')
    current = [{'image': 'one.jpg', 'conversations': [{'from': 'human', 'value': 'question'},
                                                        {'from': 'gpt', 'value': 'answer'}]}]
    previous = [{'image': 'two.jpg', 'conversations': [{'from': 'human', 'value': 'old question'},
                                                         {'from': 'gpt', 'value': 'old answer'}]}]
    current_file, previous_file, merged_file = (tmp_path / name for name in ('current.json', 'previous.json', 'merged.json'))
    current_file.write_text(json.dumps(current))
    previous_file.write_text(json.dumps(previous))
    report = prepare([str(current_file), str(previous_file)], [str(image_root), str(image_root)], str(merged_file), 2000, 1993)
    assert report['replay_by_task'] == [1]
    merged = json.loads(merged_file.read_text())
    assert merged[1]['conversations'] == previous[0]['conversations']
    assert merged[1]['image'] == str(image_root / 'two.jpg')
    profile = load_training_profile(repo_root / 'configs/training_profiles/dgg_sample_unified.yaml', repo_root=repo_root)
    from dataclasses import replace
    profile = replace(profile, primary_train_samples_per_task=1)  # tiny fixture; same batch policy
    ds = repo_root / 'configs/deepspeed/zero2.json'
    command = Command(argv=['deepspeed', '--include', 'localhost:4', '--module', 'mcitlib.backbones.llava.train_entry',
                            '--deepspeed', str(ds), '--data_path', str(merged_file), '--image_folder', str(image_root),
                            '--warmup_ratio', '0.03', '--lr_scheduler_type', 'cosine', '--model_max_length', '2048',
                            '--mm_projector_lr', '0.00002', '--learning_rate', '0.0002', '--num_train_epochs', '1',
                            '--per_device_train_batch_size', '48', '--gradient_accumulation_steps', '1'],
                      env={'MCITLIB_METHOD': 'dgg', 'MCITLIB_TRAIN_ENTRY': 'standard'}, kind='train', task_id=2)
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    revised, audit = apply_training_profile([command], profile=profile, world_size=1, expected_method='dgg',
                    expected_task_ids=(2,), data_specs_by_index={0: {'primary_path': str(current_file),
                    'primary_sha256': sha(current_file), 'primary_image_root': str(image_root),
                    'replay_sha256': sha(merged_file), 'replay_previous_samples': 1}})
    assert value(revised[0].argv, '--per_device_train_batch_size') == '8'
    assert value(revised[0].argv, '--gradient_accumulation_steps') == '8'
    assert audit[0]['training_data']['precomposed_replay_samples'] == 1


def test_dgg_anchor_load_and_resume_guard(tmp_path):
    from mcitlib.methods._dgg.runtime import guard_reference, load_guidance

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block = torch.nn.Module()
            self.block.lora_A = torch.nn.ModuleDict({'default': torch.nn.Linear(2, 1, bias=False)})
            self.block.lora_B = torch.nn.ModuleDict({'default': torch.nn.Linear(1, 2, bias=False)})

    model = Model()
    directory, output = tmp_path / 'history', tmp_path / 'output'
    directory.mkdir()
    state = {name.replace('.default.', '.'): p.detach().clone() + 1 for name, p in model.named_parameters()}
    torch.save(state, directory / 'adapter_model.bin')
    torch.save({}, directory / 'non_lora_trainables.bin')
    data = tmp_path / 'data.json'
    data.write_text('[]')
    guidance = load_guidance(model, str(directory), .1)
    assert set(guidance.history) == set(dict(model.named_parameters()))
    guard_reference(str(output), str(data), str(directory), .1, None)
    guard_reference(str(output), str(data), str(directory), .1, 'checkpoint-1')
    data.write_text('[1]')
    with pytest.raises(ValueError, match='unchanged'):
        guard_reference(str(output), str(data), str(directory), .1, 'checkpoint-1')


def test_dgg_doctor_checks_original_qa_and_stage_dependencies(repo_root, tmp_path, monkeypatch):
    import sys
    import types

    from mcitlib.doctor import FAIL, OK, Report, check_dgg_plan

    class FakeEngine:
        def backward(self, loss):
            self.optimizer.backward(loss)
            self.allreduce_gradients()

    fake_ds = types.ModuleType('deepspeed')
    fake_ds.__version__ = '0.9.5'
    fake_runtime = types.ModuleType('deepspeed.runtime')
    fake_engine = types.ModuleType('deepspeed.runtime.engine')
    fake_engine.DeepSpeedEngine = FakeEngine
    monkeypatch.setitem(sys.modules, 'deepspeed', fake_ds)
    monkeypatch.setitem(sys.modules, 'deepspeed.runtime', fake_runtime)
    monkeypatch.setitem(sys.modules, 'deepspeed.runtime.engine', fake_engine)
    paths = resolver(tmp_path, repo_root)
    data = tmp_path / 'data_root' / 'UCIT'
    for name, filename in [('ImageNet-R', 'train.json'), ('ArxivQA', 'train_4w.json')]:
        directory = data / name
        directory.mkdir(parents=True)
        (directory / filename).write_text(json.dumps([{'conversations': [{'from': 'human', 'value': 'question'},
                                                                            {'from': 'gpt', 'value': 'answer'}]}]))
    commands = build_sequence_commands(repo_root / 'configs', 'llava', 'dgg', 'ucit', paths=paths, gpus='4')
    commands = [c for c in commands if c.task_id in (1, 2)]
    report = Report()
    check_dgg_plan(report, commands)
    replay_check = next(c for c in report.checks if c.name == 'DGG replay and history')
    assert replay_check.status == OK
    boundary = next(c for c in report.checks if c.name == 'DGG DeepSpeed callback boundary')
    assert boundary.status == OK
    (data / 'ImageNet-R' / 'train.json').write_text(json.dumps([{'router_label': 'A'}]))
    report = Report()
    check_dgg_plan(report, commands)
    assert next(c for c in report.checks if c.name == 'DGG replay and history').status == FAIL


def test_dgg_real_peft_adapter_reference_and_backward(tmp_path):
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    from mcitlib.methods._dgg.runtime import load_guidance

    cfg = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=2)
    model = get_peft_model(LlamaForCausalLM(cfg), LoraConfig(r=2, lora_alpha=4,
                              target_modules=['q_proj', 'v_proj'], lora_dropout=0.0, task_type='CAUSAL_LM'))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'lora_B' in name:
                parameter.fill_(.05)
    model.save_pretrained(tmp_path)
    torch.save({}, tmp_path / 'non_lora_trainables.bin')
    guidance = load_guidance(model, str(tmp_path), 1.0)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(.01)
    tokens = torch.tensor([[1, 2, 3, 4]])
    model(input_ids=tokens, labels=tokens).loss.backward()
    before = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    assert before and guidance.apply(model.named_parameters())
    assert any(not torch.equal(p.grad, before[name]) for name, p in model.named_parameters() if name in before)


def test_dgg_sample_manifest_is_verified_and_replay_is_rechecked(repo_root, tmp_path):
    from mcitlib.methods._dgg.runtime import guard_reference
    from mcitlib.methods._dgg.sample import apply_dgg_sample_profile
    from mcitlib.registry import get_benchmark

    paths = resolver(tmp_path, repo_root)
    current_dir = tmp_path / 'data_root' / 'UCIT' / 'ArxivQA'
    old_dir = tmp_path / 'data_root' / 'UCIT' / 'ImageNet-R'
    current_dir.mkdir(parents=True)
    old_dir.mkdir(parents=True)
    row = {'conversations': [{'from': 'human', 'value': 'question'}, {'from': 'gpt', 'value': 'answer'}]}
    current_file, old_file = current_dir / 'train_4w.json', old_dir / 'train.json'
    current_file.write_text(json.dumps([row] * 3000))
    old_file.write_text(json.dumps([row] * 3000))
    digest = hashlib.sha256(current_file.read_bytes()).hexdigest()
    manifest = {'tasks': [{'benchmark': 'ucit', 'name': name, 'splits': {'train': {
        'sample_count': 3000, 'output_sha256': digest}}} for name in get_benchmark('ucit').tasks]}
    manifest_path = tmp_path / 'sample.json'
    manifest_path.write_text(json.dumps(manifest))
    commands = build_sequence_commands(repo_root / 'configs', 'llava', 'dgg', 'ucit', paths=paths, task_id=2, gpus='4')
    apply_dgg_sample_profile(commands, 'ucit', repo_root, manifest_path)
    replay, train = commands
    assert value(train.argv, '--learning_rate') == '0.0002'
    assert value(train.argv, '--num_train_epochs') == '1'
    assert value(train.argv, '--per_device_train_batch_size') == '8'
    assert value(train.argv, '--gradient_accumulation_steps') == '8'
    assert replay.argv.count('--expected-sha256') == 2
    expected = [replay.argv[i + 1] for i, flag in enumerate(replay.argv[:-1]) if flag == '--expected-sha256']
    output, replay_manifest = value(replay.argv, '--output'), value(replay.argv, '--manifest')
    prepare([str(current_file), str(old_file)], [str(tmp_path / 'data_root/UCIT')] * 2,
            output, 2000, 1993, replay_manifest, expected)
    guard_reference(str(tmp_path / 'run'), output, None, .1, None, replay_manifest=replay_manifest)
    Path(output).write_text('[]')
    with pytest.raises(ValueError, match='preparation manifest'):
        guard_reference(str(tmp_path / 'run'), output, None, .1, None, replay_manifest=replay_manifest)
    old_file.write_text('[]')
    with pytest.raises(ValueError, match='sample manifest'):
        prepare([str(current_file), str(old_file)], [str(tmp_path / 'data_root/UCIT')] * 2,
                output, 2000, 1993, replay_manifest, expected)


def test_dgg_dcl_sample_profile_overrides_full_data_compensation(repo_root, tmp_path):
    from mcitlib.methods._dgg.sample import apply_dgg_sample_profile
    from mcitlib.registry import get_benchmark

    commands = build_sequence_commands(repo_root / 'configs', 'llava', 'dgg', 'mllm_dcl',
                                       paths=resolver(tmp_path, repo_root), gpus='4')
    raw = json.dumps([{}] * 3000).encode()
    digest = hashlib.sha256(raw).hexdigest()
    for command in commands:
        if command.kind == 'train' and command.task_id == 1:
            inputs = [value(command.argv, '--data_path')]
        elif 'mcitlib.methods._dgg.replay' in command.argv:
            inputs = [command.argv[i + 1] for i, flag in enumerate(command.argv[:-1]) if flag == '--data_path']
        else:
            continue
        for path in inputs:
            file = Path(path)
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(raw)
    manifest = {'tasks': [{'benchmark': 'mllm_dcl', 'name': name, 'splits': {'train': {
        'sample_count': 3000, 'output_sha256': digest}}} for name in get_benchmark('mllm_dcl').tasks]}
    manifest_path = tmp_path / 'dcl-sample.json'
    manifest_path.write_text(json.dumps(manifest))
    apply_dgg_sample_profile(commands, 'mllm_dcl', repo_root, manifest_path)
    trains = [command for command in commands if command.kind == 'train']
    assert [value(train.argv, '--num_train_epochs') for train in trains] == ['1'] * 5


@pytest.mark.parametrize('benchmark', ['ucit', 'mllm_dcl'])
def test_dgg_single_gpu_queue_verifies_samples_and_interleaves_eval(repo_root, tmp_path, benchmark):
    import shutil

    from mcitlib.registry import get_benchmark
    from mcitlib.validation_batch import worker_commands

    project = tmp_path / 'project'
    project.mkdir()
    shutil.copytree(repo_root / 'configs', project / 'configs')
    paths = resolver(tmp_path, project)
    ordinary = build_sequence_commands(project / 'configs', 'llava', 'dgg', benchmark,
                                       paths=paths, gpus='4')
    raw = json.dumps([{}] * 3000).encode()
    digest = hashlib.sha256(raw).hexdigest()
    for command in ordinary:
        if command.kind == 'train' and command.task_id == 1:
            files = [value(command.argv, '--data_path')]
        elif 'mcitlib.methods._dgg.replay' in command.argv:
            files = [command.argv[i + 1] for i, flag in enumerate(command.argv[:-1]) if flag == '--data_path']
        else:
            continue
        for name in files:
            file = Path(name)
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(raw)
    manifest = {'tasks': [{'benchmark': benchmark, 'name': name, 'splits': {'train': {
        'sample_count': 3000, 'output_sha256': digest}}} for name in get_benchmark(benchmark).tasks]}
    (project / 'configs/validation_3k_1k/manifest.json').write_text(json.dumps(manifest))
    commands = worker_commands(project, paths, 'dgg', benchmark, '4', 29681)
    trains = [c for c in commands if c.kind == 'train']
    assert len(trains) == len(get_benchmark(benchmark).tasks)
    for index, train in enumerate(trains, 1):
        assert value(train.argv, '--include') == 'localhost:4'
        assert value(train.argv, '--num_train_epochs') == '1'
        assert value(train.argv, '--per_device_train_batch_size') == value(train.argv, '--gradient_accumulation_steps') == '8'
        assert train.metadata['training_profile']['id'] == 'dgg_sample_unified'
        after = commands[commands.index(train) + 1:]
        assert any(c.kind == 'eval' and c.task_id == index for c in after)
    evaluations = [c for c in commands if c.kind == 'eval' and '--chunk-idx' in c.argv]
    assert evaluations and all(c.env['CUDA_VISIBLE_DEVICES'] == '4' and 'parallel_group' not in c.metadata
                               for c in evaluations)
    with pytest.raises(ValueError, match='exactly one GPU'):
        worker_commands(project, paths, 'dgg', benchmark, '4,5', 29681)


def test_dgg_ucit_eval_matches_released_generation_options(repo_root, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from mcitlib.backbones.llava.overrides import import_llava
    import_llava()
    from mcitlib.methods._dgg.eval_runners import model_others
    from mcitlib.runner import build_sequence_commands

    evaluation = build_sequence_commands(repo_root / 'configs', 'llava', 'dgg', 'ucit', task_id=1, only_eval=True)
    inference = next(c for c in evaluation if '--chunk-idx' in c.argv)
    assert 'mcitlib.methods._dgg.eval_runners.model_others' in inference.argv
    calls = {}

    class Tokenizer:
        def batch_decode(self, tokens, skip_special_tokens):
            calls['decoded'] = tokens.tolist()
            return ['answer']

    class Model:
        config = SimpleNamespace(mm_use_im_start_end=False)
        def generate(self, ids, **kwargs):
            calls.update(kwargs)
            return torch.cat((ids, torch.tensor([[8, 9]])), dim=1)

    monkeypatch.setattr(model_others, 'disable_torch_init', lambda: None)
    monkeypatch.setattr(model_others, 'load_pretrained_model', lambda *args: (Tokenizer(), Model(), None, 2048))
    monkeypatch.setattr(model_others, 'Questions', lambda *args: object())
    monkeypatch.setattr(model_others, 'DataLoader', lambda *args, **kwargs: [(torch.tensor([[1, 2]]), torch.ones(1, 3))])
    monkeypatch.setattr(model_others, 'tqdm', lambda iterator, **kwargs: iterator)
    original_to = torch.Tensor.to
    def cpu_to(tensor, *args, **kwargs):
        if kwargs.get('device') == 'cuda':
            kwargs['device'] = 'cpu'
        return original_to(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'to', cpu_to)
    questions, output = tmp_path / 'questions.json', tmp_path / 'answers.jsonl'
    questions.write_text(json.dumps([{'question_id': 1, 'image': 'a.jpg', 'text': 'q'}]))
    args = SimpleNamespace(model_path='task_llava_lora', model_base='base', question_file=str(questions),
        image_folder=str(tmp_path), answers_file=str(output), conv_mode='vicuna_v1', num_chunks=1, chunk_idx=0,
        temperature=0, top_p=None, num_beams=1, max_new_tokens=128)
    model_others.eval_model(args)
    assert calls['max_new_tokens'] == 128 and calls['do_sample'] is False
    assert calls['images'].dtype == torch.float16
    assert calls['decoded'] == [[8, 9]]
    assert json.loads(output.read_text())['text'] == 'answer'
    torch.testing.assert_close(model_others.continuation(torch.tensor([[8, 9]]), torch.tensor([[1, 2, 3]])), torch.tensor([[8, 9]]))


def test_dgg_trainer_callback_adds_guidance_before_reduction_and_restores():
    from types import SimpleNamespace

    from mcitlib.methods._dgg.core import GradientGuidance
    from mcitlib.methods._dgg.runtime import DGGCallback

    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    events = []
    class Optimizer:
        def backward(self, loss):
            events.append('backward')
            loss.backward()
    class Engine:
        world_size = 1
        module = model
        optimizer = Optimizer()
        def zero_optimization_stage(self): return 2
        def bfloat16_enabled(self): return True
        def backward(self, loss):
            self.optimizer.backward(loss)
            events.append(('reduce', model.weight.grad.item()))
    engine = Engine()
    callback = DGGCallback(SimpleNamespace(model_wrapped=engine), GradientGuidance({'weight': torch.zeros(1, 1)}, 1.0))
    callback.on_train_begin(None, None, None)
    engine.backward((model.weight * 2).sum())
    assert events == ['backward', ('reduce', 3.0)]
    assert callback.evidence() == {'backward_calls': 1, 'bernoulli_successes': 1,
                                    'gradient_tensors_seen': 1, 'gradient_tensors_modified': 1}
    callback.close()
    model.zero_grad()
    engine.backward((model.weight * 2).sum())
    assert events[-1] == ('reduce', 2.0)
