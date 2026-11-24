# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Scientific profile and sample provenance checks; no GPU or training needed."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from mcitlib.config.schema import Command
from mcitlib.unified_training import apply_training_profile, load_training_profile, planned_steps


def command(path: Path, *, method='lora_ft', task=1, sequence='train', microbatch=16, accum=2, epochs=3):
    ds = path.parent / 'zero.json'
    ds.write_text(json.dumps({'scheduler': {'type': 'WarmupLR'}, 'train_batch_size': 'auto',
                              'train_micro_batch_size_per_gpu': 'auto', 'gradient_accumulation_steps': 'auto'}))
    return Command(
        argv=['deepspeed', '--include', 'localhost:4,5', '--module', 'mcitlib.backbones.llava.train_entry',
              '--deepspeed', str(ds), '--data_path', str(path), '--image_folder', str(path.parent),
              '--per_device_train_batch_size', str(microbatch),
              '--gradient_accumulation_steps', str(accum), '--num_train_epochs', str(epochs),
              '--learning_rate', '0.0001', '--mm_projector_lr', '0.00002', '--warmup_ratio', '0.03',
              '--lr_scheduler_type', 'cosine', '--model_max_length', '2048'],
        env={'MCITLIB_TRAIN_ENTRY': 'standard', 'MCITLIB_METHOD': method}, cwd='.',
        task_id=task, kind='train', sequence=sequence, metadata={},
    )


def sample(path, records):
    raw = json.dumps(records).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def profile():
    root = Path(__file__).resolve().parents[2]
    return load_training_profile(root / 'configs/training_profiles/llava_sample_unified.yaml', repo_root=root)


def test_main_train_is_unified_router_and_other_commands_unchanged(tmp_path):
    primary = tmp_path / 'train.json'
    digest = sample(primary, [{'id': i} for i in range(3000)])
    main = command(primary)
    main.metadata['effective_batch'] = 16
    router = command(primary, sequence='router', epochs=30)
    router.metadata['effective_batch'] = 16
    evaluate = Command(argv=['python', 'eval.py'], env={}, cwd='.', task_id=1, kind='eval',
                       sequence='train', metadata={})
    original = [main, router, evaluate]
    before = [item.as_dict() for item in original]
    result, audits = apply_training_profile(
        original, profile=profile(), world_size=2, expected_method='lora_ft', expected_task_ids=(1,),
        data_specs_by_index={0: {'primary_path': primary, 'primary_sha256': digest}},
    )
    assert [item.as_dict() for item in original] == before
    assert result[1].as_dict() == before[1] and result[2].as_dict() == before[2]
    assert result[1].metadata['effective_batch'] == 16
    argv = result[0].argv
    assert argv[argv.index('--per_device_train_batch_size') + 1] == '8'
    assert argv[argv.index('--gradient_accumulation_steps') + 1] == '4'
    assert argv[argv.index('--num_train_epochs') + 1] == '1'
    assert argv[argv.index('--learning_rate') + 1] == '0.0002'
    assert argv[argv.index('--mm_projector_lr') + 1] == '0.00002'
    assert result[0].metadata['training_profile'] == {
        'id': 'llava_sample_unified', 'path': 'configs/training_profiles/llava_sample_unified.yaml',
        'sha256': profile().sha256, 'role': 'main_train'}
    assert main.metadata['effective_batch'] == 16 and result[0].metadata['effective_batch'] == 64
    assert audits[0]['training_examples'] == 3000
    assert audits[0]['optimizer_steps'] == 47 and audits[0]['nominal_global_batch'] == 64
    assert audits[0]['deepspeed_configuration']['scheduler']['type'] == 'WarmupLR'
    assert audits[1]['role'] == 'auxiliary_preserved'


def test_replay_uses_verified_primary_plus_previous_records(tmp_path):
    primary, replay = tmp_path / 'train.json', tmp_path / 'replay.json'
    records = [{'id': i} for i in range(3000)]
    primary_sha = sample(primary, records)
    replay_sha = sample(replay, records + [{'id': f'previous-{i}'} for i in range(20)])
    item = command(replay, method='replay', microbatch=4)
    spec = {'primary_path': primary, 'primary_sha256': primary_sha,
            'replay_sha256': replay_sha, 'replay_previous_samples': 20, 'primary_image_root': tmp_path}
    _, audits = apply_training_profile([item], profile=profile(), world_size=2, expected_method='replay', expected_task_ids=(1,),
                                       data_specs_by_index={0: spec})
    assert audits[0]['training_data']['primary_samples'] == 3000
    assert audits[0]['training_data']['precomposed_replay_samples'] == 20
    assert audits[0]['training_examples'] == 3020
    assert audits[0]['dataloader_steps_per_rank'] == 378 and audits[0]['optimizer_steps'] == 47
    assert audits[0]['gradient_accumulation'] == 8
    sample(replay, records[1:] + records[:1] + [{'id': f'previous-{i}'} for i in range(20)])
    bad = dict(spec, replay_sha256=hashlib.sha256(replay.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match='Replay data'):
        apply_training_profile([item], profile=profile(), world_size=2, expected_method='replay', expected_task_ids=(1,),
                               data_specs_by_index={0: bad})


def test_common_batch_tail_and_unrepresentable_cases(tmp_path):
    assert planned_steps(3000, 2, 4, 8)['optimizer_steps'] == 46
    assert planned_steps(3000, 2, 4, 8)['assumed_unprocessed_example_visits_lower_bound'] == 56
    assert planned_steps(3000, 4, 4, 4)['optimizer_steps'] == 47
    primary = tmp_path / 'train.json'
    digest = sample(primary, [{}] * 3000)
    item = command(primary, microbatch=3)
    with pytest.raises(ValueError, match='cannot be represented'):
        apply_training_profile([item], profile=profile(), world_size=2, expected_method='lora_ft', expected_task_ids=(1,),
                               data_specs_by_index={0: {'primary_path': primary, 'primary_sha256': digest}})
    item = command(primary)
    with pytest.raises(ValueError, match='data evidence'):
        apply_training_profile([item], profile=profile(), world_size=2, expected_method='lora_ft', expected_task_ids=(1,), data_specs_by_index={})


def test_data_hash_world_size_and_unexpected_sequence_fail_closed(tmp_path):
    primary = tmp_path / 'train.json'
    digest = sample(primary, [{}] * 3000)
    item = command(primary)
    spec = {0: {'primary_path': primary, 'primary_sha256': digest}}
    changed = copy.deepcopy(item)
    changed.argv[changed.argv.index('--include') + 1] = 'localhost:0,1,2,3'
    with pytest.raises(ValueError, match='world size'):
        apply_training_profile([changed], profile=profile(), world_size=2, expected_method='lora_ft', expected_task_ids=(1,), data_specs_by_index=spec)
    sample(primary, [{}] * 2999)
    with pytest.raises(ValueError, match='differs from its manifest'):
        apply_training_profile([item], profile=profile(), world_size=2, expected_method='lora_ft', expected_task_ids=(1,), data_specs_by_index=spec)
    item.sequence = 'unknown'
    with pytest.raises(ValueError, match='main training task set'):
        apply_training_profile([item], profile=profile(), world_size=2, expected_method='lora_ft', expected_task_ids=(1,), data_specs_by_index=spec)


def test_replay_image_rewrite_preserves_physical_file(tmp_path):
    domain = tmp_path / 'data' / 'RS'
    (domain / 'images').mkdir(parents=True)
    (domain / 'images' / 'one.png').write_bytes(b'image')
    primary, replay = tmp_path / 'train.json', tmp_path / 'replay.json'
    rows = [{'id': i, 'image': 'images/one.png'} for i in range(3000)]
    primary_sha = sample(primary, rows)
    replay_sha = sample(replay, [{**row, 'image': 'RS/images/one.png'} for row in rows])
    item = command(replay, method='replay')
    item.argv[item.argv.index('--image_folder') + 1] = str(tmp_path / 'data')
    spec = {'primary_path': primary, 'primary_sha256': primary_sha, 'primary_image_root': domain,
            'replay_sha256': replay_sha, 'replay_previous_samples': 0}
    _, audits = apply_training_profile([item], profile=profile(), world_size=2, expected_method='replay', expected_task_ids=(1,),
                                       data_specs_by_index={0: spec})
    assert audits[0]['training_examples'] == 3000
    modified = [{**row, 'image': '../images/one.png'} for row in rows]
    spec['replay_sha256'] = sample(replay, modified)
    with pytest.raises(ValueError, match='Replay data'):
        apply_training_profile([item], profile=profile(), world_size=2, expected_method='replay', expected_task_ids=(1,),
                               data_specs_by_index={0: spec})


def test_recorded_cli_and_unreviewed_memory_fail_closed(tmp_path):
    primary = tmp_path / 'train.json'
    digest = sample(primary, [{}] * 3000)
    item = command(primary)
    spec = {0: {'primary_path': primary, 'primary_sha256': digest}}
    item.argv[item.argv.index('--warmup_ratio') + 1] = '0.07'
    with pytest.raises(ValueError, match='recorded CLI'):
        apply_training_profile([item], profile=profile(), world_size=2, expected_method='lora_ft', expected_task_ids=(1,), data_specs_by_index=spec)
    item.argv[item.argv.index('--warmup_ratio') + 1] = '0.03'
    item.argv += ['--memory_data_path', str(primary)]
    with pytest.raises(ValueError, match='memory data requires'):
        apply_training_profile([item], profile=profile(), world_size=2, expected_method='lora_ft', expected_task_ids=(1,), data_specs_by_index=spec)


def test_modalprompt_preserves_ten_epoch_method_budget(tmp_path):
    primary = tmp_path / 'train.json'
    digest = sample(primary, [{}] * 3000)
    item = command(primary, method='modalprompt', microbatch=8, accum=1, epochs=10)
    item.metadata['effective_batch'] = 32
    item.argv[item.argv.index('--include') + 1] = 'localhost:0,1,2,3'
    result, audits = apply_training_profile(
        [item], profile=profile(), world_size=4, expected_method='modalprompt', expected_task_ids=(1,),
        data_specs_by_index={0: {'primary_path': primary, 'primary_sha256': digest}})
    assert result[0].argv[result[0].argv.index('--num_train_epochs') + 1] == '10'
    assert item.metadata['effective_batch'] == 32 and result[0].metadata['effective_batch'] == 64
    assert audits[0]['resolved_epochs'] == 10
    assert audits[0]['epoch_reason'] == 'preserve_prompt_training_budget'
    assert audits[0]['planned_example_visits'] == 30000
    assert audits[0]['optimizer_steps_per_epoch'] == 47 and audits[0]['optimizer_steps'] == 470
    assert planned_steps(3000, 2, 4, 8, 10)['optimizer_steps'] == 460
    assert planned_steps(3000, 2, 4, 8, 10)['assumed_unprocessed_microbatches_total'] == 70
    bad = copy.deepcopy(item)
    bad.env['MCITLIB_METHOD'] = 'lora_ft'
    with pytest.raises(ValueError, match='method differs'):
        apply_training_profile([bad], profile=profile(), world_size=4, expected_method='modalprompt',
                               expected_task_ids=(1,),
                               data_specs_by_index={0: {'primary_path': primary, 'primary_sha256': digest}})


def test_missing_flag_values_and_deepspeed_batch_conflicts_fail(tmp_path):
    primary = tmp_path / 'train.json'
    digest = sample(primary, [{}] * 3000)
    item = command(primary)
    spec = {0: {'primary_path': primary, 'primary_sha256': digest}}
    rate = item.argv.index('--learning_rate')
    del item.argv[rate + 1]
    with pytest.raises(ValueError, match='expected one value'):
        apply_training_profile([item], profile=profile(), world_size=2, expected_method='lora_ft', expected_task_ids=(1,), data_specs_by_index=spec)
    item = command(primary)
    ds = Path(item.argv[item.argv.index('--deepspeed') + 1])
    ds.write_text(json.dumps({'train_batch_size': 16, 'train_micro_batch_size_per_gpu': 'auto',
                              'gradient_accumulation_steps': 'auto'}))
    with pytest.raises(ValueError, match='DeepSpeed train_batch_size'):
        apply_training_profile([item], profile=profile(), world_size=2, expected_method='lora_ft', expected_task_ids=(1,), data_specs_by_index=spec)
