# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 MCITlib authors
"""Octopus command, prerequisite and budget coverage."""
from __future__ import annotations

import pytest

from mcitlib.dataflow import find_gaps
from mcitlib.paths import PathResolver
from mcitlib.registry import compatibility
from mcitlib.runner import build_sequence_commands
from mcitlib.validation_batch import worker_commands


def value(command, flag):
    assert command.argv.count(flag) == 1
    return command.argv[command.argv.index(flag) + 1]


@pytest.mark.parametrize('benchmark,tasks', [('ucit', 6), ('mllm_dcl', 5)])
def test_octopus_sequence_and_sample_budget(repo_root, benchmark, tasks):
    paths = PathResolver.from_file(repo_root / 'configs/paths.example.yaml', repo_root=repo_root)
    commands = build_sequence_commands(repo_root / 'configs', 'llava', 'octopus', benchmark,
                                       paths=paths, gpus='4,5', master_port=29678)
    stage1 = [c for c in commands if c.metadata.get('octopus_stage') == 'stage1']
    stage2 = [c for c in commands if c.metadata.get('octopus_stage') == 'stage2']
    gpwc = [c for c in commands if c.metadata.get('octopus_stage') == 'gpwc']
    assert len(stage1) == tasks and len(stage2) == len(gpwc) == tasks - 1
    assert value(stage1[0], '--freeze_mm_mlp_adapter') == 'True'
    assert value(stage1[1], '--previous_task_model_path') == value(stage1[0], '--output_dir')
    assert value(stage2[0], '--previous_task_model_path') == value(stage1[1], '--output_dir')
    assert value(stage1[0], '--learning_rate') == value(stage2[0], '--learning_rate') == '0.0001'
    assert [value(c, '--num_train_epochs') for c in stage1] == (
        ['1'] * 6 if benchmark == 'ucit' else ['1', '3', '1', '2', '1'])
    assert all(value(c, '--num_train_epochs') == '1' and value(c, '--learning_rate') == '0.0001' for c in stage2)
    assert all(value(c, '--samples') == '256' and value(c, '--logical-batch') == '16' for c in gpwc)
    assert gpwc[0].env['CUDA_VISIBLE_DEVICES'] == '4'
    assert gpwc[1].argv.count('--history') == 2
    samples = [c for c in commands if c.metadata.get('octopus_stage') == 'sample']
    expected_counts = ['9600', '9600', '9600', '16000', '3200'] if benchmark == 'ucit' else ['9600'] * 4
    assert [value(c, '--count') for c in samples] == expected_counts
    assert not find_gaps(commands, (str(paths.paths.model_root), str(paths.paths.data_root)),
                         internal_roots=(str(paths.paths.ckpt_root), str(paths.results_root)))
    sampled = worker_commands(repo_root, paths, 'octopus', benchmark, '4,5', 29678)
    aliased = worker_commands(repo_root, paths, 'Octopus', benchmark, '4,5', 29678)
    assert [c.argv for c in sampled] == [c.argv for c in aliased]
    for c in sampled:
        if c.metadata.get('octopus_stage') == 'stage1':
            assert value(c, '--learning_rate') == '0.0002'
            assert value(c, '--num_train_epochs') == '1'
            assert int(value(c, '--per_device_train_batch_size')) * 2 * int(value(c, '--gradient_accumulation_steps')) == 64
        if c.metadata.get('octopus_stage') == 'stage2':
            assert value(c, '--learning_rate') == '0.0001'
            assert value(c, '--num_train_epochs') == '1'
        if c.metadata.get('octopus_stage') == 'gpwc':
            assert value(c, '--samples') == '256' and value(c, '--logical-batch') == '16'
            assert c.env['CUDA_VISIBLE_DEVICES'] == '4'
        if c.metadata.get('octopus_stage') == 'sample':
            assert value(c, '--count') == value(next(item for item in samples if item.task_id == c.task_id), '--count')
    assert not compatibility('octopus', 'internvl')[0]
    assert [value(c, '--count') for c in sampled if c.metadata.get('octopus_stage') == 'sample'] == expected_counts


def test_octopus_resolves_each_embedded_history_path(repo_root, monkeypatch):
    import json

    paths = PathResolver.from_file(repo_root / 'configs/paths.example.yaml', repo_root=repo_root)
    original = paths.resolve_templates
    def resolve(value, benchmark=None):
        resolved = original(value, benchmark)
        # Simulate a downstream resolver rewriting a whole checkpoint path.
        return '/translated/' + resolved.lstrip('/') if resolved.startswith(str(paths.paths.ckpt_root)) else resolved
    monkeypatch.setattr(paths, 'resolve_templates', resolve)
    commands = build_sequence_commands(repo_root / 'configs', 'llava', 'octopus', 'ucit',
                                       paths=paths, task_id=2, gpus='4,5')
    stage2 = next(c for c in commands if c.metadata.get('octopus_stage') == 'stage2')
    history = json.loads(stage2.env['MCITLIB_OCTOPUS_HISTORY'])
    assert len(history) == 1 and history[0].startswith('/translated/')
    assert history[0] in stage2.metadata['octopus_inputs']
