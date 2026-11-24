# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 MCITlib authors
"""Full-data and sample-budget command coverage for the new methods."""
from __future__ import annotations

import pytest

from mcitlib.config.cli import expand_task
from mcitlib.paths import PathResolver
from mcitlib.registry import benchmark_supported, compatibility
from mcitlib.validation_batch import worker_commands


def option(argv, flag):
    assert argv.count(flag) == 1
    return argv[argv.index(flag) + 1]


@pytest.mark.parametrize('method', ['same', 'sigma'])
@pytest.mark.parametrize('benchmark,total', [('ucit', 6), ('mllm_dcl', 5)])
def test_full_commands_and_sample_budget(repo_root, method, benchmark, total):
    root = repo_root / 'configs'
    for task in range(1, total + 1):
        spec = expand_task(root, method, benchmark, task)
        argv = spec['train_command']['argv']
        assert int(option(argv, '--cur_task')) == task - 1
        assert int(option(argv, '--expert_num')) == total
        assert option(argv, '--learning_rate') == '0.0002'
        if method == 'same':
            assert option(argv, '--freeze_mm_mlp_adapter') == 'True'
            assert option(argv, '--text_tower') == option(argv, '--vision_tower')
        else:
            assert '--freeze_mm_mlp_adapter' not in argv
        if task > 1:
            assert f'Task{task-1}_llava_lora' in option(argv, '--previous_task_model_path')
    resolver = PathResolver.from_file(repo_root / 'configs/paths.example.yaml', repo_root=repo_root)
    commands = worker_commands(repo_root, resolver, method, benchmark, '4,5', 29671)
    trains = [command for command in commands if command.kind == 'train']
    assert len(trains) == total
    for train in trains:
        assert option(train.argv, '--learning_rate') == '0.0002'
        assert option(train.argv, '--num_train_epochs') == '1'
        assert int(option(train.argv, '--per_device_train_batch_size')) * 2 * int(option(train.argv, '--gradient_accumulation_steps')) == 64
        assert train.metadata['training_profile']['id'] == 'llava_sample_unified'
    if method == 'same':
        evals = [command for command in commands if command.kind == 'eval' and '--model-path' in command.argv]
        assert evals
        for command in evals:
            assert command.env['MCITLIB_SAME_CLIP_TOWER']
    assert not benchmark_supported(method, 'mllm_acl')
    for backbone in ('internvl', 'videollava', 'videollama2'):
        assert not compatibility(method, backbone)[0]
