# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 MCITlib authors
"""Cross-method scope and full/sample budget regressions."""
import pytest

from mcitlib.config.cli import expand_task
from mcitlib.paths import PathResolver
from mcitlib.registry import benchmark_supported, compatibility
from mcitlib.validation_batch import worker_commands


def option(argv, name):
    assert argv.count(name) == 1
    return argv[argv.index(name) + 1]


@pytest.mark.parametrize('method', ['pclr', 'same', 'sigma', 'dgg', 'octopus'])
def test_scope_and_full_dcl_epochs(repo_root, method):
    assert compatibility(method, 'llava')[0]
    assert not compatibility(method, 'internvl')[0]
    assert all(compatibility(method, backbone)[0] for backbone in ('videollava', 'videollama2'))
    assert all(benchmark_supported(method, name) for name in ('ucit', 'mllm_dcl'))
    assert all(not benchmark_supported(method, name) for name in ('mllm_acl', 'general'))
    epochs = [int(option(expand_task(repo_root / 'configs', method, 'mllm_dcl', task)
                         ['train_command']['argv'], '--num_train_epochs')) for task in range(1, 6)]
    assert epochs == [1, 3, 1, 2, 1]


@pytest.mark.parametrize('benchmark,total', [('ucit', 6), ('mllm_dcl', 5)])
def test_pclr_sample_budget_retains_separate_consolidation(repo_root, benchmark, total):
    resolver = PathResolver.from_file(repo_root / 'configs/paths.example.yaml', repo_root=repo_root)
    commands = worker_commands(repo_root, resolver, 'pclr', benchmark, '4,5', 29671)
    main = [c for c in commands if c.kind == 'train' and c.sequence == 'train']
    assert len(main) == total
    for command in main:
        assert option(command.argv, '--learning_rate') == '0.0002'
        assert option(command.argv, '--num_train_epochs') == '1'
        assert int(option(command.argv, '--per_device_train_batch_size')) * 2 * int(
            option(command.argv, '--gradient_accumulation_steps')) == 64
    integration = [c for c in commands if c.kind == 'train' and c.sequence == 'pclr_integration']
    assert [c.task_id for c in integration] == list(range(4, total + 1))
    for command in integration:
        assert float(option(command.argv, '--learning_rate')) == 5e-5
        assert option(command.argv, '--num_train_epochs') == '1'
        assert command.metadata['pclr_phase'] == 'compression_integration'
        assert 'training_profile' not in command.metadata
