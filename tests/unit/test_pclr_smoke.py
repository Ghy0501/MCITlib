# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 MCITlib authors
"""The bounded PCLR smoke must not remap physical launcher GPU IDs."""
from pathlib import Path

import pytest

from tools.pclr_smoke import configure_validation_train, integration_evidence, launch_env, learning_evidence


@pytest.mark.parametrize('gpus,first', [('0,1', '0'), ('4,5', '4')])
def test_launcher_clears_inherited_visibility_and_pins_first_gpu(tmp_path, monkeypatch, gpus, first):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', gpus)
    root = Path(__file__).resolve().parents[2]
    train = {'argv': ['deepspeed', '--include', 'localhost:' + gpus], 'env': {}}
    env = launch_env(train, root, tmp_path, gpus)
    assert 'CUDA_VISIBLE_DEVICES' not in env
    assert env['PYTHONDONTWRITEBYTECODE'] == '1'
    assert env['HF_HUB_OFFLINE'] == env['TRANSFORMERS_OFFLINE'] == '1'
    for key in ('HF_HOME', 'XDG_CACHE_HOME', 'TRITON_CACHE_DIR', 'TORCH_EXTENSIONS_DIR'):
        assert Path(env[key]).is_relative_to(tmp_path / 'cache')
    infer = {'argv': ['python', '-m', 'mcitlib.backbones.llava.eval_runners.entry'],
             'env': {'CUDA_VISIBLE_DEVICES': first}}
    assert launch_env(infer, root, tmp_path, gpus)['CUDA_VISIBLE_DEVICES'] == first


@pytest.mark.parametrize('argv', [[], ['deepspeed'], ['deepspeed', '--include', 'localhost:0,1'],
                                   ['deepspeed', '--include', 'localhost:4,5', '--include', 'localhost:4,5']])
def test_launcher_rejects_wrong_or_ambiguous_gpu_ids(tmp_path, argv):
    with pytest.raises(ValueError):
        launch_env({'argv': argv, 'env': {}}, Path(__file__).resolve().parents[2], tmp_path, '4,5')


def test_inference_rejects_wrong_gpu(tmp_path):
    with pytest.raises(ValueError):
        launch_env({'argv': ['python'], 'env': {'CUDA_VISIBLE_DEVICES': '5'}},
                   Path(__file__).resolve().parents[2], tmp_path, '4,5')


def test_validation_steps_have_zero_warmup_and_preserve_learning_rate():
    from mcitlib.config.schema import Command

    command = Command(argv=['deepspeed', '--warmup_ratio', '0.03', '--learning_rate', '0.0002'])
    with pytest.raises(ValueError, match='at least two'):
        configure_validation_train(command, 1)
    configure_validation_train(command, 2)
    def option(flag):
        assert command.argv.count(flag) == 1
        return command.argv[command.argv.index(flag) + 1]
    assert option('--warmup_ratio') == '0'
    assert option('--max_steps') == '2'
    assert option('--learning_rate') == '0.0002'
    assert option('--logging_steps') == '1'


def test_learning_evidence_rejects_zero_updates_and_incomplete_steps(tmp_path):
    import json

    import torch

    directory = tmp_path / 'learn'
    directory.mkdir()
    (directory / 'trainer_state.json').write_text(json.dumps({'global_step': 2}))
    weights = {'layer.new_b': torch.zeros(4, 2)}
    torch.save({'phase': 'learned', 'state': weights}, directory / 'pclr.pt')
    with pytest.raises(ValueError, match='no observable new B update'):
        learning_evidence(directory, 2)
    weights['layer.new_b'][0, 0] = 0.125
    torch.save({'phase': 'learned', 'state': weights}, directory / 'pclr.pt')
    evidence = learning_evidence(directory, 2)
    assert evidence['new_b_nonzero'] == 1 and evidence['global_step'] == 2
    with pytest.raises(ValueError, match='expected optimizer steps'):
        learning_evidence(directory, 3)


def test_integration_evidence_compares_exact_retained_banks(tmp_path):
    import json

    import torch

    learned, integrated = tmp_path / 'learned', tmp_path / 'integrated'
    learned.mkdir()
    integrated.mkdir()
    old = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    new = old + 8
    teacher = {'phase': 'learned', 'state': {'layer.static_b': old, 'layer.new_b': new}}
    compressed = torch.cat((old[[0, 2]], new[[0, 2]]))
    student = {'phase': 'integrated', 'config': {'skip_interval': 2},
               'state': {'layer.static_b': compressed.clone(), 'layer.shared_b': torch.zeros(2, 2)}}
    torch.save(teacher, learned / 'pclr.pt')
    (integrated / 'trainer_state.json').write_text(json.dumps({'global_step': 2}))
    torch.save(student, integrated / 'pclr.pt')
    with pytest.raises(ValueError, match='no observable student or shared B update'):
        integration_evidence(learned, integrated, 2)
    student['state']['layer.static_b'][1, 0] += 0.25
    torch.save(student, integrated / 'pclr.pt')
    evidence = integration_evidence(learned, integrated, 2)
    assert evidence['compressed_b_changed'] == 1 and evidence['compressed_b_max_abs_change'] == .25
    assert evidence['shared_b_nonzero'] == 0 and evidence['skip_interval'] == 2
