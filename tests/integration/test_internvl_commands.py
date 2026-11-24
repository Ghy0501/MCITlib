# SPDX-License-Identifier: Apache-2.0
"""InternVL LoRA command construction from reviewed task data."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mcitlib.cli import build_parser
from mcitlib.config.loader import load_layered_config
from mcitlib.registry import compatibility
from mcitlib.runner import build_sequence_commands

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / 'configs'


def value(command, flag):
    return command.argv[command.argv.index(flag) + 1]


@pytest.mark.parametrize('benchmark,expected_rank,expected_full_epochs', [('ucit', '16', '1'),
                                                                         ('mllm_dcl', '32', '3')])
def test_lora_task2_full_preserves_internvl_config(benchmark, expected_rank, expected_full_epochs):
    commands = build_sequence_commands(CONFIGS, 'internvl', 'lora_ft', benchmark, task_id=2)
    train = next(command for command in commands if command.kind == 'train')
    assert value(train, '--lora_r') == expected_rank
    assert value(train, '--num_train_epochs') == expected_full_epochs
    assert value(train, '--learning_rate') == '0.0001'
    assert value(train, '--vision_tower') == '${model_root}/InternViT-6B-224px'
    assert value(train, '--mm_vision_select_layer') == '-4'
    assert value(train, '--previous_task_model_path').endswith('/Task1_internvl_lora')
    assert value(train, '--output_dir').endswith('/Task2_internvl_lora')
    assert '--mm_projector_lr' not in train.argv
    assert train.env['MCITLIB_BACKBONE'] == 'internvl'
    assert train.metadata['internvl_source_sha256']['train_config']
    evals = [command for command in commands if '--model-path' in command.argv]
    assert evals and all(value(command, '--model-path').endswith('/Task2_internvl_lora') for command in evals)
    assert all('InternVL-Chat-ViT-6B-Vicuna-7B' in value(command, '--model-base') for command in evals)


@pytest.mark.parametrize('benchmark', ['ucit', 'mllm_dcl'])
def test_four_gpu_sample_budget_is_recorded_and_full_is_unchanged(benchmark):
    full_before = load_layered_config(CONFIGS, 'internvl', benchmark, 'lora_ft', task_id=2)['train_config']
    commands = build_sequence_commands(CONFIGS, 'internvl', 'lora_ft', benchmark, task_id=2,
                                       execution_profile='four_gpu_validation', gpus='0,1,2,3')
    train = next(command for command in commands if command.kind == 'train')
    metadata = train.metadata['training_profile']
    assert value(train, '--include') == 'localhost:0,1,2,3'
    assert value(train, '--per_device_train_batch_size') == '1'
    assert value(train, '--gradient_accumulation_steps') == '16'
    assert value(train, '--learning_rate') == '0.0002' and value(train, '--num_train_epochs') == '1'
    assert metadata['role'] == 'main_train' and metadata['global_batch_size'] == 64
    assert metadata['backbone'] == 'internvl' and len(metadata['sha256']) == 64
    assert metadata['world_size'] == 4 and metadata['microbatch'] == 1
    assert metadata['gradient_accumulation'] == 16
    from mcitlib.summary import recorded_scientific_profile

    warnings = []
    evidence = recorded_scientific_profile(train.metadata, ROOT, train.sequence, warnings)
    assert evidence['status'] == 'file_hash_matches_at_collection' and not warnings
    assert load_layered_config(CONFIGS, 'internvl', benchmark, 'lora_ft', task_id=2)['train_config'] == full_before
    gpu_ids = {command.env['CUDA_VISIBLE_DEVICES'] for command in commands if 'CUDA_VISIBLE_DEVICES' in command.env}
    assert gpu_ids <= {'0', '1', '2', '3', '0,1,2,3'}


def test_internvl_rejects_other_gpus_and_unimplemented_benchmarks():
    assert compatibility('lora_ft', 'internvl')[0]
    assert compatibility('disco', 'internvl')[0]
    with pytest.raises(ValueError, match='restricted to GPUs'):
        build_sequence_commands(CONFIGS, 'internvl', 'lora_ft', 'ucit', task_id=1, gpus='4,5')
    with pytest.raises(ValueError, match='currently supports UCIT and MLLM-DCL'):
        build_sequence_commands(CONFIGS, 'internvl', 'disco', 'mllm_acl', task_id=1)


@pytest.mark.parametrize('method,folder', [('replay', 'Replay'), ('moelora', 'MoELoRA'),
                                           ('olora', 'OLoRA'), ('clmoe', 'CL-MoE'),
                                           ('disco', 'DISCO'), ('hide', 'HiDe'), ('smolora', 'SMoLoRA')])
@pytest.mark.parametrize('benchmark,source_bench,last', [('ucit', 'UCIT', 6), ('mllm_dcl', 'MLLM-DCL', 5)])
@pytest.mark.parametrize('which', ['first', 'second', 'last'])
def test_internvl_expert_and_replay_plans_follow_task_config(method, folder, benchmark, source_bench, last, which):
    from mcitlib.backbones.llava.train_context import training_task_id

    task = {'first': 1, 'second': 2, 'last': last}[which]
    source = json.loads((CONFIGS / 'train_configs' / folder / 'InternVL' / source_bench /
                         'train' / f'task{task}.json').read_text())
    commands = build_sequence_commands(CONFIGS, 'internvl', method, benchmark, task_id=task)
    train = next(command for command in commands if command.kind == 'train')
    assert int(value(train, '--lora_r')) == source['rank']
    assert float(value(train, '--learning_rate')) == source['lr']
    assert int(value(train, '--num_train_epochs')) == source['epoch']
    assert int(value(train, '--per_device_train_batch_size')) == source['batch_size']
    assert int(value(train, '--gradient_accumulation_steps')) == source['grad_acc']
    assert value(train, '--output_dir').endswith(source['output_dir'].split('/your_ckpts_path/')[-1])
    assert value(train, '--vision_tower').endswith('/InternViT-6B-224px')
    assert value(train, '--mm_vision_select_layer') == '-4' and '--mm_projector_lr' not in train.argv
    if task > 1:
        assert value(train, '--previous_task_model_path').endswith(source['previous_model'].split('/your_ckpts_path/')[-1])
    else:
        assert value(train, '--pretrain_mm_mlp_adapter').endswith('/InternVL-Chat-ViT-6B-Vicuna-7B/mm_projector.bin')
    for key, flag in (('expert_num', '--expert_num'), ('cur_task', '--cur_task'), ('task', '--task'),
                      ('ins_type', '--ins_type')):
        if key in source:
            assert value(train, flag) == str(source[key])
    if method == 'olora':
        assert value(train, '--gradient_checkpointing') == 'True'
    if method == 'clmoe':
        assert training_task_id('clmoe', 0, value(train, '--task'), config_root=CONFIGS) == task
    if method == 'replay' and task > 1:
        assert value(train, '--data_path').endswith('/replay.json')
    if method in {'disco', 'hide'}:
        assert value(train, '--text_tower').endswith('/clip-vit-large-patch14-336')
        assert all(value(command, '--text-tower').endswith('/clip-vit-large-patch14-336')
                   for command in commands if '--text-tower' in command.argv)
    if method == 'smolora':
        import hashlib

        expected = {'ucit': '2a4595386ca010a104d3dd2109f2da52c6cabe562aa5d6304e3a469880d4bc37',
                    'mllm_dcl': 'ea251206f6ef970a2207e150810e416c134246c3904cffac1b2c91abfcf01721'}
        asset = ROOT / f'assets/smolora/ins_emb_multi_{"ucit" if benchmark == "ucit" else "dcl"}.pkl'
        assert hashlib.sha256(asset.read_bytes()).hexdigest() == expected[benchmark]
        assert value(train, '--ins_emb').endswith(asset.name)
    eval_models = [value(command, '--model-path') for command in commands if '--model-path' in command.argv]
    assert eval_models and all(path.endswith(f'/Task{task}_internvl_lora') for path in eval_models)


@pytest.mark.parametrize('benchmark,source_bench,n', [('ucit', 'UCIT', 6), ('mllm_dcl', 'MLLM-DCL', 5)])
def test_internvl_mrlora_independent_experts_router_and_full_matrix(benchmark, source_bench, n):
    options = dict(validation_fixes=True, execution_profile='four_gpu_validation', gpus='0,1,2,3')
    experts = build_sequence_commands(CONFIGS, 'internvl', 'mrlora', benchmark, **options)
    routers = build_sequence_commands(CONFIGS, 'internvl', 'mrlora', benchmark, sequence='router', **options)
    expert_trains = [command for command in experts if command.kind == 'train']
    router_trains = [command for command in routers if command.kind == 'train']
    assert len(expert_trains) == len(router_trains) == n
    for index, (expert, router) in enumerate(zip(expert_trains, router_trains), 1):
        assert '--previous_task_model_path' not in expert.argv + router.argv
        assert value(expert, '--model_name_or_path').endswith('/InternVL-Chat-ViT-6B-Vicuna-7B')
        assert value(router, '--model_name_or_path') == value(expert, '--model_name_or_path')
        assert value(expert, '--output_dir').endswith(f'/Task{index}_internvl_lora')
        assert value(router, '--output_dir').endswith(f'/Router_task{index}_internvl_lora')
        assert value(router, '--data_path').endswith('/router.json')
        assert value(router, '--num_train_epochs') == '30'
        source = json.loads((CONFIGS / 'train_configs/MR-LoRA/InternVL' / source_bench /
                             'train_router' / f'task{index}.json').read_text())
        metadata = router.metadata['auxiliary_training_profile']
        assert float(value(router, '--learning_rate')) == source['lr']
        assert metadata['microbatch'] == 1
        assert metadata['global_batch_size'] == source['gpu_num'] * source['batch_size'] * source['grad_acc']
        assert metadata['role'] == 'auxiliary_preserved' and metadata['epochs'] == '30'
        assert '/train_router/' in load_layered_config(CONFIGS, 'internvl', benchmark, 'mrlora', task_id=index)[
            'router']['benchmarks'][benchmark]['tasks'][f'task_{index}']['source']['train_config']
    fixes = [command.metadata.get('validation_fix') for command in experts]
    expert_key = f'mrlora_{"ucit" if benchmark == "ucit" else "dcl"}_expert_inference'
    assert fixes.count(expert_key) == n * n * 4
    route_key = f'mrlora_{"ucit" if benchmark == "ucit" else "dcl"}_router_selection'
    selections = [command for command in routers if command.metadata.get('validation_fix') == route_key]
    assert len(selections) == n * (n + 1) // 2
    for selection in selections:
        inputs = selection.argv[selection.argv.index('--inputs') + 1:]
        assert len(inputs) == n and all('/internvl/mrlora/' in path for path in inputs)
        assert all(f'MR-LoRA-task{index}/merge.jsonl' in path for index, path in enumerate(inputs, 1))


@pytest.mark.parametrize('method', ['sefe', 'keeplora'])
@pytest.mark.parametrize('benchmark,source_bench', [('ucit', 'UCIT'), ('mllm_dcl', 'MLLM-DCL')])
@pytest.mark.parametrize('task', [1, 2])
def test_internvl_merged_base_regularization_and_gradient_pipeline(method, benchmark, source_bench, task):
    folder = 'RegLoRA' if method == 'sefe' else 'KeepLoRA'
    source = json.loads((CONFIGS / 'train_configs' / folder / 'InternVL' / source_bench /
                         'train' / f'task{task}.json').read_text())
    commands = build_sequence_commands(CONFIGS, 'internvl', method, benchmark, task_id=task)
    train = next(command for command in commands if command.kind == 'train')
    adapter = value(train, '--output_dir')
    assert adapter.endswith(f'/Task{task}_internvl_lora')
    assert float(value(train, '--learning_rate')) == source['lr']
    assert int(value(train, '--num_train_epochs')) == source['epoch']
    expected_base = (f'/Task{task - 1}_internvl_lora_merged' if task > 1
                     else '/InternVL-Chat-ViT-6B-Vicuna-7B')
    assert value(train, '--model_name_or_path').endswith(expected_base)
    merge = next(command for command in commands if f'mcitlib/methods/_{method}/merge.py' in command.argv)
    assert value(merge, '--model-path') == adapter
    assert value(merge, '--model-base').endswith(expected_base)
    assert value(merge, '--save-model-path') == adapter + '_merged'
    predictions = [command for command in commands if '--answers-file' in command.argv]
    assert predictions and all(value(command, '--model-path') == adapter + '_merged' for command in predictions)
    if method == 'sefe':
        key = next(command for command in commands if 'mcitlib/methods/_sefe/key_elements.py' in command.argv)
        assert key.argv[-3:] == [str(task - 1), str(Path(adapter).parent), 'internvl']
        if task > 1:
            assert value(train, '--regularization_info_path').endswith(f'/Task{task - 1}_internvl_lora_topP.pth')
        if benchmark == 'mllm_dcl':
            assert all('/DCL/' in value(command, '--answers-file') for command in predictions)
    else:
        extracts = [command for command in commands if 'mcitlib.methods._keeplora.extract_gradients' in command.argv]
        assert extracts and all(value(command, '--model-config').endswith('/model_configs/internvl.json')
                                for command in extracts)
        assert any('/train_pre/' in value(command, '--train-config') for command in extracts)
        assert any('/train_post/' in value(command, '--train-config') for command in extracts)
        assert value(train, '--init_lora_from_gradients').endswith(f'/Task{task}_internvl_lora_gradients/gradients_merged.pt')


@pytest.mark.parametrize('benchmark,source_bench,task',
                         [('ucit', 'UCIT', task) for task in range(1, 7)] +
                         [('mllm_dcl', 'MLLM-DCL', task) for task in range(1, 6)])
def test_internvl_modalprompt_train_resume_and_separate_inference(benchmark, source_bench, task):
    source = json.loads((CONFIGS / 'train_configs/ModalPrompt/InternVL' / source_bench /
                         'train' / f'task{task}.json').read_text())
    options = dict(task_id=task, validation_fixes=True, execution_profile='four_gpu_validation', gpus='0,1,2,3')
    commands = build_sequence_commands(CONFIGS, 'internvl', 'modalprompt', benchmark, **options)
    assert len(commands) == 1 and commands[0].kind == 'train'
    train = commands[0]
    assert value(train, '--output_dir').endswith(f'/Task{task}_internvl')
    assert value(train, '--num_train_epochs') == '1'
    assert train.metadata['training_profile']['epochs'] == '1'
    full = build_sequence_commands(CONFIGS, 'internvl', 'modalprompt', benchmark, task_id=task,
                                   validation_fixes=True)
    full_train = next(command for command in full if command.kind == 'train')
    assert int(value(full_train, '--num_train_epochs')) == source['epoch']
    assert value(train, '--text_tower').endswith('/clip-vit-large-patch14-336')
    assert int(value(train, '--prefix_len')) == source['prefix_len']
    assert int(value(train, '--num_tasks')) == source['num_tasks']
    assert int(value(train, '--cur_task')) == task
    if task == 1:
        assert train.argv.count('--pretrain_mm_mlp_adapter') == 1
        assert '--previous_task_model_path' not in train.argv
    else:
        assert '--pretrain_mm_mlp_adapter' not in train.argv
        assert value(train, '--previous_task_model_path').endswith(f'/Task{task - 1}_internvl')
    if benchmark == 'ucit' and task == 2:
        assert train.metadata['internvl_repairs'] == ['modalprompt_ucit_task2_output_matches_eval_and_task3_input']
    inference = build_sequence_commands(CONFIGS, 'internvl', 'modalprompt', benchmark, only_eval=True, **options)
    models = [command for command in inference if '--model-path' in command.argv]
    assert models and all(value(command, '--model-path') == value(train, '--output_dir') for command in models)
    assert all(value(command, '--model-base').endswith('/InternVL-Chat-ViT-6B-Vicuna-7B') for command in models)


def test_doctor_accepts_explicit_backbone():
    args = build_parser().parse_args(['doctor', '--backbone', 'internvl', '--method', 'lora_ft', '--benchmark', 'ucit'])
    assert args.backbone == 'internvl'


def test_backbone_log_and_result_paths_are_isolated(tmp_path):
    from mcitlib.paths import PathResolver

    resolver = PathResolver.from_file(repo_root=tmp_path)
    assert resolver.run_dir('ucit', 'lora_ft', backbone='internvl') == (
        tmp_path / 'logs/ucit/internvl/lora_ft'
    )
    assert resolver.run_dir('ucit', 'lora_ft') == tmp_path / 'logs/ucit/llava/lora_ft'
    assert resolver.method_results_dir('lora_ft', 'internvl') == tmp_path / 'results/internvl/lora_ft'
    assert resolver.method_results_dir('lora_ft') == tmp_path / 'results/lora_ft'
    commands = build_sequence_commands(CONFIGS, 'internvl', 'lora_ft', 'ucit', task_id=1)
    answers = [value(command, '--answers-file') for command in commands if '--answers-file' in command.argv]
    assert answers and all(path.startswith('${results_root}/internvl/lora_ft/') for path in answers)
