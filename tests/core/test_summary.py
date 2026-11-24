# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Focused CPU tests for single-run summary provenance and metrics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mcitlib import summary
from mcitlib.eval.metrics import continual_metrics
from mcitlib.registry import get_benchmark


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')
    return path


def train(task, sequence='train'):
    return {'argv': ['deepspeed', '--include', 'localhost:0,1', '--per_device_train_batch_size', '8',
                     '--gradient_accumulation_steps', '4', '--learning_rate', '2e-4',
                     '--num_train_epochs', '1'], 'cwd': '.', 'env': {}, 'kind': 'train',
            'task_id': task, 'sequence': sequence, 'metadata': {}}


def fixture(tmp_path, method='mrlora', benchmark='mllm_dcl'):
    project, run, results = tmp_path / 'project', tmp_path / 'run', tmp_path / 'results'
    project.mkdir()
    tasks = list(get_benchmark(benchmark).tasks)
    job = {'method': method, 'benchmark': benchmark, 'status': 'completed', 'task': len(tasks)}
    state = {'run_id': 'one-run', 'status': 'completed', 'project': str(project),
             'roots': {'results_root': str(results), 'logs_root': str(run), 'ckpt_root': str(tmp_path / 'ckpts')},
             'summary': {'completed': 1}, 'jobs': {f'{method}/{benchmark}': job}}
    write_json(run / 'state.json', state)
    commands = {'commands': [train(i + 1) for i in range(len(tasks))]}
    write_json(run / 'jobs' / method / benchmark / 'commands.json', commands)
    name = 'DCL' if benchmark == 'mllm_dcl' else 'UCIT'
    root = results / method / benchmark / method
    return run, state, commands, root / name, root / f'{name}_router', tasks


def matrix(root, tasks, offset, label):
    values = []
    for i in range(len(tasks)):
        row = []
        for j, task in enumerate(tasks[:i + 1]):
            value = float(offset + i + j)
            path = root / 'each_dataset' / task / f'{label}-task{i + 1}/Result.text'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f'Samples: 2\nAccuracy: {value}\n')
            row.append(value)
        values.append(row)
    return values


def test_router_selected_over_different_expert_scores(tmp_path):
    run, state, commands, expert, router, tasks = fixture(tmp_path)
    matrix(expert, tasks, 10, 'MR-LoRA')
    expected = matrix(router, tasks, 70, 'MR-LoRA-router')
    state['jobs']['mrlora/mllm_dcl']['metrics_result_dir'] = str(router)
    commands['metrics_result_dir'] = str(router)
    write_json(run / 'state.json', state)
    write_json(run / 'jobs/mrlora/mllm_dcl/commands.json', commands)
    job = summary.queue_report(run, 'mrlora', 'mllm_dcl')['jobs'][0]
    assert job['result_source'] == 'router' and job['matrix'] == expected
    assert job['metrics'] == continual_metrics(expected)


@pytest.mark.parametrize('benchmark', ['ucit', 'mllm_dcl'])
def test_legacy_experts_and_unrecorded_router_never_become_router_scores(tmp_path, benchmark):
    run, _, _, expert, router, tasks = fixture(tmp_path, benchmark=benchmark)
    matrix(expert, tasks, 10, 'MR-LoRA')
    matrix(router, tasks, 70, 'MR-LoRA-router')
    job = summary.queue_report(run, 'mrlora', benchmark)['jobs'][0]
    assert job['result_source'] == 'invalid' and job['results_dir'] is None
    assert all(value is None for row in job['matrix'] for value in row)
    assert all(value is None for value in job['metrics'].values())
    assert len(job['missing_cells']) == len(tasks) * (len(tasks) + 1) // 2


def test_missing_or_misnamed_router_cell_cannot_fall_back(tmp_path):
    run, state, commands, expert, router, tasks = fixture(tmp_path)
    matrix(expert, tasks, 10, 'MR-LoRA')
    matrix(router, tasks, 70, 'MR-LoRA-router')
    valid = router / 'each_dataset' / tasks[0] / 'MR-LoRA-router-task1/Result.text'
    valid.unlink()
    wrong = router / 'each_dataset' / tasks[0] / 'MR-LoRA-task1/Result.text'
    wrong.parent.mkdir(parents=True)
    wrong.write_text('Samples: 2\nAccuracy: 10\n')
    state['jobs']['mrlora/mllm_dcl']['metrics_result_dir'] = str(router)
    commands['metrics_result_dir'] = str(router)
    write_json(run / 'state.json', state)
    write_json(run / 'jobs/mrlora/mllm_dcl/commands.json', commands)
    job = summary.queue_report(run, 'mrlora', 'mllm_dcl')['jobs'][0]
    assert job['matrix'][0][0] is None and not job['matrix_complete']
    assert all(value is None for value in job['metrics'].values())
    assert any('non-router result suppressed' in x for x in job['warnings'])


def test_router_symlink_cannot_escape_run_results(tmp_path):
    run, state, commands, _, router, tasks = fixture(tmp_path)
    external = tmp_path / 'another-run'
    matrix(external, tasks, 70, 'MR-LoRA-router')
    router.parent.mkdir(parents=True)
    router.symlink_to(external, target_is_directory=True)
    state['jobs']['mrlora/mllm_dcl']['metrics_result_dir'] = str(router)
    commands['metrics_result_dir'] = str(router)
    write_json(run / 'state.json', state)
    write_json(run / 'jobs/mrlora/mllm_dcl/commands.json', commands)
    job = summary.queue_report(run, 'mrlora', 'mllm_dcl')['jobs'][0]
    assert job['result_source'] == 'invalid' and all(v is None for row in job['matrix'] for v in row)


def test_missing_training_stage_keeps_valid_metrics_and_marks_profile(tmp_path):
    run, _, commands, expert, _, tasks = fixture(tmp_path, method='lora_ft', benchmark='ucit')
    expected = matrix(expert, tasks, 50, 'LoRA')
    commands['commands'] = commands['commands'][1:]
    write_json(run / 'jobs/lora_ft/ucit/commands.json', commands)
    job = summary.queue_report(run, 'lora_ft', 'ucit')['jobs'][0]
    assert job['metrics'] == continual_metrics(expected)
    assert job['training_profile']['missing_training_stages'] == [1]
    assert job['training_profile']['uniform_comparison_eligible'] is False


def test_missing_sidecar_and_conflicting_sidecar_never_erase_valid_metrics(tmp_path):
    run, _, _, expert, _, tasks = fixture(tmp_path, method='lora_ft', benchmark='ucit')
    expected = matrix(expert, tasks, 50, 'LoRA')
    sidecar = run / 'jobs/lora_ft/ucit/uniform-profile.json'
    write_json(sidecar, {'commands_sha256': 'wrong', 'experiment_id': 'unified'})
    job = summary.queue_report(run, 'lora_ft', 'ucit')['jobs'][0]
    assert job['training_profile']['uniform_sidecar']['status'] == 'conflicting'
    assert job['training_profile']['uniform_sidecar']['experiment_id'] is None
    assert job['metrics'] == continual_metrics(expected)
    sidecar.unlink()
    job = summary.queue_report(run, 'lora_ft', 'ucit')['jobs'][0]
    assert job['training_profile']['uniform_sidecar']['status'] == 'unavailable'
    assert job['metrics'] == continual_metrics(expected)


def test_measured_zero_and_missing_cell_remain_distinct(tmp_path):
    run, _, _, expert, _, tasks = fixture(tmp_path, method='lora_ft', benchmark='ucit')
    matrix(expert, tasks, 50, 'LoRA')
    first = expert / 'each_dataset' / tasks[0] / 'LoRA-task1/Result.text'
    first.write_text('Samples: 2\nAccuracy: 0\n')
    missing = expert / 'each_dataset' / tasks[0] / 'LoRA-task2/Result.text'
    missing.unlink()
    job = summary.queue_report(run, 'lora_ft', 'ucit')['jobs'][0]
    assert job['matrix'][0][0] == 0 and job['matrix'][1][0] is None
    assert all(value is None for value in job['metrics'].values())


def test_profile_file_hash_conflict_warns_but_does_not_erase_metrics(tmp_path):
    run, state, commands, expert, _, tasks = fixture(tmp_path, method='lora_ft', benchmark='ucit')
    expected = matrix(expert, tasks, 50, 'LoRA')
    project = Path(state['project'])
    profile = project / 'configs/training_profiles/profile.yaml'
    profile.parent.mkdir(parents=True)
    profile.write_text('id: example\n')
    metadata = {'id': 'example', 'path': 'configs/training_profiles/profile.yaml',
                'sha256': hashlib.sha256(profile.read_bytes()).hexdigest(), 'role': 'main_train'}
    commands['commands'][0]['metadata']['training_profile'] = metadata
    write_json(run / 'jobs/lora_ft/ucit/commands.json', commands)
    job = summary.queue_report(run, 'lora_ft', 'ucit')['jobs'][0]
    assert job['training_profile']['training'][0]['scientific_profile']['id'] == 'example'
    profile.write_text('id: changed\n')
    job = summary.queue_report(run, 'lora_ft', 'ucit')['jobs'][0]
    assert job['training_profile']['status'] == 'conflicting'
    assert job['training_profile']['training'][0]['scientific_profile']['id'] is None
    assert job['metrics'] == continual_metrics(expected)


def test_verified_sidecar_exposes_producer_planned_samples_and_steps(tmp_path):
    run, state, commands, expert, _, tasks = fixture(tmp_path, method='lora_ft', benchmark='ucit')
    expected = matrix(expert, tasks, 50, 'LoRA')
    project = Path(state['project'])
    profile_file = project / 'configs/training_profiles/profile.yaml'
    profile_file.parent.mkdir(parents=True)
    profile_file.write_text('id: example\n')
    declaration = {'id': 'example', 'path': 'configs/training_profiles/profile.yaml',
                   'sha256': hashlib.sha256(profile_file.read_bytes()).hexdigest(), 'role': 'main_train'}
    for command in commands['commands']:
        command['metadata']['training_profile'] = declaration
    write_json(run / 'jobs/lora_ft/ucit/commands.json', commands)
    flags = ('--learning_rate', '--per_device_train_batch_size', '--gradient_accumulation_steps', '--num_train_epochs')
    rows = [{'command_index': index, 'task': command['task_id'], 'sequence': 'train', 'role': 'main_train',
             'after': {flag: summary.base._argument(command['argv'], flag) for flag in flags},
             'resolved_epochs': 1, 'training_examples': 3000, 'optimizer_steps': 47}
            for index, command in enumerate(commands['commands'])]
    sidecar = {'schema_version': 1, 'experiment_id': 'experiment-one',
               'policy': {'global_batch': 64, 'base_learning_rate': '0.0002',
                          **{key: declaration[key] for key in ('id', 'path', 'sha256')}},
               'commands_sha256': summary.commands_hash(commands['commands']), 'training_commands': rows}
    path = write_json(run / 'jobs/lora_ft/ucit/uniform-profile.json', sidecar)
    job = summary.queue_report(run, 'lora_ft', 'ucit')['jobs'][0]
    assert job['metrics'] == continual_metrics(expected)
    assert job['training_profile']['uniform_sidecar']['status'] == 'recorded_and_matched'
    assert job['training_profile']['training'][0]['planned_dataset_examples'] == 3000
    assert job['training_profile']['training'][0]['planned_optimizer_steps'] == 47
    sidecar['example_only'] = True
    write_json(path, sidecar)
    job = summary.queue_report(run, 'lora_ft', 'ucit')['jobs'][0]
    assert job['training_profile']['uniform_sidecar']['status'] == 'unverified'
    assert job['training_profile']['training'][0]['planned_optimizer_steps'] is None
    del sidecar['example_only']
    sidecar['training_commands'][0]['after']['--num_train_epochs'] = '10'
    write_json(path, sidecar)
    job = summary.queue_report(run, 'lora_ft', 'ucit')['jobs'][0]
    assert job['training_profile']['uniform_sidecar']['status'] == 'conflicting'
    assert job['training_profile']['training'][0]['planned_optimizer_steps'] is None
    assert job['metrics'] == continual_metrics(expected)


def test_deepspeed_metadata_reports_actual_config_kind_without_claiming_effective_lr(tmp_path):
    ds = write_json(tmp_path / 'zero3.json', {'optimizer': {'type': 'AdamW', 'params': {'lr': 'auto'}},
                                             'scheduler': {'type': 'WarmupLR'}})
    cmd = train(1)
    cmd['argv'] += ['--deepspeed', str(ds), '--lr_scheduler_type', 'cosine']
    info = summary.planned_training({'commands': [cmd]}, None, [], tmp_path)['training'][0]
    assert info['lr_scheduler_recorded'] == 'cosine'
    assert info['deepspeed_metadata']['scheduler_type'] == 'WarmupLR'
    assert info['deepspeed_metadata']['optimizer_lr_config'] == 'auto'


def test_sample_counts_are_not_invented_for_router_training():
    profile = {'training': [{'task': 1, 'sequence': 'train'}, {'task': 1, 'sequence': 'router'}]}
    job = {'tasks': ['RS'], 'planned_samples': {'RS': {'train': {'sample_count': 3000}}}}
    summary.add_planned_samples(profile, job)
    assert profile['training'][0]['planned_train_samples'] == 3000
    assert profile['training'][1]['planned_train_samples'] is None
    assert all(x['planned_optimizer_steps'] is None for x in profile['training'])


def test_filter_and_output_include_missing_and_profile_csv(tmp_path):
    run, state, _, expert, _, tasks = fixture(tmp_path, method='lora_ft', benchmark='ucit')
    matrix(expert, tasks, 50, 'LoRA')
    state['jobs']['mrlora/mllm_dcl'] = {'method': 'mrlora', 'benchmark': 'mllm_dcl', 'status': 'pending'}
    state['summary'] = {'completed': 1, 'pending': 1}
    write_json(run / 'state.json', state)
    report = summary.queue_report(run, 'mrlora', 'mllm_dcl')
    assert len(report['jobs']) == 1 and report['selection']['run_jobs'] == 2
    assert len(summary.queue_report(run, None, None)['jobs']) == 2
    outputs = summary.write_summary(report, tmp_path / 'output')
    assert 'Missing cells:' in Path(outputs['markdown']).read_text()
    assert 'N/A' in Path(outputs['training_profiles_csv']).read_text()
    assert all(Path(path).is_file() for path in outputs.values())


def eval_fixture(tmp_path):
    old, _, commands, _, _, tasks = fixture(tmp_path, method='mrlora', benchmark='ucit')
    results, logs = tmp_path / 'eval-results', tmp_path / 'eval-logs'
    router = results / 'mrlora/UCIT_router'
    scoring = []
    for i in range(len(tasks)):
        for task in tasks[:i + 1]:
            dest = router / 'each_dataset' / task / f'MR-LoRA-router-task{i + 1}'
            scoring.append({'argv': ['python', '--output-dir', str(dest), '--result-file', str(dest / 'routed_predictions.jsonl')], 'task_id': i + 1,
                            'metadata': {'validation_fix': 'mrlora_ucit_router_scoring'}})
    finish = {'argv': ['python', '-m', 'mcitlib.methods._mrlora.ucit_eval', '_finish',
                       '--results-root', str(router), '--output', str(logs / 'metrics.json')],
              'metadata': {'validation_fix': 'mrlora_ucit_verified_metrics'}}
    plan = {'results_root': str(results), 'logs_dir': str(logs), 'old_run_dir': str(old),
            'source': {'state_sha256': hashlib.sha256((old / 'state.json').read_bytes()).hexdigest(),
                       'commands_sha256': hashlib.sha256((old / 'jobs/mrlora/ucit/commands.json').read_bytes()).hexdigest()},
            'commands': scoring + [finish]}
    path = write_json(tmp_path / 'plan.json', plan)
    state = write_json(tmp_path / 'eval-state.json', {'status': 'running',
                       'original_plan_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'actions': []})
    return path, state, router, logs, tasks, finish


def test_evaluation_only_partial_and_controller_binding(tmp_path):
    path, state, router, _, tasks, _ = eval_fixture(tmp_path)
    dest = router / 'each_dataset' / tasks[0] / 'MR-LoRA-router-task1/Result.text'
    dest.parent.mkdir(parents=True)
    dest.write_text('Samples: 2\nAccuracy: 70\n')
    job = summary.evaluation_report(path, state)['jobs'][0]
    assert job['status'] == 'running' and job['matrix'][0][0] == 70
    assert len(job['missing_cells']) == 20 and all(v is None for v in job['metrics'].values())
    bad = json.loads(state.read_text())
    bad['original_plan_sha256'] = 'bad'
    write_json(state, bad)
    with pytest.raises(ValueError, match='does not bind'):
        summary.evaluation_report(path, state)


def test_evaluation_plan_rejects_score_input_from_expert_root(tmp_path):
    path, _, _, _, _, _ = eval_fixture(tmp_path)
    payload = json.loads(path.read_text())
    payload['commands'][0]['argv'][-1] = str(tmp_path / 'expert/merge.jsonl')
    write_json(path, payload)
    with pytest.raises(ValueError, match='malformed router'):
        summary.evaluation_plan(path)


def test_evaluation_only_completion_requires_finish_marker_and_metrics_source(tmp_path):
    path, state, router, logs, tasks, finish = eval_fixture(tmp_path)
    expected = matrix(router, tasks, 70, 'MR-LoRA-router')
    payload = json.loads(state.read_text())
    payload.update(status='completed', actions=[{'name': 'resume_execute', 'status': 'completed'}])
    write_json(state, payload)
    job = summary.evaluation_report(path, state)['jobs'][0]
    assert job['status'] == 'unknown' and all(v is None for v in job['metrics'].values())
    ident = summary.sha(json.dumps(finish, sort_keys=True, ensure_ascii=False).encode())[:20]
    write_json(logs / '.mcitlib/steps' / f'{ident}-done.json',
               {'id': ident, 'status': 'completed', 'return_code': 0, 'command': finish})
    write_json(logs / 'metrics.json', {'source': str(router), 'tasks': tasks, 'matrix': expected})
    job = summary.evaluation_report(path, state)['jobs'][0]
    assert job['status'] == 'completed' and job['metrics'] == continual_metrics(expected)
    wrong = json.loads((logs / 'metrics.json').read_text())
    wrong['source'] = str(tmp_path / 'expert')
    write_json(logs / 'metrics.json', wrong)
    job = summary.evaluation_report(path, state)['jobs'][0]
    assert job['status'] == 'unknown' and all(v is None for v in job['metrics'].values())
