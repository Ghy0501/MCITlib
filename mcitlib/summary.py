#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Provenance-aware, single-run experiment summaries.

Uses mcitlib.reporting and its original metrics; never starts model or GPU work.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcitlib import reporting as base
from mcitlib.registry import get_benchmark, get_method

METRICS = ('mft', 'mfn', 'maa', 'bwt')
MAX_METADATA = 8 * 1024 * 1024


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checked_json(path: Path, warnings: list[str]) -> tuple[Any, str] | tuple[None, None]:
    try:
        if path.stat().st_size > MAX_METADATA:
            raise ValueError('metadata exceeds size limit')
        raw = path.read_bytes()
        if raw != path.read_bytes():
            raise ValueError('file changed while read')
        return json.loads(raw), sha(raw)
    except (OSError, ValueError, UnicodeError) as error:
        warnings.append(f'unavailable metadata {path}: {error}')
        return None, None


def number(raw: str | None, *, positive: bool = False) -> float | None:
    if raw is None:
        return None
    try:
        value = float(raw)
    except (ValueError, OverflowError):
        return None
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        return None
    return value


def commands_hash(commands: list[Any]) -> str:
    return sha(json.dumps(commands, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode())


def recorded_scientific_profile(metadata: dict, project: Path | None, sequence: Any,
                                warnings: list[str]) -> dict[str, Any]:
    recorded = metadata.get('training_profile')
    result = {'id': None, 'status': 'unavailable', 'recorded': recorded}
    if recorded is None:
        return result
    if (not isinstance(recorded, dict) or not isinstance(recorded.get('id'), str) or
            not isinstance(recorded.get('path'), str) or not isinstance(recorded.get('sha256'), str) or
            recorded.get('role') != 'main_train' or sequence != 'train' or project is None):
        warnings.append('invalid scientific training profile declaration; profile is N/A')
        return {**result, 'status': 'conflicting'}
    path = Path(recorded['path'])
    if path.is_absolute():
        warnings.append('scientific training profile path must be project-relative')
        return {**result, 'status': 'conflicting'}
    root = project.resolve()
    resolved = (root / path).resolve()
    if root not in resolved.parents:
        warnings.append('scientific training profile path escapes the experiment project')
        return {**result, 'status': 'conflicting'}
    try:
        if resolved.stat().st_size > MAX_METADATA:
            raise ValueError('profile file exceeds metadata size limit')
        raw = resolved.read_bytes()
        if sha(raw) != recorded['sha256']:
            raise ValueError('profile file hash differs')
        content = base.yaml.safe_load(raw)
        if not isinstance(content, dict) or content.get('id') != recorded['id']:
            raise ValueError('profile id differs from file')
    except (OSError, ValueError, base.yaml.YAMLError) as error:
        warnings.append(f'scientific training profile cannot be verified: {error}')
        return {**result, 'status': 'conflicting'}
    return {'id': recorded['id'], 'status': 'file_hash_matches_at_collection', 'recorded': recorded,
            'observed_file': str(resolved),
            'note': 'Collect-time file match does not prove launch-time application or optimizer behavior.'}


def deepspeed_metadata(raw_path: str | None, project: Path | None, warnings: list[str]) -> dict[str, Any] | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        if project is None:
            return None
        path = project / path
    local_warnings: list[str] = []
    payload, digest = checked_json(path, local_warnings)
    for warning in local_warnings:
        if warning not in warnings:
            warnings.append(warning)
    if not isinstance(payload, dict):
        return None
    optimizer = payload.get('optimizer') if isinstance(payload.get('optimizer'), dict) else {}
    scheduler = payload.get('scheduler') if isinstance(payload.get('scheduler'), dict) else {}
    return {'path': str(path.resolve()), 'sha256_at_collection': digest,
            'optimizer_type': optimizer.get('type'), 'scheduler_type': scheduler.get('type'),
            'optimizer_lr_config': (optimizer.get('params') or {}).get('lr') if isinstance(optimizer.get('params'), dict) else None,
            'note': 'File is observed at collection; optimizer/scheduler effective behavior is not reconstructed.'}


def planned_training(payload: Any, state_profile: Any, warnings: list[str],
                     project: Path | None = None) -> dict[str, Any]:
    """Record exact argv options; do not assert the optimizer actually used them."""
    result = {'state_execution_profile': state_profile, 'commands_execution_profile': None,
              'training': None, 'gradient_preparation': None, 'status': 'unavailable',
              'qualification': 'Values are recorded command plans, not proof of executed steps or effective DeepSpeed LR.'}
    if not isinstance(payload, dict) or not isinstance(payload.get('commands'), list):
        return result
    commands = payload['commands']
    resource = base._command_resources(payload, warnings)
    training = []
    for item in resource['training'] or []:
        command = commands[item['command_index']]
        argv = command['argv']
        metadata = command.get('metadata') if isinstance(command.get('metadata'), dict) else {}
        lr = base._argument(argv, '--learning_rate')
        epochs = base._argument(argv, '--num_train_epochs')
        ds_config = base._argument(argv, '--deepspeed')
        training.append({**item, 'learning_rate': number(lr), 'learning_rate_recorded': lr,
                         'num_train_epochs': number(epochs, positive=True), 'num_train_epochs_recorded': epochs,
                         'max_steps_recorded': base._argument(argv, '--max_steps'),
                         'mm_projector_lr_recorded': base._argument(argv, '--mm_projector_lr'),
                         'weight_decay_recorded': base._argument(argv, '--weight_decay'),
                         'lr_scheduler_recorded': base._argument(argv, '--lr_scheduler_type'),
                         'deepspeed_config': ds_config, 'deepspeed_metadata': deepspeed_metadata(ds_config, project, warnings),
                         'scientific_profile': recorded_scientific_profile(metadata, project, item.get('sequence'), warnings)})
    cmd_profile = payload.get('execution_profile')
    recorded = [x.get('execution_profile') for x in training if x.get('execution_profile') is not None]
    conflict = (state_profile is not None and cmd_profile is not None and state_profile != cmd_profile)
    if cmd_profile is not None and any(item != cmd_profile for item in recorded):
        conflict = True
    if state_profile is not None and any(item != state_profile for item in recorded):
        conflict = True
    if any(item['scientific_profile']['status'] == 'conflicting' for item in training):
        conflict = True
    if conflict:
        warnings.append('execution profile conflicts across state, commands or training metadata; uniform comparison is unavailable')
    result.update(commands_execution_profile=cmd_profile, training=training,
                  gradient_preparation=resource['gradient_preparation'],
                  status='conflicting' if conflict else ('recorded' if training else 'unavailable'))
    return result


def uniform_sidecar(path: Path, commands: Any, warnings: list[str]) -> dict[str, Any]:
    """A sidecar is evidence only when its command digest and fields are checked."""
    unavailable = {'status': 'unavailable', 'path': str(path), 'experiment_id': None, 'policy': None}
    if not path.is_file():
        return unavailable
    payload, digest = checked_json(path, warnings)
    if not isinstance(payload, dict) or not isinstance(commands, list):
        warnings.append('uniform profile sidecar has no verifiable command list')
        return {**unavailable, 'status': 'invalid'}
    if payload.get('example_only') is True:
        warnings.append('uniform profile sidecar is a preview example, not a run record')
        return {**unavailable, 'status': 'unverified', 'sha256': digest}
    if payload.get('commands_sha256') != commands_hash(commands):
        warnings.append('uniform profile sidecar command hash differs; policy is N/A')
        return {**unavailable, 'status': 'conflicting', 'sha256': digest}
    records = payload.get('training_commands')
    if (payload.get('schema_version') != 1 or not isinstance(payload.get('experiment_id'), str) or
            not isinstance(payload.get('policy'), dict) or not isinstance(records, list)):
        warnings.append('uniform profile sidecar lacks required policy or training records')
        return {**unavailable, 'status': 'conflicting', 'sha256': digest}
    policy = payload['policy']
    declared_batch = policy.get('global_batch', policy.get('global_batch_size'))
    declared_lr = number(str(policy.get('base_learning_rate')), positive=True)
    if type(declared_batch) is not int or declared_batch < 1 or declared_lr is None:
        warnings.append('uniform profile sidecar does not declare a valid global batch and base LR')
        return {**unavailable, 'status': 'conflicting', 'sha256': digest}
    fields = ('--learning_rate', '--per_device_train_batch_size', '--gradient_accumulation_steps', '--num_train_epochs')
    selected: dict[int, dict[str, Any]] = {}
    for item in records:
        if not isinstance(item, dict) or type(item.get('command_index')) is not int:
            warnings.append('uniform profile sidecar has an invalid command index')
            return {**unavailable, 'status': 'conflicting', 'sha256': digest}
        index = item['command_index']
        if not 0 <= index < len(commands) or index in selected:
            warnings.append('uniform profile sidecar has a duplicate or out-of-range command index')
            return {**unavailable, 'status': 'conflicting', 'sha256': digest}
        command = commands[index]
        if (not isinstance(command, dict) or command.get('kind') != 'train' or
                item.get('task') != command.get('task_id') or item.get('sequence') != command.get('sequence') or
                item.get('role') not in {'main_train', 'auxiliary_preserved'}):
            warnings.append('uniform profile sidecar disagrees with command task, sequence or role')
            return {**unavailable, 'status': 'conflicting', 'sha256': digest}
        if item['role'] == 'main_train':
            metadata = command.get('metadata') if isinstance(command.get('metadata'), dict) else {}
            declaration = metadata.get('training_profile') if isinstance(metadata.get('training_profile'), dict) else {}
            after = item.get('after')
            argv = command.get('argv')
            if (declaration.get('role') != 'main_train' or command.get('sequence') != 'train' or
                    any(declaration.get(key) != policy.get(key) for key in ('id', 'path', 'sha256')) or
                    not isinstance(after, dict) or not isinstance(argv, list) or
                    any(base._argument(argv, flag) is None or after.get(flag) != base._argument(argv, flag) for flag in fields) or
                    number(base._argument(argv, '--learning_rate'), positive=True) != declared_lr or
                    number(base._argument(argv, '--num_train_epochs'), positive=True) is None or
                    number(str(item.get('resolved_epochs')), positive=True) != number(base._argument(argv, '--num_train_epochs'), positive=True)):
                warnings.append('uniform profile sidecar disagrees with a main-training command')
                return {**unavailable, 'status': 'conflicting', 'sha256': digest}
            world = base._world_size(argv)
            batch = base._positive_int(base._argument(argv, '--per_device_train_batch_size'))
            accumulation = base._positive_int(base._argument(argv, '--gradient_accumulation_steps'))
            expected = {'world_size': world, 'per_device_microbatch': batch, 'gradient_accumulation': accumulation,
                        'nominal_global_batch': world * batch * accumulation if all(x is not None for x in (world, batch, accumulation)) else None}
            if (expected['nominal_global_batch'] != declared_batch or
                    any(item.get(key) is not None and item[key] != value for key, value in expected.items())):
                warnings.append('uniform profile sidecar disagrees with planned training resources')
                return {**unavailable, 'status': 'conflicting', 'sha256': digest}
        elif item.get('argv') is not None and item['argv'] != command.get('argv'):
            warnings.append('uniform profile sidecar disagrees with preserved auxiliary training argv')
            return {**unavailable, 'status': 'conflicting', 'sha256': digest}
        samples, steps = item.get('training_examples'), item.get('optimizer_steps')
        if any(value is not None and (type(value) is not int or value < 1) for value in (samples, steps)):
            warnings.append('uniform profile sidecar contains invalid planned sample or step counts')
            return {**unavailable, 'status': 'conflicting', 'sha256': digest}
        selected[index] = {'command_index': index, 'task': item.get('task'), 'sequence': item.get('sequence'),
                           'role': item['role'], 'training_examples': samples, 'optimizer_steps': steps,
                           'resolved_epochs': item.get('resolved_epochs')}
    expected_main = {i for i, command in enumerate(commands) if isinstance(command, dict) and
                     command.get('kind') == 'train' and isinstance(command.get('metadata'), dict) and
                     isinstance(command['metadata'].get('training_profile'), dict) and
                     command['metadata']['training_profile'].get('role') == 'main_train'}
    if not expected_main or expected_main != {i for i, item in selected.items() if item['role'] == 'main_train'}:
        warnings.append('uniform profile sidecar does not cover exactly the declared main-training commands')
        return {**unavailable, 'status': 'conflicting', 'sha256': digest}
    return {'status': 'recorded_and_matched', 'path': str(path), 'sha256': digest,
            'experiment_id': payload['experiment_id'], 'policy': payload['policy'],
            'training_commands': list(selected.values()),
            'qualification': 'Producer-planned counts; data and runtime optimizer steps were not independently verified.'}


def suppress_metrics(job: dict[str, Any]) -> None:
    job['metrics'] = {name: None for name in METRICS}


def strict_router(job: dict[str, Any]) -> None:
    """Reject the base reporter's legacy expert fallback for both MR benchmarks."""
    if job.get('method') != 'mrlora' or job.get('benchmark') not in {'ucit', 'mllm_dcl'}:
        return
    if job.get('result_source') == 'router':
        return
    old_root = job.get('results_dir')
    if old_root and not job.get('expert_results_dir'):
        job['expert_results_dir'] = old_root
    job['matrix'] = [[None] * (i + 1) for i in range(len(job['tasks']))]
    job['observed_samples'] = [[None] * (i + 1) for i in range(len(job['tasks']))]
    job['matrix_sources'] = [[None] * (i + 1) for i in range(len(job['tasks']))]
    job['matrix_complete'] = False
    job['results_dir'] = None
    job['result_source'] = 'invalid'
    suppress_metrics(job)
    job['warnings'].append('MR router source missing or conflicting; expert scores and aggregate metrics are unavailable')
    job['provenance']['result_selection']['kind'] = 'router_unavailable'
    job['provenance']['result_selection']['results_dir'] = None


def validate_router_cells(job: dict[str, Any], root: Path) -> None:
    """A router directory must not silently accept expert-labelled score files."""
    invalid = False
    for i, row in enumerate(job['matrix_sources']):
        for j, sources in enumerate(row):
            if not sources:
                continue
            expected = root / 'each_dataset' / job['tasks'][j] / f'MR-LoRA-router-task{i + 1}' / 'Result.text'
            for source in sources:
                path = Path(source)
                if path.name != 'Result.text':
                    continue  # Base reporter validates the recorded metrics source.
                resolved = path.resolve()
                if path != expected or root not in resolved.parents or resolved != expected.resolve():
                    job['matrix'][i][j] = None
                    job['observed_samples'][i][j] = None
                    job['matrix_sources'][i][j] = None
                    invalid = True
                    job['warnings'].append(f'non-router result suppressed at stage {i + 1}, task {job["tasks"][j]}')
                    break
    if invalid:
        job['matrix_complete'] = False
        suppress_metrics(job)


def missing_cells(job: dict[str, Any]) -> list[dict[str, Any]]:
    return [{'stage': i + 1, 'eval_task': job['tasks'][j]} for i, row in enumerate(job['matrix'])
            for j, value in enumerate(row) if value is None]


def add_planned_samples(profile: dict[str, Any], job: dict[str, Any]) -> None:
    """Use only sample counts already bound to this job by the base reporter."""
    for item in profile.get('training') or []:
        item['planned_train_samples'] = None
        item['planned_dataset_examples'] = None
        item['planned_optimizer_steps'] = None
        task = item.get('task')
        if item.get('sequence') != 'train' or type(task) is not int or not 1 <= task <= len(job['tasks']):
            continue
        sample = job.get('planned_samples', {}).get(job['tasks'][task - 1])
        training = sample.get('train') if isinstance(sample, dict) else None
        count = training.get('sample_count') if isinstance(training, dict) else None
        if type(count) is int and count >= 0:
            item['planned_train_samples'] = count


def add_sidecar_plans(profile: dict[str, Any]) -> None:
    sidecar = profile['uniform_sidecar']
    if sidecar['status'] != 'recorded_and_matched' or profile['status'] == 'conflicting':
        return
    by_index = {item['command_index']: item for item in sidecar['training_commands']}
    for item in profile.get('training') or []:
        recorded = by_index.get(item['command_index'])
        if recorded:
            item['planned_dataset_examples'] = recorded['training_examples']
            item['planned_optimizer_steps'] = recorded['optimizer_steps']


def annotate_queue_job(report: dict[str, Any], job: dict[str, Any]) -> None:
    strict_router(job)
    if job.get('method') == 'mrlora' and job.get('result_source') == 'router':
        roots = report['manifest']['config'].get('roots') or {}
        result_root = Path(roots['results_root']).resolve() if roots.get('results_root') else None
        router = Path(job['results_dir']).resolve()
        if result_root is None or result_root not in router.parents:
            job['warnings'].append('router results escape the selected run results root')
            job['result_source'] = 'invalid'
            strict_router(job)
        else:
            validate_router_cells(job, router)
    path_info = job['provenance'].get('commands') or {}
    path = Path(path_info['path']) if path_info.get('path') else None
    payload, observed = checked_json(path, job['warnings']) if path and path.is_file() else (None, None)
    if payload is not None and (not path_info.get('sha256') or observed != path_info['sha256']):
        job['warnings'].append('commands changed during report collection; training profile is N/A')
        payload = None
    project_text = report['manifest']['code']['experiment'].get('project')
    project = Path(project_text) if project_text else None
    profile = planned_training(payload, report.get('execution_profile'), job['warnings'], project)
    add_planned_samples(profile, job)
    commands = payload.get('commands') if isinstance(payload, dict) else None
    directory = Path(report['run_dir']) / 'jobs' / job['method'] / job['benchmark']
    profile['uniform_sidecar'] = uniform_sidecar(directory / 'uniform-profile.json', commands, job['warnings'])
    add_sidecar_plans(profile)
    task_ids = {x.get('task') for x in profile['training'] or []
                if x.get('sequence') == 'train' and type(x.get('task')) is int}
    required = set(range(1, len(job['tasks']) + 1))
    profile['missing_training_stages'] = sorted(required - task_ids)
    if profile['missing_training_stages']:
        job['warnings'].append('training commands do not cover every matrix stage; missing profile values are N/A and uniform comparison is unavailable')
    profile['uniform_comparison_eligible'] = False
    profile['uniform_comparison_note'] = 'No cross-run comparison is performed; absent or unverified scientific profile is not assumed uniform.'
    job['training_profile'] = profile
    job['missing_cells'] = missing_cells(job)
    report['manifest']['jobs'][job['key']].update(job['provenance'], training_profile=profile)


def queue_report(run_dir: Path, method: str | None, benchmark: str | None) -> dict[str, Any]:
    report = base.build_report(run_dir)
    all_jobs = report['jobs']
    selected = [j for j in all_jobs if (method is None or j['method'] == method) and
                (benchmark is None or j['benchmark'] == benchmark)]
    if not selected:
        raise ValueError('no job in this run matches the requested method and benchmark')
    report['jobs'] = selected
    report['selection'] = {'method': method, 'benchmark': benchmark, 'selected_jobs': len(selected),
                           'run_jobs': len(all_jobs), 'run_summary_is_unfiltered': True}
    report['manifest']['jobs'] = {j['key']: report['manifest']['jobs'][j['key']] for j in selected}
    report['manifest']['environment']['jobs'] = {j['key']: j['environment'] for j in selected}
    for job in selected:
        annotate_queue_job(report, job)
    report['manifest']['selection'] = report['selection']
    return report


def required_option(argv: list[Any], flag: str) -> str:
    value = base._argument(argv, flag)
    if value is None:
        raise ValueError(f'evaluation plan must contain exactly one {flag}')
    return value


def evaluation_plan(path: Path) -> tuple[dict[str, Any], str, Path, Path, list[str]]:
    """Accept only the planned UCIT router output and its original finish command."""
    warnings: list[str] = []
    plan, digest = checked_json(path, warnings)
    if not isinstance(plan, dict) or not isinstance(plan.get('commands'), list):
        raise ValueError(f'cannot read evaluation plan: {warnings}')
    if not digest or not plan['commands']:
        raise ValueError('evaluation plan has no commands')
    root = Path(plan['results_root'])
    logs = Path(plan['logs_dir'])
    if not root.is_absolute() or not logs.is_absolute():
        raise ValueError('evaluation results and logs must be absolute')
    root, logs = root.resolve(), logs.resolve()
    router = (root / 'mrlora/UCIT_router').resolve()
    if root not in router.parents:
        raise ValueError('router path escapes the recorded results root')
    final = plan['commands'][-1]
    argv = final.get('argv', [])
    if (not isinstance(argv, list) or 'mcitlib.methods._mrlora.ucit_eval' not in argv or '_finish' not in argv or
            Path(required_option(argv, '--results-root')).resolve() != router or
            Path(required_option(argv, '--output')).resolve() != logs / 'metrics.json' or
            final.get('metadata', {}).get('validation_fix') != 'mrlora_ucit_verified_metrics'):
        raise ValueError('evaluation plan does not end with its own UCIT router finish command')
    tasks = list(get_benchmark('ucit').tasks)
    expected = {(stage, task) for stage in range(1, len(tasks) + 1) for task in tasks[:stage]}
    actual = set()
    for command in plan['commands']:
        if command.get('metadata', {}).get('validation_fix') != 'mrlora_ucit_router_scoring':
            continue
        output = Path(required_option(command['argv'], '--output-dir')).resolve()
        try:
            rel = output.relative_to(router / 'each_dataset')
        except ValueError as error:
            raise ValueError('router score output escapes the planned router directory') from error
        stage = command.get('task_id')
        if (len(rel.parts) != 2 or rel.parts[1] != f'MR-LoRA-router-task{stage}' or
                (stage, rel.parts[0]) in actual or
                Path(required_option(command['argv'], '--result-file')).resolve() != output / 'routed_predictions.jsonl'):
            raise ValueError('duplicate or malformed router score output')
        actual.add((stage, rel.parts[0]))
    if actual != expected:
        raise ValueError('evaluation plan does not name the complete UCIT lower triangle')
    return plan, digest, router, logs, tasks


def evaluation_status(path: Path | None, plan_hash: str, final_command: dict[str, Any],
                      logs: Path, router: Path, tasks: list[str],
                      warnings: list[str]) -> tuple[str, dict[str, Any] | None]:
    if path is None:
        warnings.append('no evaluation execution state supplied; aggregate metrics are unavailable')
        return 'unknown', None
    state, digest = checked_json(path, warnings)
    if not isinstance(state, dict) or state.get('original_plan_sha256') != plan_hash:
        raise ValueError('execution state does not bind the supplied evaluation plan')
    status = state.get('status', 'unknown')
    action = next((x for x in state.get('actions', []) if x.get('name') == 'resume_execute'), None)
    if status == 'completed':
        identity = sha(json.dumps(final_command, sort_keys=True, ensure_ascii=False).encode())[:20]
        candidates = list((logs / '.mcitlib/steps').glob(f'{identity}-*.json'))
        successful = []
        for marker in candidates:
            row, marker_hash = checked_json(marker, warnings)
            if (isinstance(row, dict) and row.get('status') == 'completed' and row.get('return_code') == 0 and
                    row.get('id') == identity and row.get('command') == final_command):
                successful.append({'path': str(marker), 'sha256': marker_hash})
        if not action or action.get('status') != 'completed' or not successful:
            warnings.append('completed controller lacks the matching successful final metrics marker')
            status = 'unknown'
        elif not (logs / 'metrics.json').is_file() or base._matrix_payload(logs / 'metrics.json', tasks, warnings, router) is None:
            warnings.append('completed controller lacks a valid metrics file bound to this router directory')
            status = 'unknown'
    elif status not in {'running', 'failed', 'stopped'}:
        status = 'unknown'
    return status, {'path': str(path.resolve()), 'sha256': digest, 'reported_status': state.get('status'),
                    'action': action}


def evaluation_report(plan_path: Path, execution_state: Path | None) -> dict[str, Any]:
    snapshot = now()
    plan_path = plan_path.expanduser().resolve()
    plan, digest, router, logs, tasks = evaluation_plan(plan_path)
    warnings: list[str] = []
    status, state_info = evaluation_status(execution_state, digest, plan['commands'][-1], logs, router, tasks, warnings)
    matrix, samples, sources, conflict = base._matrix(router, logs / 'metrics.json', tasks, status, warnings, router)
    complete = all(value is not None for row in matrix for value in row)
    metrics = base.continual_metrics(matrix) if status == 'completed' and complete and not conflict else {
        name: None for name in METRICS}
    source_run = Path(plan['old_run_dir']).expanduser().resolve()
    state_file = source_run / 'state.json'
    commands_file = source_run / 'jobs/mrlora/ucit/commands.json'
    old_state, old_state_hash = checked_json(state_file, warnings)
    old_commands, old_commands_hash = checked_json(commands_file, warnings)
    recorded = plan.get('source', {})
    if old_state_hash != recorded.get('state_sha256') or old_commands_hash != recorded.get('commands_sha256'):
        warnings.append('recorded source training state/commands changed; training profile is N/A')
        old_state, old_commands = None, None
    state_profile = old_state.get('execution_profile') if isinstance(old_state, dict) else None
    old_project = Path(old_state['project']) if isinstance(old_state, dict) and old_state.get('project') else None
    profile = planned_training(old_commands, state_profile, warnings, old_project)
    profile['uniform_sidecar'] = uniform_sidecar(source_run / 'jobs/mrlora/ucit/uniform-profile.json',
                                                 old_commands.get('commands') if isinstance(old_commands, dict) else None,
                                                 warnings)
    profile['uniform_comparison_eligible'] = False
    profile['uniform_comparison_note'] = 'No cross-run comparison is performed; absent or unverified scientific profile is not assumed uniform.'
    resource = base._command_resources(old_commands, warnings)
    environment = {'name': None, 'python': plan.get('python'), 'python_version_from_doctor': None}
    selection = {'kind': 'router', 'results_dir': str(router), 'source': 'evaluation_plan',
                 'evaluation_plan': str(plan_path), 'evaluation_plan_sha256': digest}
    provenance = {'result_selection': selection, 'training_state': {'path': str(state_file), 'sha256': old_state_hash},
                  'training_commands': {'path': str(commands_file), 'sha256': old_commands_hash}}
    job = {'key': 'mrlora/ucit', 'method': 'mrlora', 'benchmark': 'ucit', 'status': status,
           'phase': 'evaluation_only', 'task': None, 'step': None, 'latest_event': None, 'event_count': None,
           'started_at': None, 'finished_at': None, 'elapsed_seconds': None, 'elapsed_is_open': False,
           'error': None, 'failure_log': None, 'error_tail': None, 'logs': {'evaluation_logs': str(logs)},
           'doctor_failures': [], 'tasks': tasks, 'matrix': matrix, 'matrix_complete': complete,
           'matrix_sources': sources, 'observed_samples': samples, 'metrics': metrics,
           'planned_samples': {task: None for task in tasks}, 'results_dir': str(router),
           'result_source': 'router', 'expert_results_dir': str(Path(plan['results_root']) / 'mrlora/UCIT'),
           'environment': environment, 'resources': resource, 'provenance': provenance, 'warnings': warnings,
           'training_profile': profile}
    add_planned_samples(profile, job)
    add_sidecar_plans(profile)
    validate_router_cells(job, router)
    job['missing_cells'] = missing_cells(job)
    state_info = state_info or {'path': None, 'sha256': None, 'reported_status': None, 'action': None}
    sample = plan.get('sample') if isinstance(plan.get('sample'), dict) else None
    manifest = {'run_id': logs.name, 'snapshot_at': snapshot, 'state_file': state_info,
                'config': {'roots': {'logs_root': str(logs), 'results_root': plan['results_root'],
                                     'ckpt_root': None}},
                'environment': {'execution_profile': state_profile, 'jobs': {'mrlora/ucit': environment}},
                'sample': sample, 'sample_counts': {}, 'jobs': {'mrlora/ucit': {**provenance, 'training_profile': profile}},
                'lineage': {'previous_run': str(source_run), 'transition_receipt': None, 'related_runs': []},
                'selection': {'method': 'mrlora', 'benchmark': 'ucit', 'mode': 'evaluation_only'},
                'note': 'Scores come solely from this evaluation plan router root; related training run supplies provenance only.'}
    return {'schema_version': 1, 'snapshot_at': snapshot, 'collected_at': now(), 'run_dir': str(logs),
            'run_id': logs.name, 'status': status, 'current': 'mrlora/ucit' if status == 'running' else None,
            'execution_profile': state_profile, 'lineage': manifest['lineage'],
            'started_at': None, 'updated_at': None, 'finished_at': None, 'summary': {status: 1},
            'recorded_summary': None, 'snapshot_note': 'Live files are read sequentially, not atomically.',
            'warnings': [], 'jobs': [job], 'manifest': manifest, 'selection': manifest['selection']}


def atomic_write(path: Path, content: str) -> None:
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='', dir=path.parent,
                                         prefix='.summary-', delete=False) as stream:
            name = stream.name
            stream.write(content)
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def write_summary(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    report['manifest']['summary_reporter'] = {'source': str(Path(__file__).resolve()),
                                              'sha256': sha(Path(__file__).read_bytes()),
                                              'note': 'Collect-time fingerprint; not launch-time evidence.'}
    paths = base.write_report(report, output_dir)
    # Base renderer describes its original metadata rule; replace that sentence
    # for strict-router reports, where an absent declaration is also unavailable.
    markdown = Path(paths['markdown'])
    text = markdown.read_text(encoding='utf-8').replace(
        'Matrix source: N/A; the explicitly recorded result directory is invalid or inconsistent.',
        'Matrix source: N/A; no trustworthy router directory was recorded, or declarations conflict.')
    text += '\n## Training profile and missing cells\n\n'
    text += 'Training values are command plans; they do not establish executed steps, effective DeepSpeed LR, or uniformity across runs.\n'
    fields = ['method', 'benchmark', 'kind', 'stage', 'sequence', 'command_index', 'state_execution_profile',
              'commands_execution_profile', 'command_execution_profile', 'world_size', 'per_device_batch_size',
              'gradient_accumulation_steps', 'effective_batch_size', 'learning_rate', 'num_train_epochs',
              'planned_train_samples', 'planned_dataset_examples', 'planned_optimizer_steps',
              'max_steps_recorded', 'mm_projector_lr_recorded', 'lr_scheduler_recorded', 'deepspeed_config',
              'deepspeed_optimizer_type', 'deepspeed_scheduler_type', 'scientific_profile_id',
              'num_chunks', 'chunk_index', 'cuda_visible_devices', 'merge_only', 'uniform_experiment_id']
    csv_handle = io.StringIO(newline='')
    writer = csv.DictWriter(csv_handle, fieldnames=fields)
    writer.writeheader()
    for job in report['jobs']:
        profile = job['training_profile']
        sidecar = profile['uniform_sidecar']
        text += f"\n### {base._md(job['key'])}\n\n"
        text += f"Profile status: {base._md(profile['status'])}; uniform policy: {base._md(sidecar['status'])}.\n\n"
        text += 'Uniform comparison: not established; no cross-run comparison is performed.\n\n'
        if profile.get('missing_training_stages'):
            text += 'Missing training command stages: ' + ', '.join(map(str, profile['missing_training_stages'])) + '.\n\n'
        missing = job['missing_cells']
        text += ('Missing cells: ' + '; '.join(f"stage {x['stage']}, {base._md(x['eval_task'])}" for x in missing)
                 if missing else 'Missing cells: none') + '.\n\n'
        text += '| Stage | Sequence | Scientific profile | World | Per-device batch | Accumulation | Effective batch | LR (argv) | Epochs (argv) | Manifest samples | Planned dataset examples | Planned optimizer steps | DS optimizer | DS scheduler |\n'
        text += '| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n'
        training = profile.get('training') or []
        if not training:
            text += '| ' + ' | '.join(['N/A'] * 14) + ' |\n'
            writer.writerow({key: ({'method': job['method'], 'benchmark': job['benchmark'], 'kind': 'training'}
                                   .get(key, 'N/A')) for key in fields})
        for item in training:
            ds = item.get('deepspeed_metadata') or {}
            values = [item.get('task'), item.get('sequence'), item['scientific_profile']['id'], item.get('world_size'), item.get('per_device_batch_size'),
                      item.get('gradient_accumulation_steps'), item.get('effective_batch_size'),
                      item.get('learning_rate_recorded') if item.get('learning_rate') is not None else None,
                      item.get('num_train_epochs_recorded') if item.get('num_train_epochs') is not None else None,
                      item.get('planned_train_samples'), item.get('planned_dataset_examples'), item.get('planned_optimizer_steps'),
                      ds.get('optimizer_type'), ds.get('scheduler_type')]
            text += '| ' + ' | '.join(base._md(value) for value in values) + ' |\n'
            row = {**item, 'method': job['method'], 'benchmark': job['benchmark'], 'kind': 'training',
                   'stage': item.get('task'), 'scientific_profile_id': item['scientific_profile']['id'],
                   'deepspeed_optimizer_type': ds.get('optimizer_type'), 'deepspeed_scheduler_type': ds.get('scheduler_type'),
                   'state_execution_profile': profile['state_execution_profile'],
                   'commands_execution_profile': profile['commands_execution_profile'],
                   'command_execution_profile': item.get('execution_profile'),
                   'uniform_experiment_id': sidecar.get('experiment_id')}
            writer.writerow({key: row.get(key) if row.get(key) is not None else 'N/A' for key in fields})
        preparation = profile.get('gradient_preparation') or []
        if preparation:
            text += '\nGradient preparation is recorded separately from main and router training.\n'
        for item in preparation:
            row = {**item, 'method': job['method'], 'benchmark': job['benchmark'], 'kind': 'gradient_preparation',
                   'stage': item.get('task'), 'state_execution_profile': profile['state_execution_profile'],
                   'commands_execution_profile': profile['commands_execution_profile'],
                   'command_execution_profile': item.get('execution_profile')}
            writer.writerow({key: row.get(key) if row.get(key) is not None else 'N/A' for key in fields})
    atomic_write(markdown, text)
    profile_csv = Path(output_dir).expanduser().resolve() / 'training_profiles.csv'
    atomic_write(profile_csv, csv_handle.getvalue())
    paths['training_profiles_csv'] = str(profile_csv)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run-dir', type=Path, help='one validation queue directory containing state.json')
    source.add_argument('--evaluation-plan', type=Path, help='one MR UCIT evaluation-only plan JSON')
    parser.add_argument('--execution-state', type=Path, help='evaluation-only controller state bound to this plan')
    parser.add_argument('--method', help='optional method filter, or mrlora for evaluation-only')
    parser.add_argument('--benchmark', help='optional benchmark filter, or ucit for evaluation-only')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        method = get_method(args.method).id if args.method else None
        benchmark = get_benchmark(args.benchmark).id if args.benchmark else None
        if args.run_dir:
            if args.execution_state:
                raise ValueError('--execution-state applies only to --evaluation-plan')
            report = queue_report(args.run_dir, method, benchmark)
        else:
            if method not in {None, 'mrlora'} or benchmark not in {None, 'ucit'}:
                raise ValueError('this evaluation-only plan is MR-LoRA UCIT')
            report = evaluation_report(args.evaluation_plan, args.execution_state)
        paths = write_summary(report, args.output_dir)
    except (KeyError, ValueError, OSError, TypeError, json.JSONDecodeError) as error:
        print(f'unified report: {error}', file=sys.stderr)
        return 2
    print(json.dumps(paths, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
