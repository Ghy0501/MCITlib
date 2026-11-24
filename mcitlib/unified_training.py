# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Load and audit a uniform LLaVA main-training profile without executing it."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml

_FIELDS = {
    'id', 'backbone', 'primary_train_samples_per_task', 'evaluation_samples_per_task',
    'global_batch_size', 'max_per_device_train_batch_size', 'base_learning_rate',
    'primary_default_epochs', 'method_primary_epoch_overrides',
    'recorded_warmup_ratio', 'recorded_lr_scheduler_type', 'recorded_model_max_length',
    'recorded_mm_projector_lr',
    'replay_memory', 'auxiliary_learning_rates', 'method_specific_parameters',
    'auxiliary_training_sequences',
}


@dataclass(frozen=True)
class TrainingProfile:
    id: str
    path: str
    sha256: str
    primary_train_samples_per_task: int
    evaluation_samples_per_task: int
    global_batch_size: int
    max_per_device_train_batch_size: int
    base_learning_rate: str
    primary_default_epochs: int
    method_primary_epoch_overrides: tuple[tuple[str, int, str], ...]
    recorded_warmup_ratio: str
    recorded_lr_scheduler_type: str
    recorded_model_max_length: int
    recorded_mm_projector_lr: str


def load_training_profile(path: str | Path, *, repo_root: str | Path) -> TrainingProfile:
    """Read a profile and bind its exact bytes to a project-relative path."""
    root, source = Path(repo_root).resolve(), Path(path).resolve()
    if root not in source.parents:
        raise ValueError('training profile must be inside its project')
    raw = source.read_bytes()
    data = yaml.safe_load(raw)
    if not isinstance(data, dict) or set(data) != _FIELDS:
        raise ValueError('training profile fields differ from the supported schema')
    if not isinstance(data['id'], str) or not data['id'] or data['backbone'] != 'llava':
        raise ValueError('training profile must identify an LLaVA profile')
    for name in ('primary_train_samples_per_task', 'evaluation_samples_per_task', 'global_batch_size',
                 'max_per_device_train_batch_size', 'primary_default_epochs', 'recorded_model_max_length'):
        if type(data[name]) is not int or data[name] <= 0:
            raise ValueError(f'{name} must be a positive integer')
    overrides = data['method_primary_epoch_overrides']
    if (not isinstance(overrides, dict) or
            any(not isinstance(method, str) or not method or not isinstance(value, dict) or
                set(value) != {'epochs', 'reason'} or type(value['epochs']) is not int or value['epochs'] <= 0 or
                not isinstance(value['reason'], str) or not value['reason']
                for method, value in overrides.items())):
        raise ValueError('method primary epoch overrides must declare positive epochs and a reason')
    for name in ('replay_memory', 'auxiliary_learning_rates', 'method_specific_parameters',
                 'auxiliary_training_sequences'):
        if data[name] != 'preserve':
            raise ValueError(f'{name} must explicitly preserve the method behavior')
    for name in ('base_learning_rate', 'recorded_warmup_ratio', 'recorded_mm_projector_lr'):
        if not isinstance(data[name], str):
            raise ValueError(f'{name} must be a quoted decimal string')
        try:
            value = Decimal(data[name])
        except InvalidOperation as error:
            raise ValueError(f'invalid {name}') from error
        if not value.is_finite() or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if not isinstance(data['recorded_lr_scheduler_type'], str) or not data['recorded_lr_scheduler_type']:
        raise ValueError('recorded CLI scheduler must be a nonempty string')
    return TrainingProfile(
        id=data['id'], path=str(source.relative_to(root)), sha256=hashlib.sha256(raw).hexdigest(),
        primary_train_samples_per_task=data['primary_train_samples_per_task'],
        evaluation_samples_per_task=data['evaluation_samples_per_task'],
        global_batch_size=data['global_batch_size'],
        max_per_device_train_batch_size=data['max_per_device_train_batch_size'],
        base_learning_rate=data['base_learning_rate'], primary_default_epochs=data['primary_default_epochs'],
        method_primary_epoch_overrides=tuple((name, row['epochs'], row['reason'])
                                             for name, row in sorted(overrides.items())),
        recorded_warmup_ratio=data['recorded_warmup_ratio'],
        recorded_lr_scheduler_type=data['recorded_lr_scheduler_type'],
        recorded_model_max_length=data['recorded_model_max_length'],
        recorded_mm_projector_lr=data['recorded_mm_projector_lr'],
    )


def _position(argv: list[str], flag: str) -> int:
    if (argv.count(flag) != 1 or argv.index(flag) + 1 >= len(argv) or
            argv[argv.index(flag) + 1].startswith('--')):
        raise ValueError(f'expected one value for {flag}')
    return argv.index(flag) + 1


def _read_array(path: str | Path, expected_sha256: str | None = None) -> tuple[list, str]:
    path = Path(path)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ValueError(f'training data is unavailable or unsafe: {path}')
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f'training data differs from its manifest: {path}')
    value = json.loads(raw)
    if not isinstance(value, list):
        raise ValueError(f'training data is not a JSON array: {path}')
    return value, digest


def _same_replay_current(primary: list, replay: list, primary_root: Path, replay_root: Path,
                         *, absolute_replay: bool = False) -> bool:
    """Permit only image path rewriting that resolves to the same actual file."""
    if not primary_root.is_absolute() or not replay_root.is_absolute() or not primary_root.is_dir() or not replay_root.is_dir():
        raise ValueError('reviewed primary and Replay image roots must exist')
    for original, observed in zip(primary, replay):
        if not isinstance(original, dict) or not isinstance(observed, dict):
            return False
        if {k: v for k, v in original.items() if k != 'image'} != {k: v for k, v in observed.items() if k != 'image'}:
            return False
        left, right = original.get('image'), observed.get('image')
        if left is None and right is None:
            continue
        if not isinstance(left, str) or not isinstance(right, str):
            return False
        left_path, right_path = Path(left), Path(right)
        if (left_path.is_absolute() or (right_path.is_absolute() and not absolute_replay) or
                '..' in left_path.parts or '..' in right_path.parts):
            return False
        if (primary_root / left_path).resolve(strict=True) != (replay_root / right_path).resolve(strict=True):
            return False
    return True


def training_data_evidence(command: Any, *, primary_path: str | Path, primary_sha256: str,
                           expected_primary_samples: int, replay_sha256: str | None = None,
                           expected_replay_previous: int | None = None,
                           primary_image_root: str | Path | None = None,
                           memory_sha256: str | None = None, expected_memory_samples: int | None = None) -> dict[str, Any]:
    """Verify actual input bytes, including the precomposed Replay file when used.

    The caller obtains the expected primary and replay hashes from its reviewed
    sample/auxiliary manifests. Replay starts with the verified primary records;
    rewritten image paths must resolve to the same files from the two roots.
    A separate --memory_data_path, if present, is counted as the LLaVA loader
    concatenates it.
    """
    if (command.kind != 'train' or 'mcitlib.backbones.llava.train_entry' not in command.argv or
            command.env.get('MCITLIB_TRAIN_ENTRY') not in {'standard', 'moe'}):
        raise ValueError('training dataset loader is not a reviewed LLaVA entry')
    if not isinstance(primary_sha256, str) or len(primary_sha256) != 64:
        raise ValueError('primary training data requires its reviewed SHA-256')
    primary_path = Path(primary_path)
    primary, primary_digest = _read_array(primary_path, primary_sha256)
    if len(primary) != expected_primary_samples:
        raise ValueError('primary training sample count differs from the profile')
    data_path = Path(command.argv[_position(command.argv, '--data_path')])
    data, data_digest = _read_array(data_path)
    method = command.env.get('MCITLIB_METHOD')
    if method == 'replay' or (method == 'dgg' and command.task_id != 1):
        if (not isinstance(replay_sha256, str) or len(replay_sha256) != 64 or
                type(expected_replay_previous) is not int or expected_replay_previous < 0):
            raise ValueError('precomposed Replay requires reviewed auxiliary evidence')
        if primary_image_root is None:
            raise ValueError('Replay requires its reviewed primary image root')
        replay_root = Path(command.argv[_position(command.argv, '--image_folder')])
        if (data_digest != replay_sha256 or len(data) != len(primary) + expected_replay_previous or
                not _same_replay_current(primary, data[:len(primary)], Path(primary_image_root), replay_root,
                                         absolute_replay=method == 'dgg')):
            raise ValueError('Replay data differs from the reviewed primary and previous samples')
    elif data_path != primary_path or data_digest != primary_digest:
        raise ValueError('main training does not use the reviewed primary sample file')
    files = [{'flag': '--data_path', 'path': str(data_path), 'sha256': data_digest, 'samples': len(data)}]
    total = len(data)
    if '--memory_data_path' in command.argv:
        if (not isinstance(memory_sha256, str) or len(memory_sha256) != 64 or
                type(expected_memory_samples) is not int or expected_memory_samples < 0):
            raise ValueError('separate memory data requires reviewed hash and count')
        memory_path = Path(command.argv[_position(command.argv, '--memory_data_path')])
        memory, memory_digest = _read_array(memory_path, memory_sha256)
        if len(memory) != expected_memory_samples:
            raise ValueError('memory sample count differs from its reviewed manifest')
        files.append({'flag': '--memory_data_path', 'path': str(memory_path), 'sha256': memory_digest,
                      'samples': len(memory)})
        total += len(memory)
    return {
        'primary_path': str(primary_path), 'primary_sha256': primary_digest,
        'primary_samples': len(primary), 'precomposed_replay_samples': len(data) - len(primary),
        'files': files, 'total_samples': total,
    }


def planned_steps(samples: int, world_size: int, microbatch: int, accumulation: int,
                  epochs: int = 1) -> dict[str, int]:
    """Calculate current HF/Accelerate total step budget for integer epochs.

    Assumes a finite map-style dataset, drop_last=False and Accelerate
    even_batches=True. HF uses floor(loader/accumulation) as its per-epoch
    budget, while unfinished accumulation can cross epoch boundaries. Counts
    below are predictions, not observations of a completed training run.
    """
    if any(type(value) is not int or value <= 0 for value in
           (samples, world_size, microbatch, accumulation, epochs)):
        raise ValueError('sample and batch counts must be positive')
    divisor = world_size * microbatch
    loader = (samples + divisor - 1) // divisor
    budget_per_epoch = max(loader // accumulation, 1)
    updates = budget_per_epoch * epochs
    available_batches = loader * epochs
    used = min(available_batches, updates * accumulation)
    return {
        'training_examples': samples, 'resolved_epochs': epochs, 'planned_example_visits': samples * epochs,
        'world_size': world_size, 'per_device_microbatch': microbatch,
        'gradient_accumulation': accumulation, 'nominal_global_batch': world_size * microbatch * accumulation,
        'dataloader_steps_per_rank': loader, 'optimizer_steps_per_epoch': budget_per_epoch,
        'optimizer_steps': updates,
        'distributed_padding_examples_per_full_epoch': loader * divisor - samples,
        'assumed_unprocessed_microbatches_total': available_batches - used,
        'assumed_unprocessed_example_visits_lower_bound': max(samples * epochs - used * divisor, 0),
    }


def apply_training_profile(commands: list[Any], *, profile: TrainingProfile, world_size: int,
                           expected_method: str, expected_task_ids: tuple[int, ...],
                           data_specs_by_index: dict[int, dict[str, Any]]) -> tuple[list[Any], list[dict[str, Any]]]:
    """Copy and adjust main-training commands; leave router and other stages intact.

    Data specs, keyed by command index, must bind each main task to its reviewed
    primary file and, for Replay, the reviewed precomposed file. No model,
    method-specific, auxiliary LR, data path, router or evaluation argument is
    changed. Unsupported loaders, ambiguous flags and sample mismatches fail.
    """
    if type(world_size) is not int or world_size <= 0:
        raise ValueError('world size must be positive')
    if not isinstance(expected_method, str) or not expected_method:
        raise ValueError('expected method identity is required')
    if (not expected_task_ids or len(set(expected_task_ids)) != len(expected_task_ids) or
            any(type(task) is not int or task <= 0 for task in expected_task_ids)):
        raise ValueError('expected main task IDs must be distinct positive integers')
    main_indices = [i for i, c in enumerate(commands) if c.kind == 'train' and c.sequence == 'train']
    main_tasks = [commands[i].task_id for i in main_indices]
    if main_tasks != list(expected_task_ids) or set(data_specs_by_index) != set(main_indices):
        raise ValueError('main training task set or data evidence differs from the reviewed job')
    revised = copy.deepcopy(commands)
    audits = []
    for index, command in enumerate(revised):
        if command.kind != 'train':
            continue
        if command.env.get('MCITLIB_METHOD') != expected_method:
            raise ValueError('training command method differs from the reviewed job')
        if command.sequence != 'train':
            if command.sequence not in {'router', 'pclr_integration'}:
                raise ValueError(f'unknown training sequence: {command.sequence}')
            audits.append({'command_index': index, 'task': command.task_id, 'sequence': command.sequence,
                           'role': 'auxiliary_preserved', 'argv': list(command.argv)})
            continue
        if index not in data_specs_by_index:
            raise ValueError(f'primary sample evidence missing for command {index}')
        spec = data_specs_by_index[index]
        allowed = {'primary_path', 'primary_sha256', 'primary_image_root', 'replay_sha256',
                   'replay_previous_samples', 'memory_sha256', 'memory_samples'}
        if set(spec) - allowed:
            raise ValueError('unrecognized training data evidence fields')
        if command.env.get('MCITLIB_METHOD') not in {'replay', 'dgg'} and (
                spec.get('replay_sha256') is not None or spec.get('replay_previous_samples') is not None):
            raise ValueError('Replay sample evidence was supplied for another method')
        if '--memory_data_path' not in command.argv and (
                spec.get('memory_sha256') is not None or spec.get('memory_samples') is not None):
            raise ValueError('memory evidence was supplied without a memory input')
        evidence = training_data_evidence(
            command, primary_path=spec['primary_path'], primary_sha256=spec['primary_sha256'],
            expected_primary_samples=profile.primary_train_samples_per_task,
            replay_sha256=spec.get('replay_sha256'), expected_replay_previous=spec.get('replay_previous_samples'),
            primary_image_root=spec.get('primary_image_root'), memory_sha256=spec.get('memory_sha256'),
            expected_memory_samples=spec.get('memory_samples'),
        )
        argv = command.argv
        included = argv[_position(argv, '--include')].split(':')
        devices = included[1].split(',') if len(included) == 2 and included[0] == 'localhost' else []
        if (len(devices) != world_size or len(set(devices)) != world_size or
                any(not device.isdecimal() for device in devices)):
            raise ValueError('recorded DeepSpeed GPUs differ from the reviewed world size')
        if '--max_steps' in argv and int(argv[_position(argv, '--max_steps')]) > 0:
            raise ValueError('max_steps overrides the profile epoch budget')
        if ('--dataloader_drop_last' in argv and
                argv[_position(argv, '--dataloader_drop_last')].lower() not in {'false', '0'}):
            raise ValueError('drop_last changes the documented loader calculation')
        recorded = {name: argv[_position(argv, name)] for name in (
            '--warmup_ratio', '--lr_scheduler_type', '--model_max_length', '--mm_projector_lr')}
        try:
            if (Decimal(recorded['--warmup_ratio']) != Decimal(profile.recorded_warmup_ratio) or
                    recorded['--lr_scheduler_type'] != profile.recorded_lr_scheduler_type or
                    int(recorded['--model_max_length']) != profile.recorded_model_max_length or
                    Decimal(recorded['--mm_projector_lr']) != Decimal(profile.recorded_mm_projector_lr)):
                raise ValueError('recorded CLI parameters differ from the reviewed profile')
        except InvalidOperation as error:
            raise ValueError('invalid recorded CLI decimal parameter') from error
        ds_path = Path(argv[_position(argv, '--deepspeed')])
        if not ds_path.is_absolute() or not ds_path.is_file() or ds_path.is_symlink():
            raise ValueError('DeepSpeed configuration is unavailable')
        ds_raw = ds_path.read_bytes()
        ds_config = json.loads(ds_raw)
        if not isinstance(ds_config, dict):
            raise ValueError('DeepSpeed configuration must be an object')
        ds_evidence = {'path': str(ds_path), 'sha256': hashlib.sha256(ds_raw).hexdigest(),
                       'optimizer': ds_config.get('optimizer'), 'scheduler': ds_config.get('scheduler')}
        positions = {name: _position(argv, name) for name in (
            '--learning_rate', '--num_train_epochs', '--per_device_train_batch_size',
            '--gradient_accumulation_steps')}
        before = {name: argv[position] for name, position in positions.items()}
        previous_microbatch = int(before['--per_device_train_batch_size'])
        if previous_microbatch <= 0:
            raise ValueError('invalid original microbatch')
        microbatch = min(previous_microbatch, profile.max_per_device_train_batch_size)
        divisor = world_size * microbatch
        if profile.global_batch_size % divisor:
            raise ValueError('common batch cannot be represented by integer accumulation')
        accumulation = profile.global_batch_size // divisor
        for name, expected in (('train_micro_batch_size_per_gpu', microbatch),
                               ('gradient_accumulation_steps', accumulation),
                               ('train_batch_size', profile.global_batch_size)):
            if ds_config.get(name) not in ('auto', expected):
                raise ValueError(f'DeepSpeed {name} conflicts with the reviewed batch')
        if ds_config.get('optimizer') is not None:
            optimizer = ds_config['optimizer']
            if not isinstance(optimizer, dict) or not isinstance(optimizer.get('params'), dict):
                raise ValueError('DeepSpeed optimizer configuration is not auditable')
            optimizer_lr = optimizer['params'].get('lr')
            try:
                if optimizer_lr != 'auto' and Decimal(str(optimizer_lr)) != Decimal(profile.base_learning_rate):
                    raise ValueError('DeepSpeed optimizer LR conflicts with the reviewed base LR')
            except InvalidOperation as error:
                raise ValueError('DeepSpeed optimizer LR is invalid') from error
        argv[positions['--learning_rate']] = profile.base_learning_rate
        method = expected_method
        override = next(((epochs, reason) for name, epochs, reason in profile.method_primary_epoch_overrides
                         if name == method), None)
        epochs, reason = override or (profile.primary_default_epochs, 'default_sample_training_budget')
        argv[positions['--num_train_epochs']] = str(epochs)
        argv[positions['--per_device_train_batch_size']] = str(microbatch)
        argv[positions['--gradient_accumulation_steps']] = str(accumulation)
        if 'training_profile' in command.metadata:
            raise ValueError('training profile metadata is already present')
        command.metadata['effective_batch'] = world_size * microbatch * accumulation
        command.metadata['training_profile'] = {'id': profile.id, 'path': profile.path,
                                                'sha256': profile.sha256, 'role': 'main_train'}
        auxiliary_rates = {argv[i]: argv[i + 1] for i in range(len(argv) - 1)
                           if argv[i].startswith('--') and
                           argv[i].endswith(('_lr', '-lr', '_learning_rate', '-learning-rate')) and
                           argv[i] != '--learning_rate'}
        audits.append({
            'command_index': index, 'task': command.task_id, 'sequence': command.sequence, 'role': 'main_train',
            'before': before, 'after': {name: argv[position] for name, position in positions.items()},
            'epoch_reason': reason,
            'auxiliary_learning_rates_unchanged': auxiliary_rates, 'recorded_cli_parameters': recorded,
            'deepspeed_configuration': ds_evidence, 'training_data': evidence,
            **planned_steps(evidence['total_samples'], world_size, microbatch, accumulation, epochs),
        })
    return revised, audits
