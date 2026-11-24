# SPDX-License-Identifier: Apache-2.0
"""Apply and record the explicit InternVL four-GPU sample budget."""
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


def _read(root):
    path = Path(root) / 'training_profiles/internvl_sample_unified.yaml'
    raw = path.read_bytes()
    value = yaml.safe_load(raw)
    expected = {'id': 'internvl_sample_unified', 'backbone': 'internvl', 'world_size': 4,
                'global_batch_size': 64, 'microbatch': 1, 'learning_rate': '0.0002',
                'main_epochs': 1, 'modalprompt_epochs': 1, 'router_policy': 'preserve'}
    if value != expected:
        raise ValueError('InternVL sample training profile differs from the reviewed schema')
    return value, hashlib.sha256(raw).hexdigest(), str(path.relative_to(Path(root).parent))


def _value(argv, name):
    if argv.count(name) != 1 or argv.index(name) + 1 >= len(argv):
        raise ValueError(f'InternVL sample training requires exactly one {name}')
    return argv[argv.index(name) + 1]


def _set(argv, name, value):
    _value(argv, name)
    argv[argv.index(name) + 1] = str(value)


def prepare_sample_training(commands, root, method, sequence):
    """Set main-training values before four-GPU batch-preserving conversion."""
    profile, sha256, path = _read(root)
    for command in commands:
        if command.kind != 'train':
            continue
        argv = command.argv
        host, devices = _value(argv, '--include').split(':', 1)
        original_world = len(devices.split(','))
        if host != 'localhost' or original_world < 1 or profile['global_batch_size'] % original_world:
            raise ValueError('cannot represent InternVL sample batch on original launcher')
        original = {flag: _value(argv, flag) for flag in (
            '--per_device_train_batch_size', '--gradient_accumulation_steps', '--learning_rate', '--num_train_epochs')}
        if sequence == 'train':
            _set(argv, '--per_device_train_batch_size', profile['microbatch'])
            _set(argv, '--gradient_accumulation_steps', profile['global_batch_size'] // original_world)
            _set(argv, '--learning_rate', profile['learning_rate'])
            _set(argv, '--num_train_epochs', profile['modalprompt_epochs'] if method == 'modalprompt'
                 else profile['main_epochs'])
        else:
            # Preserve the router's effective batch and its source LR/epochs,
            # while avoiding an unnecessary per-device activation spike.
            original_global = (original_world * int(original['--per_device_train_batch_size']) *
                               int(original['--gradient_accumulation_steps']))
            _set(argv, '--per_device_train_batch_size', 1)
            _set(argv, '--gradient_accumulation_steps', original_global // original_world)
        key = 'training_profile' if sequence == 'train' else 'auxiliary_training_profile'
        command.metadata[key] = {'id': profile['id'], 'sha256': sha256, 'path': path,
            'backbone': 'internvl', 'role': 'main_train' if sequence == 'train' else 'auxiliary_preserved',
            'source_values': original}


def finalize_sample_training(commands):
    """Bind metadata to the actual four-GPU argv, including preserved router work."""
    for command in commands:
        if command.kind != 'train':
            continue
        metadata = command.metadata.get('training_profile') or command.metadata.get('auxiliary_training_profile')
        if not metadata:
            raise ValueError('missing InternVL sample training profile metadata')
        argv = command.argv
        world = len(_value(argv, '--include').split(':', 1)[1].split(','))
        microbatch = int(_value(argv, '--per_device_train_batch_size'))
        accumulation = int(_value(argv, '--gradient_accumulation_steps'))
        metadata.update(world_size=world, microbatch=microbatch, gradient_accumulation=accumulation,
                        global_batch_size=world * microbatch * accumulation,
                        learning_rate=_value(argv, '--learning_rate'), epochs=_value(argv, '--num_train_epochs'))
        if world != 4 or (metadata['role'] == 'main_train' and metadata['global_batch_size'] != 64):
            raise ValueError('InternVL sample training budget is not four-GPU global batch 64')
