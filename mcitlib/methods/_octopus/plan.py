# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 MCITlib authors
"""Build Octopus stages from a resolved MCITlib task command."""
from __future__ import annotations

import copy
import json
from pathlib import PurePosixPath

from ...config.schema import Command


def _value(argv, flag):
    if argv.count(flag) != 1:
        raise ValueError(f'Octopus needs exactly one {flag}')
    return argv[argv.index(flag) + 1]


def _set(argv, flag, value):
    if flag in argv:
        if argv.count(flag) != 1:
            raise ValueError(f'duplicate {flag}')
        argv[argv.index(flag) + 1] = str(value)
    else:
        argv.extend([flag, str(value)])


def task_paths(stage2_output, task, aux_root=None):
    token = '/Stage2/'
    if stage2_output.count(token) != 1:
        raise ValueError('Octopus checkpoint must be under Stage2')
    stage1 = stage2_output.replace(token, '/Stage1/')
    parent = PurePosixPath(stage2_output).parent
    history = [str(parent / f'Task{i}_llava_lora') for i in range(1, task)]
    if aux_root is not None:
        if not isinstance(aux_root, str) or not aux_root.strip() or '..' in PurePosixPath(aux_root).parts:
            raise ValueError('Octopus auxiliary root must be a nonempty path without parent traversal')
        auxiliary = PurePosixPath(aux_root)
    else:
        auxiliary = parent
    return stage1, history, str(auxiliary / f'task{task}_stage2_data.json'), str(auxiliary / f'task{task}_gpwc.pt')


def train_stages(train: Command, config: dict, task: int) -> list[Command]:
    """Build stage 1, preparation/GPWC, stage 2 and temporary cleanup."""
    output = _value(train.argv, '--output_dir')
    stage1, history, sample, gpwc = task_paths(output, task, config.get('octopus_aux_root'))
    source = _value(train.argv, '--data_path')
    folder = _value(train.argv, '--image_folder')
    model = _value(train.argv, '--model_name_or_path')
    vision = _value(train.argv, '--vision_tower')
    order = train.metadata['order']
    _set(train.argv, '--output_dir', stage1)
    train.env['MCITLIB_OCTOPUS_STAGE'] = '1'
    train.metadata['octopus_stage'] = 'stage1'
    commands = [train]
    if task == 1:
        commands.append(Command(argv=['python', '-m', 'mcitlib.methods._octopus.workflow', 'copy-first',
                                      '--input', stage1, '--output_dir', output], kind='extra', task_id=task,
                                metadata={'order': order + .1, 'octopus_inputs': [stage1], 'octopus_stage': 'copy'}))
        return commands
    samples = int(config['octopus_stage2_samples'])
    if samples < 1 or not history:
        raise ValueError('Octopus stage 2 requires history and a positive sample count')
    commands.append(Command(argv=['python', '-m', 'mcitlib.methods._octopus.workflow', 'sample',
                                  '--data_path', source, '--count', str(samples), '--output', sample],
                            kind='extra', task_id=task, metadata={'order': order + .1, 'octopus_stage': 'sample'}))
    argv = ['python', '-m', 'mcitlib.methods._octopus.gpwc', '--model-base', model, '--vision-tower', vision,
            '--data_path', source, '--image_folder', folder, '--samples', '256', '--logical-batch', '16',
            '--microbatch', '1', '--output', gpwc]
    for path in history:
        argv.extend(['--history', path])
    commands.append(Command(argv=argv, kind='extra', task_id=task,
                            metadata={'order': order + .2, 'octopus_stage': 'gpwc', 'octopus_inputs': history}))
    second = copy.deepcopy(train)
    second.argv = list(train.argv)
    _set(second.argv, '--output_dir', output)
    _set(second.argv, '--previous_task_model_path', stage1)
    _set(second.argv, '--data_path', sample)
    # Stage 2 has its own method budget even when the main-stage benchmark
    # uses task-specific full-data epoch compensation.
    _set(second.argv, '--num_train_epochs', config['octopus_stage2_epochs'])
    _set(second.argv, '--learning_rate', config['octopus_stage2_lr'])
    second.env.update(MCITLIB_OCTOPUS_STAGE='2', MCITLIB_OCTOPUS_HISTORY=json.dumps(history),
                      MCITLIB_OCTOPUS_GPWC=gpwc, MCITLIB_OCTOPUS_LAMBDA1=str(config['octopus_lambda_1']),
                      MCITLIB_OCTOPUS_LAMBDA2=str(config['octopus_lambda_2']))
    second.metadata.update(order=order + .3, octopus_stage='stage2', octopus_inputs=[stage1, sample, gpwc, *history])
    commands.append(second)
    commands.append(Command(argv=['python', '-m', 'mcitlib.methods._octopus.workflow', 'cleanup', '--input', gpwc,
                                  '--checkpoint', output, '--receipt', gpwc + '.receipt.json'],
                            kind='extra', task_id=task,
                            metadata={'order': 1000000, 'octopus_inputs': [gpwc, output], 'octopus_stage': 'cleanup'}))
    return commands
