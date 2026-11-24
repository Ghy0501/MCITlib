# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 MCITlib authors
"""Validate the actual PCLR query and checkpoint/stage prerequisites."""
from __future__ import annotations

import json
from pathlib import Path


def check_pclr(report, commands):
    from ...doctor import FAIL, OK

    errors, datasets, produced, learned, integrated = [], set(), set(), {}, set()

    def value(command, flag):
        if command.argv.count(flag) != 1:
            raise ValueError(f'task {command.task_id}: expected one {flag}')
        return command.argv[command.argv.index(flag) + 1]

    def checkpoint(directory, expected_tasks, expected_phase=None):
        if directory in produced:
            return
        path = Path(directory) / 'pclr.pt'
        if not path.is_file():
            errors.append(f'missing prior PCLR boundary {path}')
            return
        try:
            import torch
            data = torch.load(path, map_location='cpu', weights_only=True)
            state = data['state']
            if (data['format'] != 'mcitlib-pclr-v1' or not data['targets'] or
                    state['task_text'].shape[0] != expected_tasks or
                    state['task_vision'].shape[0] != expected_tasks or
                    (expected_phase is not None and data['phase'] != expected_phase)):
                raise ValueError('format, phase, targets, or task-query history differs')
        except Exception as error:
            errors.append(f'invalid PCLR boundary {path}: {error}')

    for command in commands:
        if command.env.get('MCITLIB_METHOD') != 'pclr':
            continue
        try:
            if command.kind == 'train':
                stage = command.env.get('MCITLIB_PCLR_STAGE')
                if int(command.env['MCITLIB_PCLR_TASK_ID']) != command.task_id:
                    raise ValueError('PCLR command task identity differs from its checkpoint history')
                opts = json.loads(command.env['MCITLIB_PCLR_OPTIONS'])
                if set(opts) != {'new_rank', 'top_rank', 'alpha', 'share_rank', 'skip_interval'}:
                    raise ValueError('PCLR pool settings are incomplete')
                output = value(command, '--output_dir')
                if stage == 'learn':
                    data, folder = value(command, '--data_path'), value(command, '--image_folder')
                    identity = (data, folder)
                    if identity not in datasets:
                        datasets.add(identity)
                        if not Path(data).is_file() or not Path(folder).is_dir():
                            errors.append(f'missing PCLR query data or image root: {data} / {folder}')
                        else:
                            with open(data, encoding='utf-8') as handle:
                                rows = json.load(handle)
                            if not isinstance(rows, list) or not rows:
                                raise ValueError(f'{data}: query dataset must be a nonempty JSON list')
                            for index, row in enumerate(rows):
                                conversations = row.get('conversations') if isinstance(row, dict) else None
                                if not isinstance(conversations, list) or not any(
                                    isinstance(item, dict) and item.get('from') == 'human' and
                                    isinstance(item.get('value'), str) and item['value'].strip()
                                    for item in conversations
                                ):
                                    errors.append(f'{data}: record {index} has no human instruction')
                                    break
                                if 'image' in row and (not isinstance(row['image'], str) or not row['image'] or
                                                       not (Path(folder) / row['image']).is_file()):
                                    errors.append(f'{data}: record {index} has missing/unsupported image {row["image"]!r}')
                                    break
                    previous = value(command, '--previous_task_model_path') if '--previous_task_model_path' in command.argv else None
                    if command.task_id and command.task_id > 1:
                        if not previous:
                            errors.append(f'task {command.task_id}: missing prior checkpoint')
                        else:
                            checkpoint(previous, command.task_id - 1)
                    elif previous:
                        errors.append('task 1 unexpectedly reads a prior PCLR checkpoint')
                    learned[command.task_id] = output
                elif stage == 'integrate':
                    previous = value(command, '--previous_task_model_path')
                    if previous != learned.get(command.task_id):
                        errors.append(f'task {command.task_id}: integration does not follow its learning output')
                    checkpoint(previous, command.task_id, 'learned')
                    if float(value(command, '--learning_rate')) <= 0 or float(value(command, '--num_train_epochs')) <= 0:
                        errors.append(f'task {command.task_id}: invalid integration training budget')
                    integrated.add(command.task_id)
                else:
                    errors.append(f'task {command.task_id}: unknown PCLR stage {stage!r}')
                produced.add(output)
            elif command.kind == 'eval' and '--model-path' in command.argv:
                checkpoint(value(command, '--model-path'), command.task_id)
        except (ValueError, KeyError, OSError, IndexError, json.JSONDecodeError) as error:
            errors.append(f'PCLR prerequisite: {error}')
    for task, output in learned.items():
        if (task is not None and task >= 4 or output.endswith('/learning')) and task not in integrated:
            errors.append(f'task {task}: expected compression/integration stage is absent')
    report.add(FAIL if errors else OK, 'PCLR query data, task history, and stages',
               '; '.join(errors[:8]) if errors else f'{len(datasets)} complete training datasets and all stage dependencies checked',
               'restore task data/images and matching PCLR boundary checkpoints; do not bypass consolidation')
