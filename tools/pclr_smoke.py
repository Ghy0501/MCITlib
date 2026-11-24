# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 MCITlib authors
"""Bounded two-task PCLR learning, forced consolidation, and real generation.

This validation-only schedule compresses after task 2 instead of task 4. It is
not a training recipe for reporting paper or full-benchmark results.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

from mcitlib.gpus import parse_gpus
from mcitlib.paths import PathResolver, process_env, repo_root
from mcitlib.runner import build_sequence_commands
from mcitlib.smoke import _first_records, _validate_records


def value(command, flag):
    return command.argv[command.argv.index(flag) + 1]


def set_value(command, flag, val):
    if flag in command.argv:
        command.argv[command.argv.index(flag) + 1] = str(val)
    else:
        command.argv.extend([flag, str(val)])


def configure_validation_train(command, max_steps):
    """Bound this diagnostic without changing either normal training profile."""
    if max_steps < 2:
        raise ValueError('PCLR smoke requires at least two optimizer steps')
    for flag, val in (('--per_device_train_batch_size', 1), ('--per_device_eval_batch_size', 1),
                      ('--gradient_accumulation_steps', 1), ('--dataloader_num_workers', 0),
                      ('--save_strategy', 'no'), ('--max_steps', max_steps), ('--warmup_ratio', 0),
                      ('--logging_steps', 1), ('--report_to', 'none')):
        set_value(command, flag, val)


def launch_env(command, root, output, gpus):
    """Launch the selected physical GPUs and pin generation to the first one."""
    ids = parse_gpus(gpus)
    if ids is None or len(ids) != 2:
        raise ValueError('PCLR smoke requires two distinct physical GPU IDs')
    argv = command['argv']
    if not argv:
        raise ValueError('PCLR smoke command must not be empty')
    env = process_env(command['env'], root)
    if argv[0] == 'deepspeed':
        if (argv.count('--include') != 1 or argv.index('--include') + 1 >= len(argv) or
                argv[argv.index('--include') + 1] != 'localhost:' + ','.join(ids)):
            raise ValueError('PCLR smoke DeepSpeed inclusion differs from the selected GPUs')
        # The launcher maps physical IDs itself. Inheriting CUDA visibility here
        # would remap them to logical 0/1 before --include is interpreted.
        env.pop('CUDA_VISIBLE_DEVICES', None)
    elif command['env'].get('CUDA_VISIBLE_DEVICES') != ids[0]:
        raise ValueError('PCLR smoke inference must be pinned to the first selected GPU')
    env.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_HOME=str(output / 'cache/huggingface'), XDG_CACHE_HOME=str(output / 'cache/xdg'),
               TRITON_CACHE_DIR=str(output / 'cache/triton'),
               TORCH_EXTENSIONS_DIR=str(output / 'cache/torch_extensions'))
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
    return env


def learning_evidence(checkpoint, expected_steps):
    """Prove a learning stage changed its zero-initialized new B banks."""
    import torch

    checkpoint = Path(checkpoint)
    state = torch.load(checkpoint / 'pclr.pt', map_location='cpu')
    trainer = json.loads((checkpoint / 'trainer_state.json').read_text(encoding='utf-8'))
    if state.get('phase') != 'learned' or trainer.get('global_step', 0) < expected_steps:
        raise ValueError('PCLR learning checkpoint did not complete the expected optimizer steps')
    banks = [value for name, value in state['state'].items() if name.endswith('.new_b')]
    if not banks or not all(torch.isfinite(bank).all().item() for bank in banks):
        raise ValueError('PCLR new B banks are missing or non-finite')
    nonzero = sum(int(torch.count_nonzero(bank)) for bank in banks)
    if not nonzero:
        raise ValueError('PCLR learning made no observable new B update')
    return {'phase': 'learned', 'global_step': trainer['global_step'], 'new_b_tensors': len(banks),
            'new_b_elements': sum(bank.numel() for bank in banks), 'new_b_nonzero': nonzero,
            'new_b_max_abs': max(float(bank.abs().max()) for bank in banks if bank.numel())}


def integration_evidence(learned_checkpoint, integrated_checkpoint, expected_steps):
    """Compare the integrated student with the exact pre-optimization compression."""
    import torch

    from mcitlib.methods._pclr.core import retained_indices

    learned_checkpoint, integrated_checkpoint = Path(learned_checkpoint), Path(integrated_checkpoint)
    teacher = torch.load(learned_checkpoint / 'pclr.pt', map_location='cpu')
    student = torch.load(integrated_checkpoint / 'pclr.pt', map_location='cpu')
    trainer = json.loads((integrated_checkpoint / 'trainer_state.json').read_text(encoding='utf-8'))
    if (teacher.get('phase') != 'learned' or student.get('phase') != 'integrated' or
            trainer.get('global_step', 0) < expected_steps):
        raise ValueError('PCLR integration checkpoint did not complete the expected optimizer steps')
    interval = int(student['config']['skip_interval'])
    source, target = teacher['state'], student['state']
    keys = [name for name in source if name.endswith('.static_b')]
    changed = shared_nonzero = elements = 0
    max_diff = 0.0
    for key in keys:
        prefix = key.removesuffix('static_b')
        old, new = source[key], source[prefix + 'new_b']
        expected = torch.cat((old[retained_indices(old.shape[0], interval)],
                              new[retained_indices(new.shape[0], interval)]), dim=0)
        actual, shared = target[key], target[prefix + 'shared_b']
        if (expected.shape != actual.shape or not torch.isfinite(actual).all().item() or
                not torch.isfinite(shared).all().item()):
            raise ValueError(f'invalid PCLR compressed student bank: {key}')
        changed += int(torch.count_nonzero(actual != expected))
        shared_nonzero += int(torch.count_nonzero(shared))
        elements += actual.numel()
        if actual.numel():
            max_diff = max(max_diff, float((actual.float() - expected.float()).abs().max()))
    if not keys or not (changed or shared_nonzero):
        raise ValueError('PCLR integration made no observable student or shared B update')
    return {'phase': 'integrated', 'global_step': trainer['global_step'], 'skip_interval': interval,
            'compressed_b_tensors': len(keys), 'compressed_b_elements': elements,
            'compressed_b_changed': changed, 'compressed_b_max_abs_change': max_diff,
            'shared_b_nonzero': shared_nonzero}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--paths', type=Path, required=True)
    p.add_argument('--data-project-root', type=Path, required=True,
                   help='root relative to which the paths file resolves sample mappings')
    p.add_argument('--run-dir', type=Path, required=True, help='new isolated output directory')
    p.add_argument('--benchmark', choices=('ucit', 'mllm_dcl'), default='ucit')
    p.add_argument('--gpus', default='0,1')
    p.add_argument('--master-port', type=int, default=29681)
    p.add_argument('--samples', type=int, default=2, choices=range(1, 9))
    p.add_argument('--max-steps', type=int, default=2, choices=range(2, 4))
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    ids = parse_gpus(args.gpus)
    if ids is None or len(ids) != 2:
        raise ValueError('PCLR smoke requires two distinct physical GPU IDs')
    args.gpus = ','.join(ids)
    root = repo_root()
    resolver = PathResolver.from_file(args.paths, repo_root=args.data_project_root.resolve())
    output = args.run_dir.resolve()
    if output.exists() or output.is_symlink():
        raise ValueError(f'run directory already exists: {output}')
    if not output.is_relative_to(root.parent.parent):
        raise ValueError('run directory must be inside the enclosing control workspace')
    records, trains = [], []
    for task in (1, 2):
        plan = build_sequence_commands(root / 'configs', 'llava', 'pclr', args.benchmark, paths=resolver,
                                       task_id=task, gpus=args.gpus, master_port=args.master_port)
        train = next(c for c in plan if c.kind == 'train' and c.sequence == 'train')
        source, folder = Path(value(train, '--data_path')), Path(value(train, '--image_folder'))
        rows = _first_records(source, args.samples)
        _validate_records(rows, folder)
        records.append((rows, folder))
        set_value(train, '--data_path', output / f'task{task}-data.json')
        set_value(train, '--output_dir', output / ('Task1_llava_lora' if task == 1 else 'Task2_llava_lora/learning'))
        configure_validation_train(train, args.max_steps)
        if task == 2:
            set_value(train, '--previous_task_model_path', output / 'Task1_llava_lora')
        train.env.update(TRITON_CACHE_DIR=str(output / 'cache/triton'),
                         TORCH_EXTENSIONS_DIR=str(output / 'cache/torch_extensions'))
        trains.append(train)
    integrate = copy.deepcopy(trains[1])
    integrate.sequence = 'pclr_integration'
    integrate.env['MCITLIB_PCLR_STAGE'] = 'integrate'
    integrate.metadata['pclr_phase'] = 'compression_integration'
    set_value(integrate, '--previous_task_model_path', output / 'Task2_llava_lora/learning')
    set_value(integrate, '--output_dir', output / 'Task2_llava_lora')
    set_value(integrate, '--learning_rate', '5e-5')
    set_value(integrate, '--num_train_epochs', '1')
    plan = {'schedule': 'validation-only compression after task 2', 'samples_per_task': args.samples,
            'max_steps_per_stage': args.max_steps, 'validation_only_warmup_ratio': 0,
            'train': [x.as_dict() for x in [*trains, integrate]],
            'inference': []}
    base = value(trains[0], '--model_name_or_path')
    first_gpu = args.gpus.split(',')[0]
    for task, (_, folder) in enumerate(records, 1):
        cmd = [sys.executable, '-m', 'mcitlib.backbones.llava.eval_runners.entry',
               'mcitlib.methods._pclr.smoke_infer', '--benchmark', args.benchmark,
               '--adapter', str(output / 'Task2_llava_lora'), '--model-base', base,
               '--subset', str(output / f'task{task}-data.json'), '--image-folder', str(folder),
               '--output', str(output / f'prediction-task{task}.json')]
        plan['inference'].append({'argv': cmd, 'env': {'MCITLIB_METHOD': 'pclr', 'CUDA_VISIBLE_DEVICES': first_gpu}})
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return
    output.mkdir(parents=True, exist_ok=False)
    for task, (rows, _) in enumerate(records, 1):
        (output / f'task{task}-data.json').write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')
    (output / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    for index, command in enumerate(plan['train'] + plan['inference'], 1):
        argv = list(command['argv'])
        env = launch_env(command, root, output, args.gpus)
        with (output / f'stage-{index}.log').open('x', encoding='utf-8') as log:
            subprocess.run(argv, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        if index > 3:
            continue
        evidence_path = output / f'stage-{index}.evidence.json'
        try:
            if index in (1, 2):
                evidence = learning_evidence(Path(value(trains[index - 1], '--output_dir')), args.max_steps)
            else:
                evidence = integration_evidence(output / 'Task2_llava_lora/learning', output / 'Task2_llava_lora',
                                                args.max_steps)
        except Exception as error:
            evidence_path.write_text(json.dumps({'status': 'failed', 'error': str(error)}, indent=2) + '\n',
                                     encoding='utf-8')
            raise
        evidence_path.write_text(json.dumps({'status': 'verified', **evidence}, indent=2) + '\n', encoding='utf-8')
    print(f'PCLR smoke complete: {output}')


if __name__ == '__main__':
    main()
