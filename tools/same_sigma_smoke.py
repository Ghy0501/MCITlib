# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 MCITlib authors
"""Prepare and execute a two-task, two-GPU SAME/SiGMA integration smoke.

Preparation reads existing data and writes only under a new project run directory.
Execution is separate and runs exactly two optimizer steps and one generation per task.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from mcitlib.config.schema import PathSet
from mcitlib.paths import PathResolver, managed_path, process_env, repo_root
from mcitlib.runner import build_sequence_commands
from mcitlib.smoke import _first_records, _validate_records
from mcitlib.validation_batch import checkpoint_error


def option(argv, flag):
    if argv.count(flag) != 1:
        raise ValueError(f'expected one {flag}')
    return argv[argv.index(flag) + 1]


def set_option(argv, flag, value):
    if flag in argv:
        argv[argv.index(flag) + 1] = str(value)
    else:
        argv.extend([flag, str(value)])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args):
    root = repo_root()
    run = managed_path(args.run_dir, root, label='smoke run directory')
    if run.exists():
        raise FileExistsError(f'choose a new run directory: {run}')
    ids = args.gpus.split(',')
    if len(ids) != 2 or len(set(ids)) != 2 or any(not i.isdecimal() for i in ids):
        raise ValueError('exactly two distinct numeric GPU IDs are required')
    source = Path(args.source_project).resolve()
    original = PathResolver.from_file(args.paths, repo_root=source)
    paths = PathSet(original.paths.model_root, original.paths.data_root, run / 'ckpts', root,
                    run / 'results', run / 'logs')
    resolver = PathResolver(paths, original.mappings)
    run.mkdir(parents=True, exist_ok=False)
    stages = []
    for task in (1, 2):
        commands = build_sequence_commands(root / 'configs', 'llava', args.method, args.benchmark,
                                           paths=resolver, task_id=task, gpus=args.gpus,
                                           master_port=args.master_port, validation_fixes=True)
        train = next(item for item in commands if item.kind == 'train')
        src = Path(option(train.argv, '--data_path'))
        folder = Path(option(train.argv, '--image_folder'))
        records = _first_records(src, 4)
        _validate_records(records, folder)
        subset = run / f'task{task}-train.json'
        subset.write_text(json.dumps(records, ensure_ascii=False) + '\n', encoding='utf-8')
        for key, value in {
            '--data_path': subset, '--per_device_train_batch_size': 1, '--per_device_eval_batch_size': 1,
            '--gradient_accumulation_steps': 1, '--num_train_epochs': 1, '--max_steps': 2,
            '--learning_rate': '0.0002', '--dataloader_num_workers': 0, '--save_strategy': 'no',
            '--logging_nan_inf_filter': 'False',
        }.items():
            set_option(train.argv, key, value)
        checkpoint = option(train.argv, '--output_dir')
        if task == 2 and option(train.argv, '--previous_task_model_path') != stages[0]['checkpoint']:
            raise ValueError('task 2 does not load task 1 checkpoint')
        infer = {
            'argv': [sys.executable, '-m', 'mcitlib.backbones.llava.eval_runners.entry',
                     'mcitlib.backbones.llava.smoke_infer', '--benchmark', args.benchmark,
                     '--adapter', checkpoint, '--model-base', option(train.argv, '--model_name_or_path'),
                     '--subset', str(subset), '--image-folder', str(folder), '--output', str(run / f'task{task}-generation.json')],
            'cwd': str(root), 'env': {'MCITLIB_METHOD': args.method, 'CUDA_VISIBLE_DEVICES': ids[0]},
        }
        stages.append({'task': task, 'source': str(src), 'source_sha256': sha(src),
                       'subset': str(subset), 'subset_sha256': sha(subset), 'checkpoint': checkpoint,
                       'train': train.as_dict(), 'generate': infer})
    plan = {'version': 1, 'project': str(root), 'run_dir': str(run), 'method': args.method,
            'benchmark': args.benchmark, 'gpus': args.gpus, 'master_port': args.master_port,
            'samples_per_task': 4, 'optimizer_steps_per_task': 2, 'generation_per_task': 1,
            'stages': stages}
    out = run / 'plan.json'
    out.write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
    print(out)


def execute(args):
    root = repo_root()
    path = managed_path(args.plan, root, label='smoke plan')
    plan = json.loads(path.read_text(encoding='utf-8'))
    run = managed_path(plan['run_dir'], root, label='smoke run directory')
    if path != run / 'plan.json' or plan['project'] != str(root) or plan['version'] != 1:
        raise ValueError('plan does not belong to this project and run directory')
    for stage in plan['stages']:
        if sha(Path(stage['source'])) != stage['source_sha256'] or sha(Path(stage['subset'])) != stage['subset_sha256']:
            raise ValueError('smoke input changed after preparation')
        if Path(stage['checkpoint']).exists():
            raise FileExistsError('refusing to overwrite a checkpoint')
    for stage in plan['stages']:
        for name in ('train', 'generate'):
            command = stage[name]
            log = run / f'task{stage["task"]}-{name}.log'
            env = process_env(command['env'], root)
            env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
            if 'CUDA_VISIBLE_DEVICES' not in command['env']:
                env.pop('CUDA_VISIBLE_DEVICES', None)
            env.update(HF_HOME=str(run / 'cache/huggingface'), TRANSFORMERS_OFFLINE='1', HF_HUB_OFFLINE='1',
                       XDG_CACHE_HOME=str(run / 'cache/xdg'), TORCH_EXTENSIONS_DIR=str(run / 'cache/torch'),
                       TRITON_CACHE_DIR=str(run / 'cache/triton'))
            with log.open('x', encoding='utf-8') as handle:
                result = subprocess.run(command['argv'], cwd=command['cwd'],
                                        env=env, stdin=subprocess.DEVNULL,
                                        stdout=handle, stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                raise RuntimeError(f'{name} task {stage["task"]} failed; see {log}')
            if name == 'train':
                error = checkpoint_error(command['argv'])
                if error:
                    raise RuntimeError(error)
            else:
                out = run / f'task{stage["task"]}-generation.json'
                generated = json.loads(out.read_text(encoding='utf-8'))
                if not generated.get('loaded') or not 0 <= generated['new_tokens'] <= 16:
                    raise RuntimeError(f'invalid generation output: {out}')
    print(f'two-task smoke completed: {run}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    make = actions.add_parser('prepare')
    make.add_argument('--method', choices=('same', 'sigma'), required=True)
    make.add_argument('--benchmark', choices=('ucit', 'mllm_dcl'), required=True)
    make.add_argument('--paths', required=True)
    make.add_argument('--source-project', required=True)
    make.add_argument('--run-dir', required=True)
    make.add_argument('--gpus', default='0,1')
    make.add_argument('--master-port', type=int, default=29671)
    run = actions.add_parser('execute')
    run.add_argument('--plan', required=True)
    args = parser.parse_args()
    if not 1024 <= getattr(args, 'master_port', 1024) <= 65535:
        raise ValueError('invalid master port')
    (prepare if args.action == 'prepare' else execute)(args)


if __name__ == '__main__':
    main()
