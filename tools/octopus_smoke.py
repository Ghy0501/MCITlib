# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 MCITlib authors
"""Prepare or run a bounded two-task Octopus train/GPWC/reload/generation smoke."""
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


def _get(argv, flag):
    if argv.count(flag) != 1:
        raise ValueError(f'expected one {flag}')
    return argv[argv.index(flag) + 1]


def _set(argv, flag, value):
    if flag in argv:
        argv[argv.index(flag) + 1] = str(value)
    else:
        argv.extend([flag, str(value)])


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare(args):
    root = repo_root()
    run = managed_path(args.run_dir, root, label='Octopus smoke directory')
    ids = args.gpus.split(',')
    if run.exists() or len(ids) != 2 or len(set(ids)) != 2 or any(not item.isdecimal() for item in ids):
        raise ValueError('use a new run directory and two distinct numeric GPUs')
    source = PathResolver.from_file(args.paths, repo_root=Path(args.source_project).resolve())
    roots = PathSet(source.paths.model_root, source.paths.data_root, run / 'ckpts', root, run / 'results', run / 'logs')
    paths = PathResolver(roots, source.mappings)
    run.mkdir(parents=True, exist_ok=False)
    stages = []
    for task in (1, 2):
        commands = build_sequence_commands(root / 'configs', 'llava', 'octopus', args.benchmark, paths=paths,
                                           task_id=task, gpus=args.gpus, master_port=args.master_port,
                                           validation_fixes=True)
        selected = [c for c in commands if c.metadata.get('octopus_stage')]
        first = next(c for c in selected if c.metadata['octopus_stage'] == 'stage1')
        source_path = Path(_get(first.argv, '--data_path'))
        folder = Path(_get(first.argv, '--image_folder'))
        rows = _first_records(source_path, 4)
        _validate_records(rows, folder)
        subset = run / f'task{task}-train.json'
        subset.write_text(json.dumps(rows, ensure_ascii=False) + '\n', encoding='utf-8')
        for c in selected:
            stage = c.metadata['octopus_stage']
            if stage in {'stage1', 'sample', 'gpwc'}:
                _set(c.argv, '--data_path', subset)
            if stage == 'sample':
                _set(c.argv, '--count', 4)
            if stage == 'gpwc':
                _set(c.argv, '--samples', 4)
                _set(c.argv, '--logical-batch', 2)
                _set(c.argv, '--microbatch', 1)
            if stage in {'stage1', 'stage2'}:
                for flag, val in {'--per_device_train_batch_size': 1, '--per_device_eval_batch_size': 1,
                                  '--gradient_accumulation_steps': 1, '--max_steps': 2,
                                  '--save_strategy': 'no', '--dataloader_num_workers': 0,
                                  '--logging_nan_inf_filter': 'False'}.items():
                    _set(c.argv, flag, val)
        checkpoint = next(_get(c.argv, '--output_dir') for c in selected if c.metadata['octopus_stage'] == ('copy' if task == 1 else 'stage2'))
        inference = {'argv': [sys.executable, '-m', 'mcitlib.backbones.llava.eval_runners.entry',
                              'mcitlib.backbones.llava.smoke_infer', '--benchmark', args.benchmark,
                              '--adapter', checkpoint, '--model-base', _get(first.argv, '--model_name_or_path'),
                              '--subset', str(subset), '--image-folder', str(folder), '--output', str(run / f'task{task}-generation.json')],
                     'cwd': str(root), 'env': {'MCITLIB_METHOD': 'octopus', 'CUDA_VISIBLE_DEVICES': ids[0]}}
        before = [c.as_dict() for c in selected if c.metadata['octopus_stage'] != 'cleanup']
        after = [c.as_dict() for c in selected if c.metadata['octopus_stage'] == 'cleanup']
        stages.append({'task': task, 'source': str(source_path), 'source_sha256': _sha(source_path),
                       'subset': str(subset), 'subset_sha256': _sha(subset), 'checkpoint': checkpoint,
                       'before_generation': before, 'generation': inference, 'after_generation': after})
    plan = {'version': 1, 'project': str(root), 'run_dir': str(run), 'benchmark': args.benchmark,
            'gpus': args.gpus, 'master_port': args.master_port, 'stages': stages}
    output = run / 'plan.json'
    output.write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
    print(output)


def execute(args):
    root = repo_root()
    plan_path = managed_path(args.plan, root, label='Octopus plan')
    plan = json.loads(plan_path.read_text())
    run = managed_path(plan['run_dir'], root, label='Octopus run directory')
    if plan_path != run / 'plan.json' or plan['project'] != str(root) or plan['version'] != 1:
        raise ValueError('plan does not belong to this candidate')
    for stage in plan['stages']:
        if _sha(stage['source']) != stage['source_sha256'] or _sha(stage['subset']) != stage['subset_sha256']:
            raise ValueError('input changed since smoke preparation')
        if Path(stage['checkpoint']).exists():
            raise FileExistsError('refusing to overwrite a checkpoint')
    for stage in plan['stages']:
        sequence = [*stage['before_generation'], stage['generation'], *stage['after_generation']]
        for index, command in enumerate(sequence):
            label = command.get('metadata', {}).get('octopus_stage', 'generation')
            log = run / f'task{stage["task"]}-{index}-{label}.log'
            env = process_env(command.get('env', {}), root)
            env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
            if 'CUDA_VISIBLE_DEVICES' not in command.get('env', {}):
                env.pop('CUDA_VISIBLE_DEVICES', None)
            env.update(HF_HOME=str(run / 'cache/huggingface'), XDG_CACHE_HOME=str(run / 'cache/xdg'),
                       TORCH_EXTENSIONS_DIR=str(run / 'cache/torch'), TRITON_CACHE_DIR=str(run / 'cache/triton'),
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            with log.open('x') as handle:
                result = subprocess.run(command['argv'], cwd=command.get('cwd', str(root)), env=env,
                                        stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'Octopus {label} failed; see {log}')
            if command.get('kind') == 'train':
                problem = checkpoint_error(command['argv'])
                if problem:
                    raise RuntimeError(problem)
            if label == 'generation':
                result = json.loads((run / f'task{stage["task"]}-generation.json').read_text())
                if not result.get('loaded') or not 1 <= result['new_tokens'] <= 16:
                    raise RuntimeError('invalid Octopus bounded generation')
    print(f'Octopus two-task smoke completed: {run}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    prep = actions.add_parser('prepare')
    prep.add_argument('--benchmark', choices=('ucit', 'mllm_dcl'), required=True)
    prep.add_argument('--paths', required=True)
    prep.add_argument('--source-project', required=True)
    prep.add_argument('--run-dir', required=True)
    prep.add_argument('--gpus', default='0,1')
    prep.add_argument('--master-port', type=int, default=29685)
    execute_parser = actions.add_parser('execute')
    execute_parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    if not 1024 <= getattr(args, 'master_port', 1024) <= 65535:
        raise ValueError('invalid master port')
    (prepare if args.action == 'prepare' else execute)(args)


if __name__ == '__main__':
    main()
