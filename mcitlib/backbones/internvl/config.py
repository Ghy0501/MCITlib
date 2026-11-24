# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Build InternVL command values from the checked-in per-task configurations."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from ...registry import get_benchmark, get_method


def _portable(value):
    if isinstance(value, dict):
        return {key: _portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable(item) for item in value]
    if not isinstance(value, str):
        return value
    replacements = (
        ('/opt/mcitlib-example/MCITlib/checkpoints/', '${ckpt_root}/MCITlib/'),
        ('/your_ckpts_path', '${ckpt_root}'),
        ('/your_model_path', '${model_root}'),
        ('/your_data_path', '${data_root}'),
        ('/LLaVA/', '/InternVL/'),
        ('_llava_lora', '_internvl_lora'),
        ('_llava', '_internvl'),
        ('${model_root}/llava-v1.5-7b', '${model_root}/InternVL-Chat-ViT-6B-Vicuna-7B'),
    )
    for old, new in replacements:
        value = value.replace(old, new)
    return value


def _json(path):
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f'expected a JSON object: {path}')
    return _portable(value), hashlib.sha256(raw).hexdigest()


def _set(argv, flag, value, *, required=True):
    if argv.count(flag) != 1 or argv.index(flag) + 1 >= len(argv):
        if not required and flag not in argv:
            return
        raise ValueError(f'expected one {flag} in InternVL command')
    argv[argv.index(flag) + 1] = str(value)


def _remove(argv, flag):
    if flag in argv:
        if argv.count(flag) != 1 or argv.index(flag) + 1 >= len(argv):
            raise ValueError(f'expected at most one {flag} in InternVL command')
        index = argv.index(flag)
        del argv[index:index + 2]


def _refresh_records(values):
    """Keep inspection fields derived from the same argv that will execute."""
    from ...config.compact import _parse_options

    for field in ('train_command', 'eval_commands', 'extra_commands', 'other_commands'):
        records = [values[field]] if field == 'train_command' and field in values else values.get(field, [])
        for record in records:
            argv = record['argv']
            if argv[0] == 'deepspeed':
                entry = next((index for index, token in enumerate(argv) if token.endswith('.py')), None)
                if entry is None:
                    raise ValueError('missing training entry in InternVL command')
                record['launcher'] = _parse_options(argv[1:entry])[0]
                record['entry'] = argv[entry]
                record['args'], record['positional'] = _parse_options(argv[entry + 1:])
            else:
                record['args'], record['positional'] = _parse_options(argv[1:])


def _configure_task(values, model, train, evaluate, *, independent=False, merged_base=False):
    values['model_config'], values['train_config'], values['eval_config'] = model, train, evaluate
    argv = values['train_command']['argv']
    for flag, value in (
        ('--model_name_or_path', train.get('previous_model', model['model_name']) if merged_base else model['model_name']),
        ('--vision_tower', model['vision_tower']),
        ('--mm_vision_select_layer', -4), ('--lora_r', train['rank']),
        ('--lora_alpha', int(train['rank']) * 2), ('--output_dir', train['output_dir']),
        ('--num_train_epochs', train['epoch']), ('--per_device_train_batch_size', train['batch_size']),
        ('--gradient_accumulation_steps', train['grad_acc']), ('--learning_rate', train['lr']),
    ):
        _set(argv, flag, value)
    _set(argv, '--pretrain_mm_mlp_adapter', model['mm_projector'], required=False)
    _set(argv, '--text_tower', model['clip_tower'], required=False)
    if 'previous_model' in train and not independent and not merged_base:
        _set(argv, '--previous_task_model_path', train['previous_model'])
    for key, flag in (('expert_num', '--expert_num'), ('cur_task', '--cur_task'), ('task', '--task'),
                      ('ins_type', '--ins_type')):
        if key in train:
            _set(argv, flag, train[key])
    if 'gradient_checkpointing' in train:
        _set(argv, '--gradient_checkpointing', train['gradient_checkpointing'])
    # These InternVL scripts do not set a distinct projector learning rate.
    _remove(argv, '--mm_projector_lr')
    for command in values.get('eval_commands', []):
        _set(command['argv'], '--model-base', model['model_name'], required=False)
        _set(command['argv'], '--model-path', evaluate['model_path'], required=False)
        _set(command['argv'], '--text-tower', model['clip_tower'], required=False)
        if 'num_task' in evaluate:
            _set(command['argv'], '--num-task', evaluate['num_task'], required=False)


def _configure_modalprompt(values, model, train, evaluate, task_id, benchmark):
    values['model_config'], values['train_config'], values['eval_config'] = model, train, evaluate
    if train['output_dir'] != evaluate['model_path']:
        if benchmark != 'ucit' or task_id != 2 or not train['output_dir'].endswith('/Task1_internvl'):
            raise ValueError('unexpected ModalPrompt train/eval checkpoint mismatch')
        train['output_dir'] = evaluate['model_path']
        values['internvl_repairs'] = ['modalprompt_ucit_task2_output_matches_eval_and_task3_input']
    argv = values['train_command']['argv']
    for flag, value in (
        ('--model_name_or_path', model['model_name']), ('--vision_tower', model['vision_tower']),
        ('--text_tower', model['clip_tower']), ('--mm_vision_select_layer', -4),
        ('--output_dir', train['output_dir']), ('--prefix_len', train['prefix_len']),
        ('--cur_task', train['cur_task']), ('--num_tasks', train['num_tasks']),
        ('--num_train_epochs', train['epoch']), ('--per_device_train_batch_size', train['batch_size']),
        ('--gradient_accumulation_steps', train['grad_acc']), ('--learning_rate', train['lr']),
    ):
        _set(argv, flag, value)
    _remove(argv, '--mm_projector_lr')
    # Some LLaVA task templates repeated this flag; InternVL task 1 has one,
    # later tasks load the previous prompt state instead of the base projector.
    while '--pretrain_mm_mlp_adapter' in argv:
        index = argv.index('--pretrain_mm_mlp_adapter')
        del argv[index:index + 2]
    if task_id == 1:
        argv.extend(['--pretrain_mm_mlp_adapter', model['mm_projector']])
    else:
        argv.extend(['--previous_task_model_path', train['previous_model']])
    for command in values.get('eval_commands', []):
        for flag, value in (('--model-base', model['model_name']), ('--model-path', evaluate['model_path']),
                            ('--text-tower', model['clip_tower']), ('--prefix-len', evaluate['prefix_len']),
                            ('--cur-task', evaluate['cur_task']), ('--num-tasks', evaluate['num_tasks'])):
            _set(command['argv'], flag, value, required=False)


def load_internvl_config(root, benchmark, method, task_id=None, cli_overrides=None):
    """Expand one method/task, preserving the source's full-data training budget."""
    from ...config.loader import deep_merge, load_layered_config

    root = Path(root)
    spec = get_method(method)
    bench = get_benchmark(benchmark)
    if method not in {'lora_ft', 'replay', 'moelora', 'olora', 'clmoe', 'mrlora', 'sefe', 'keeplora',
                      'disco', 'hide', 'smolora', 'modalprompt'} or bench.id not in {'ucit', 'mllm_dcl'}:
        raise ValueError(f'InternVL {method}/{benchmark} is not yet enabled')
    # Reuse the mature LLaVA command ordering and score/merge entry points.
    values = _portable(copy.deepcopy(load_layered_config(root, 'llava', bench.id, spec.id, task_id=task_id)))
    values = deep_merge(values, {'backbone': 'internvl'})
    values.setdefault('overrides', {})['internvl'] = copy.deepcopy(values['overrides']['llava'])
    model, model_sha = _json(root / 'model_configs/internvl.json')
    values['model_config'] = model
    if task_id is None:
        return deep_merge(values, cli_overrides or {})
    folder = root / 'train_configs' / spec.v3_dir / 'InternVL' / bench.v3_name
    train_path, eval_path = folder / 'train' / f'task{task_id}.json', folder / 'eval' / f'task{task_id}.json'
    train, train_sha = _json(train_path)
    evaluate, eval_sha = _json(eval_path)
    if method == 'keeplora' and not evaluate['model_path'].endswith('_merged'):
        # The DCL source JSON names an adapter while its evaluator loads a full
        # model without model-base; use the merge produced immediately before it.
        evaluate['model_path'] += '_merged'
    if method == 'modalprompt':
        _configure_modalprompt(values, model, train, evaluate, task_id, bench.id)
    else:
        _configure_task(values, model, train, evaluate, independent=method == 'mrlora',
                        merged_base=method in {'sefe', 'keeplora'})
    values['source'].update(model_config='model_configs/internvl.json',
                            train_config=str(train_path.relative_to(root)),
                            eval_config=str(eval_path.relative_to(root)))
    values['internvl_source_sha256'] = {'model_config': model_sha, 'train_config': train_sha,
                                        'eval_config': eval_sha}
    if method == 'sefe':
        for command in values.get('extra_commands', []):
            argv = command['argv']
            if 'key_elements_identification.py' in argv:
                if argv[-1] != 'llava':
                    raise ValueError('unexpected SEFE key identification base model')
                argv[-1] = 'internvl'
        if bench.id == 'mllm_dcl':
            # The source config mislabeled its DCL output tree as UCIT.
            evaluate['result_path'] = evaluate['result_path'].replace('./results/UCIT/', './results/DCL/')
            for command in values.get('eval_commands', []):
                command['argv'] = [token.replace('./results/UCIT/', './results/DCL/') for token in command['argv']]
    if method == 'keeplora':
        for command in values.get('extra_commands', []):
            _set(command['argv'], '--model-config', '${repo_root}/configs/model_configs/internvl.json', required=False)
        for split in ('train_pre', 'train_post'):
            source = folder / split / f'task{task_id}.json'
            _, sha = _json(source)
            values['internvl_source_sha256'][split] = sha
    if method == 'mrlora':
        router = values['router']['benchmarks'][bench.id]['tasks'][f'task_{task_id}']
        router_train_path = folder / 'train_router' / f'task{task_id}.json'
        router_eval_path = folder / 'eval_router' / f'task{task_id}.json'
        router_train, router_train_sha = _json(router_train_path)
        router_eval, router_eval_sha = _json(router_eval_path)
        # Each expert and router is independently initialized from the base model;
        # the reference always runs Task1.sh / Task1_router.sh for every task.
        _configure_task(router, model, router_train, router_eval, independent=True)
        router['source'].update(model_config='model_configs/internvl.json',
                                train_config=str(router_train_path.relative_to(root)),
                                eval_config=str(router_eval_path.relative_to(root)))
        router['internvl_source_sha256'] = {'model_config': model_sha, 'train_config': router_train_sha,
                                            'eval_config': router_eval_sha}
        _refresh_records(router)
    _refresh_records(values)
    return deep_merge(values, cli_overrides or {})
