#!/usr/bin/env python3
"""Materialize reviewed offline model/data/train/eval JSON; never run models."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any

from video_sample import promote_noreplace

TOKEN = re.compile(r'\$\{([A-Z0-9_]+)\}')


def expand(value: Any, variables: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: expand(item, variables) for key, item in value.items()}
    if isinstance(value, list):
        return [expand(item, variables) for item in value]
    if not isinstance(value, str):
        return value
    for _ in range(8):
        match = TOKEN.search(value)
        if match is None:
            if re.search(r'/your_[A-Za-z0-9_]+', value):
                raise ValueError(f'unresolved legacy path placeholder: {value}')
            return value
        key = match.group(1)
        if key not in variables:
            raise ValueError(f'no value for placeholder {key}')
        value = TOKEN.sub(lambda item: variables[item.group(1)], value)
    raise ValueError(f'recursive or unresolved variable: {value}')


def absolute_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f'{label} must be an absolute path: {value}')
    return Path(value).resolve(strict=False)


def require_in_root(value: Any, root: str, label: str) -> Path:
    path = absolute_path(value, label)
    if not path.is_relative_to(absolute_path(root, label + ' root')):
        raise ValueError(f'{label} escapes its variant root: {value}')
    return path


def audit_variant(variant: dict[str, Any], model: dict[str, Any], scope: dict[str, str],
                  mode: str, gpus: int) -> list[dict[str, Any]]:
    """Reject unmapped writes/references and record known absent merge producers."""
    stages = variant['stages']
    train = [expand(s[f'sample_train_{gpus}gpu'] if mode == 'sample' else s['full_train'], scope) for s in stages]
    evaluate = [expand(s[f'sample_eval_{gpus}gpu'] if mode == 'sample' else s['full_eval']['config'], scope) for s in stages]
    auxiliary = [(a['role'], Path(a['path']).name, expand(a['config'], scope))
                 for a in variant['auxiliary_configs_unmodified']]
    model_path = absolute_path(expand(model['legacy_model_config']['config']['model_name'], scope), 'base model')
    for key, raw in {**model['legacy_model_config']['config'], **model['loader_config_overrides']}.items():
        if key in ('model_name', 'vision_tower', 'mm_projector', 'text_tower', 'clip_tower',
                   'mm_vision_tower', 'mm_image_tower', 'mm_video_tower'):
            path = absolute_path(expand(raw, scope), f'model {key}')
            if not path.exists():
                raise ValueError(f'missing static model input {key}: {path}')
    main_outputs = [require_in_root(c.get('output_dir'), scope['CKPT_ROOT'], 'train output_dir') for c in train]
    if len(set(main_outputs)) != len(main_outputs):
        raise ValueError('main training output_dir is duplicated across stages')
    aux_outputs = {require_in_root(c['output_dir'], scope['CKPT_ROOT'], f'{role} output_dir')
                   for role, _, c in auxiliary if 'output_dir' in c}
    all_outputs = set(main_outputs) | aux_outputs
    missing: list[dict[str, Any]] = []

    def reference(raw: Any, field: str, consumer: str, allowed: set[Path] | None = None) -> Path:
        path = absolute_path(raw, f'{consumer} {field}')
        if field == 'model_path' and path == model_path:
            return path
        require_in_root(str(path), scope['CKPT_ROOT'], f'{consumer} {field}')
        candidates = allowed if allowed is not None else all_outputs
        if path in candidates:
            return path
        if str(path).endswith('_merged'):
            unmerged = Path(str(path)[:-len('_merged')])
            if unmerged in candidates and unmerged in main_outputs:
                producer_stage = main_outputs.index(unmerged) + 1
                kind = 'first' if producer_stage == 1 else 'later'
                evidence = variant.get('merged_output_evidence', {}).get(kind, {})
                if evidence.get('status') != 'declared_in_legacy_script':
                    missing.append({'consumer': consumer, 'field': field, 'expected_path': str(path),
                                    'source_output': str(unmerged), 'producer_stage': producer_stage,
                                    'reason': 'missing_merge_producer', 'evidence': evidence})
                return path
        raise ValueError(f'{consumer} {field} has no planned producer: {path}')

    for index, (stage_info, cfg, ev) in enumerate(zip(stages, train, evaluate), 1):
        dataset = variant['datasets'][index - 1]
        data = (dataset['sample'] if mode == 'sample' else
                dataset['full']['config'] if variant['benchmark'] == 'CL-VISTA' else dataset['full']['resolved_config'])
        data = expand(data, scope)
        required = ('train_path', 'test_q_path', 'test_a_path', 'train_folder', 'test_folder') if (
            variant['benchmark'] == 'CL-VISTA') else ('train_path', 'test_path', 'train_folder', 'test_folder')
        for key in required:
            path = absolute_path(data.get(key), f'task{index} {key}')
            if not path.exists():
                raise ValueError(f'missing static dataset input {key}: {path}')
        for key in ('replay_path', 'router_path', 'router_folder', 'anno_path'):
            if data.get(key) is not None:
                path = absolute_path(data[key], f'task{index} {key}')
                if not path.exists():
                    raise ValueError(f'missing static dataset input {key}: {path}')
        if 'previous_model' in cfg:
            reference(cfg['previous_model'], 'previous_model', f'train task{index}', set(main_outputs[:index - 1]))
        if 'key_path' in cfg:
            key_path = require_in_root(cfg['key_path'], scope['CKPT_ROOT'], f'train task{index} key_path')
            if not main_outputs[index - 1].is_relative_to(key_path):
                raise ValueError(f'train task{index} key_path does not contain its outputs')
        for asset in stage_info.get('external_inputs', []):
            path = absolute_path(cfg.get(asset['field']), f'train task{index} {asset["field"]}')
            if not any(path.is_relative_to(absolute_path(root, 'asset root')) for root in
                       (str(Path(scope['PROJECT_ROOT']) / 'assets/smolora'),
                        str(Path(scope['PREPARATIONS_ROOT']) / 'assets'))):
                raise ValueError(f'input asset outside approved roots: {path}')
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != asset['sha256']:
                raise ValueError(f'missing or changed input asset: {path}')
        reference(ev.get('model_path'), 'model_path', f'eval task{index}', {main_outputs[index - 1]})
        require_in_root(ev.get('result_path'), scope['RESULTS_ROOT'], f'eval task{index} result_path')
        for key in ('judge_path', 'text_tower'):
            if key in ev and not absolute_path(ev[key], f'eval task{index} {key}').exists():
                raise ValueError(f'missing static evaluation input {key}: {ev[key]}')
    for role, filename, cfg in auxiliary:
        consumer = f'{role}/{filename}'
        for field in ('model_path', 'previous_model', 'space_path'):
            if field in cfg:
                reference(cfg[field], field, consumer)
        if 'result_path' in cfg:
            require_in_root(cfg['result_path'], scope['RESULTS_ROOT'], f'{consumer} result_path')
    return missing


def materialize(plan_path: Path, destination: Path, run_id: str, mode: str,
                gpus: int, overrides: dict[str, str]) -> dict[str, Any]:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', run_id):
        raise ValueError('run ID must use letters, numbers, dot, underscore or dash')
    if mode not in ('sample', 'full') or gpus not in (2, 4):
        raise ValueError('unsupported profile')
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'will not overwrite existing configuration output: {destination}')
    raw = plan_path.read_bytes()
    plan = json.loads(raw)
    variables = dict(plan['variables'])
    unknown = set(overrides) - set(variables)
    if unknown:
        raise ValueError(f'unknown variable override(s): {sorted(unknown)}')
    variables.update(overrides)
    variables['RUN_ID'] = run_id
    stage = destination.parent / f'.{destination.name}.staging-{uuid.uuid4().hex}'
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage.mkdir()
    files = []
    namespace_roots = []
    readiness = []

    def write(relative: Path, obj: Any, scope: dict[str, str] | None = None) -> None:
        content = (json.dumps(expand(obj, scope or variables), ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
        path = stage / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(content)
        files.append({'path': str(relative), 'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)})

    try:
        for model, info in plan['models'].items():
            write(Path(model) / 'model.json', info['legacy_model_config']['config'])
            write(Path(model) / 'loader-overrides.json', {'overrides': info['loader_config_overrides'],
                                                         'status': 'pending_loader_integration'})
        for variant in plan['variants']:
            base = Path(variant['model']) / variant['method'] / variant['benchmark']
            scoped = dict(variables)
            for key in ('CKPT_ROOT', 'RESULTS_ROOT', 'LOG_ROOT'):
                scoped[key] = str(Path(expand(variables[key], variables)) / base)
            namespace_roots.append({'model': variant['model'], 'method': variant['method'],
                                    'benchmark': variant['benchmark'],
                                    **{key: scoped[key] for key in ('CKPT_ROOT', 'RESULTS_ROOT', 'LOG_ROOT')}})
            missing_producers = audit_variant(variant, plan['models'][variant['model']], scoped, mode, gpus)
            readiness.append({'model': variant['model'], 'method': variant['method'],
                              'benchmark': variant['benchmark'], 'execution': variant['execution_readiness'],
                              'execution_allowed': False, 'missing_producers': missing_producers,
                              'sample_auxiliary': variant['sample_auxiliary'],
                              'full_auxiliary': variant['full_auxiliary'],
                              'full_auxiliary_files': {x['task']: x['full_auxiliary'] for x in variant['datasets']},
                              'sample_datasets': {x['task']: x['sample_status'] for x in variant['datasets']},
                              'router_protocol': variant.get('router_protocol')})
            if len(variant['stages']) != len(variant['datasets']):
                raise ValueError(f'task count mismatch in {base}')
            for stage_info, dataset in zip(variant['stages'], variant['datasets']):
                if stage_info['task'] != dataset['task']:
                    raise ValueError(f'task order mismatch in {base}')
                filename = f"task{stage_info['stage']}.json"
                if mode == 'sample':
                    train = stage_info[f'sample_train_{gpus}gpu']
                    evaluate = stage_info[f'sample_eval_{gpus}gpu']
                    data = dataset['sample']
                else:
                    train = stage_info['full_train']
                    evaluate = stage_info['full_eval']['config']
                    data = (dataset['full']['config'] if variant['benchmark'] == 'CL-VISTA'
                            else dataset['full']['resolved_config'])
                write(base / 'train' / filename, train, scoped)
                write(base / 'eval' / filename, evaluate, scoped)
                write(base / 'data' / filename, data, scoped)
            for aux in variant['auxiliary_configs_unmodified']:
                name = Path(aux['path']).name
                write(base / 'aux' / aux['role'] / name, aux['config'], scoped)
        for key in ('CKPT_ROOT', 'RESULTS_ROOT', 'LOG_ROOT'):
            if len({item[key] for item in namespace_roots}) != len(namespace_roots):
                raise ValueError(f'cross-variant {key} collision')
        manifest = {'schema_version': 1, 'run_id': run_id, 'mode': mode, 'nominal_gpus': gpus,
                    'plan': str(plan_path), 'plan_sha256': hashlib.sha256(raw).hexdigest(),
                    'variables': variables, 'namespace_roots': namespace_roots, 'variants': readiness,
                    'files': files, 'file_count': len(files),
                    'readiness': 'blocked_loader_and_entry_not_integrated',
                    'note': 'Full mode preserves legacy per-task parameters; sample GPU values are untested plans. '
                            'Auxiliary router configs retain their legacy parameters. No model execution performed.'}
        write(Path('manifest.json'), manifest)
        promote_noreplace(stage, destination)
    except Exception as error:
        raise RuntimeError(f'configuration materialization failed; staging retained at {stage}: {error}') from error
    return {'output': str(destination), 'run_id': run_id, 'mode': mode, 'files': len(files),
            'readiness': 'blocked_loader_and_entry_not_integrated'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--mode', choices=('sample', 'full'), default='sample')
    parser.add_argument('--gpus', type=int, choices=(2, 4), default=4)
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    args = parser.parse_args()
    try:
        overrides = {}
        for item in args.set:
            key, separator, value = item.partition('=')
            if not separator or not value or key in overrides:
                raise ValueError(f'invalid or duplicate variable override: {item}')
            overrides[key] = value
        result = materialize(args.plan.resolve(), args.output.resolve(), args.run_id, args.mode, args.gpus, overrides)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f'materialize: {error}', file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
