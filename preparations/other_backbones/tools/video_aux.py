#!/usr/bin/env python3
"""Recover video replay QA from verified router identities; leave base sample untouched."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import video_sample as base

LABEL = {'counting': 'F', 'space': 'G', 'traffic': 'D', 'movie': 'A',
         'gui': 'C', 'science': 'E', 'sports': 'H', 'star': 'B'}


def key(row: dict[str, Any]) -> tuple[str, Any, str]:
    kind, value = base.typed_id(row.get('id'))
    video = row.get('video')
    if not isinstance(video, str) or not video:
        raise ValueError('row lacks video')
    return kind, value, video


def build(configs: Path, root: Path, sample_root: Path, destination: Path) -> Path:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'will not overwrite auxiliary dataset: {destination}')
    manifest_path = sample_root / 'manifest.json'
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if tuple(manifest['task_order']) != base.TASKS:
        raise ValueError('base sample task order differs from CL-VISTA protocol')
    entries = {item['task']: item for item in manifest['domains']}
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f'.{destination.name}.staging-{uuid.uuid4().hex}'
    stage.mkdir()
    history: dict[str, dict[tuple[str, Any, str], dict[str, Any]]] = {}
    results = []
    try:
        for index, task in enumerate(base.TASKS):
            cfg_path = configs / f'{task}.json'
            cfg = json.loads(cfg_path.read_bytes())
            train, train_src = base.read_array(base.source_path(root, cfg, 'train_path'))
            sample, sample_src = base.read_array(sample_root / task / 'train.json')
            entry = entries[task]
            if (sample_src['sha256'] != entry['outputs']['train.json']['sha256'] or
                    train_src['sha256'] != entry['source']['train']['sha256'] or
                    len(sample) != len(entry['selected']['train'])):
                raise ValueError(f'{task}: base/source training hash or count mismatch')
            train_index = base.unique_index(train, 'id', f'{task}/original_train')
            selected_keys = set()
            for row, info in zip(sample, entry['selected']['train']):
                original_index = info['source_index']
                if not 0 <= original_index < len(train) or row != train[original_index]:
                    raise ValueError(f'{task}: sampled train row differs from source')
                selected_keys.add(key(row))
            if len(selected_keys) != len(sample):
                raise ValueError(f'{task}: duplicate sampled train identity')
            router_path = base.source_path(root, cfg, 'router_path')
            router_bytes = router_path.read_bytes()
            router = json.loads(router_bytes)
            if not isinstance(router, list):
                raise ValueError(f'{task}: router data is not an array')
            groups: dict[str, dict[tuple[str, Any, str], dict[str, Any]]] = defaultdict(dict)
            for row in router:
                identity = key(row)
                domain = identity[2].split('/', 1)[0].lower()
                if domain not in base.TASKS[:index + 1] or identity in groups[domain]:
                    raise ValueError(f'{task}: unknown or duplicate router identity {identity}')
                label = next((turn.get('value') for turn in row.get('conversations', [])
                              if turn.get('from') == 'gpt'), None)
                if label != LABEL[domain]:
                    raise ValueError(f'{task}: router label does not match documented domain mapping')
                if base.identity(base.relative_media(root, identity[2])) is None:
                    raise ValueError(f'{task}: router video missing: {identity[2]}')
                groups[domain][identity] = row
            if set(groups) != set(base.TASKS[:index + 1]) or any(len(rows) != 100 for rows in groups.values()):
                raise ValueError(f'{task}: router must contain exactly 100 unique identities per seen domain')
            for prior in base.TASKS[:index]:
                if set(groups[prior]) != set(history[prior]):
                    raise ValueError(f'{task}: prior router identities differ from original {prior} set')
            current = {}
            for identity in groups[task]:
                typed = identity[:2]
                if typed not in train_index:
                    raise ValueError(f'{task}: router ID not found in original training data')
                original_index = train_index[typed]
                original = train[original_index]
                if key(original) != identity:
                    raise ValueError(f'{task}: router video does not match its original QA')
                current[identity] = {'row': original, 'source_index': original_index}
            history[task] = current
            legacy, legacy_src = base.read_array(base.source_path(root, cfg, 'replay_path'))
            replay = []
            replay_indices = []
            counts: Counter[str] = Counter()
            previous_seen: dict[str, set[tuple[str, Any, str]]] = defaultdict(set)
            selected_seen = set()
            current_seen = set()
            for source_index, row in enumerate(legacy):
                identity = key(row)
                domain = identity[2].split('/', 1)[0].lower()
                counts[domain] += 1
                if domain == task:
                    typed = identity[:2]
                    if typed not in train_index or train[train_index[typed]] != row or identity in current_seen:
                        raise ValueError(f'{task}: current-domain replay row differs from original QA')
                    current_seen.add(identity)
                    if identity in selected_keys:
                        if identity in selected_seen:
                            raise ValueError(f'{task}: duplicate current sample in legacy replay')
                        selected_seen.add(identity)
                        replay.append(row)
                        replay_indices.append({'legacy_replay_index': source_index, 'domain': domain,
                                               'original_train_index': train_index[typed], 'id': row['id'], 'video': row['video']})
                elif domain in history and domain != task:
                    reference = history[domain].get(identity)
                    if reference is None or reference['row'] != row or identity in previous_seen[domain]:
                        raise ValueError(f'{task}: prior replay does not equal recovered original QA')
                    previous_seen[domain].add(identity)
                    replay.append(reference['row'])
                    replay_indices.append({'legacy_replay_index': source_index, 'domain': domain,
                                           'original_train_index': reference['source_index'], 'id': row['id'], 'video': row['video']})
                else:
                    raise ValueError(f'{task}: legacy replay contains unexpected domain {domain}')
            expected_counts = {prior: 100 for prior in base.TASKS[:index]}
            expected_counts[task] = len(train)
            if (dict(counts) != expected_counts or current_seen != {key(row) for row in train} or
                    selected_seen != selected_keys or
                    any(previous_seen[prior] != set(history[prior]) for prior in base.TASKS[:index])):
                raise ValueError(f'{task}: legacy replay quota or selected QA coverage mismatch')
            router_out = stage / task / 'router.json'
            router_out.parent.mkdir(parents=True, exist_ok=True)
            with router_out.open('xb') as stream:
                stream.write(router_bytes)
            replay_info = base.write_json(stage / task / 'replay.json', replay)
            replay_info['path'] = str(destination / task / 'replay.json')
            results.append({'task': task, 'base_train_sha256': sample_src['sha256'],
                            'source_train': train_src, 'source_router': {'path': str(router_path),
                            'sha256': hashlib.sha256(router_bytes).hexdigest(), 'rows': len(router)},
                            'source_legacy_replay': legacy_src, 'legacy_domain_counts': dict(counts),
                            'router': {'path': str(destination / task / 'router.json'),
                                       'sha256': hashlib.sha256(router_bytes).hexdigest(), 'rows': len(router)},
                            'replay': {**replay_info, 'rows': len(replay), 'source_rows': replay_indices},
                            'router_domain_counts': {domain: len(group) for domain, group in groups.items()},
                            'status': 'source_qa_recovered_and_hash_verified'})
            print(json.dumps({'task': task, 'router': len(router), 'replay': len(replay)}), flush=True)
        output = {'schema_version': 1, 'base_manifest': str(manifest_path),
                  'base_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
                  'video_root': str(root), 'task_order': base.TASKS, 'router_label_to_domain': {v: k for k, v in LABEL.items()},
                  'domains': results, 'status': 'data_verified_model_integration_pending',
                  'note': 'Router records preserve original expert labels. Replay uses source QA matched by typed ID and video, '
                          'never router answers; earlier-domain rows can be outside the base main-train sample. '
                          'Legacy replay order is preserved after selecting current sample.'}
        base.write_json(stage / 'manifest.json', output)
        base.promote_noreplace(stage, destination)
    except Exception as error:
        raise RuntimeError(f'auxiliary build failed; unpromoted staging retained at {stage}: {error}') from error
    return destination / 'manifest.json'


def main() -> int:
    project = base.find_repo()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configs', type=Path, default=project / 'configs/data_configs/CL-VISTA')
    parser.add_argument('--video-root', type=Path, default=Path('/data/mcitlib/CL-VISTA'))
    parser.add_argument('--sample-root', type=Path, default=project / 'data/validation_3k_1k_video')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        output = args.output.resolve()
        for forbidden in (project / 'data/validation_3k_1k', project / 'data/validation_aux', args.sample_root.resolve()):
            if output == forbidden or forbidden in output.parents:
                raise ValueError(f'aux output must not modify existing sample: {output}')
        result = build(args.configs.resolve(), args.video_root.resolve(), args.sample_root.resolve(), output)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f'video auxiliary: {error}', file=sys.stderr)
        return 2
    print(json.dumps({'manifest': str(result)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
