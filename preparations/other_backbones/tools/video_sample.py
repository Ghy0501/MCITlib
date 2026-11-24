#!/usr/bin/env python3
"""Build one reproducible CL-VISTA JSON sample for both video backbones."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import random
import sys
import uuid
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

TASKS = ('counting', 'space', 'traffic', 'movie', 'gui', 'science', 'sports', 'star')
VIDEO_LLAVA = ('.mp4', '.avi', '.mov', '.mkv')
VIDEO_LLAMA2 = ('.mp4', '.webm', '.avi', '.mov', '.mkv')

def find_repo() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / 'mcitlib').is_dir() and (parent / 'configs/data_configs/CL-VISTA').is_dir():
            return parent
    raise ValueError('cannot locate project root; pass --configs from within the project')


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_array(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = path.read_bytes()
    rows = json.loads(data)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f'expected JSON array of objects: {path}')
    return rows, {'path': str(path), 'sha256': digest(data), 'rows': len(rows), 'bytes': len(data)}


def typed_id(value: Any) -> tuple[str, Any]:
    if type(value) not in (int, str) or (isinstance(value, str) and not value):
        raise ValueError(f'ID must be nonempty string or integer, got {value!r}')
    return type(value).__name__, value


def unique_index(rows: list[dict[str, Any]], field: str, label: str) -> dict[tuple[str, Any], int]:
    index = {}
    for i, row in enumerate(rows):
        key = typed_id(row.get(field))
        if key in index:
            raise ValueError(f'{label} duplicate {field} {key!r}: rows {index[key]} and {i}')
        index[key] = i
    return index


def relative_media(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f'invalid media path: {value!r}')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'media must be a relative path within dataset root: {value!r}')
    return root.joinpath(*path.parts)


def identity(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
        return (stat.st_dev, stat.st_ino) if path.is_file() else None
    except OSError:
        return None


def test_media(root: Path, video_name: Any) -> dict[str, Any]:
    """Reproduce each legacy loader's extension order and require the same file."""
    base = relative_media(root, video_name)
    def first(extensions: tuple[str, ...]) -> Path | None:
        return next((Path(str(base) + ext) for ext in extensions if Path(str(base) + ext).is_file()), None)
    first_vl, first_v2 = first(VIDEO_LLAVA), first(VIDEO_LLAMA2)
    same = first_vl is not None and first_v2 is not None and identity(first_vl) == identity(first_v2)
    reason = None if same else ('different_first_match' if first_vl and first_v2 else 'missing_common_video')
    return {'video_llava': str(first_vl) if first_vl else None,
            'video_llama2': str(first_v2) if first_v2 else None, 'same_file': same, 'reason': reason}


def sample_indices(eligible: list[int], target: int, seed: int, task: str, split: str) -> list[int]:
    material = f'{seed}\0{task}\0{split}'.encode()
    rng = random.Random(int.from_bytes(hashlib.sha256(material).digest(), 'big'))
    return sorted(rng.sample(eligible, min(target, len(eligible))))


def repeated_pairs(rows: list[dict[str, Any]], video_key: str, question_getter: Any,
                   *, normalized: bool = False) -> dict[str, Any]:
    pairs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        video = row.get(video_key)
        question = question_getter(row)
        if not isinstance(video, str) or not isinstance(question, str):
            raise ValueError(f'invalid video/question at row {index}')
        compared = ' '.join(question.split()).casefold() if normalized else question
        pairs[(video, digest(compared.encode()))].append({'index': index, 'id': row.get('id', row.get('question_id'))})
    groups = [{'video': video, 'question_sha256': question, 'rows': items}
              for (video, question), items in pairs.items() if len(items) > 1]
    return {'groups': len(groups), 'extra_rows': sum(len(x['rows']) - 1 for x in groups), 'details': groups,
            'normalization': 'whitespace collapsed and casefolded' if normalized else 'exact original question'}


def human_question(row: dict[str, Any]) -> str:
    turns = row.get('conversations')
    if not isinstance(turns, list):
        raise ValueError('training row lacks conversations')
    messages = [item.get('value') for item in turns if isinstance(item, dict) and item.get('from') == 'human']
    if not messages or not isinstance(messages[0], str):
        raise ValueError('training row lacks human question')
    return messages[0]


def source_path(root: Path, cfg: dict[str, Any], key: str) -> Path:
    value = cfg[key]
    prefix = '/your_data_path/CL-VISTA/'
    if not isinstance(value, str) or not value.startswith(prefix):
        raise ValueError(f'unexpected {key}: {value!r}')
    return relative_media(root, value[len(prefix):])


def prepare_domain(task: str, cfg: dict[str, Any], root: Path, seed: int,
                   train_target: int, test_target: int) -> tuple[dict[str, Any], dict[str, Any]]:
    train, train_src = read_array(source_path(root, cfg, 'train_path'))
    questions, question_src = read_array(source_path(root, cfg, 'test_q_path'))
    answers, answer_src = read_array(source_path(root, cfg, 'test_a_path'))
    unique_index(train, 'id', f'{task}/train')
    q_index = unique_index(questions, 'question_id', f'{task}/questions')
    a_index = unique_index(answers, 'question_id', f'{task}/answers')
    if set(q_index) != set(a_index):
        raise ValueError(f'{task}: question/answer IDs differ: '
                         f'{len(set(q_index) - set(a_index))} missing answers, '
                         f'{len(set(a_index) - set(q_index))} unmatched answers')
    train_ok, train_bad, train_ids = [], [], {}
    for i, row in enumerate(train):
        p = relative_media(root, row.get('video'))
        ident = identity(p)
        if ident is None:
            train_bad.append({'index': i, 'id': row['id'], 'video': row['video'], 'reason': 'missing_video'})
        else:
            train_ok.append(i)
            train_ids[i] = ident
    test_ok, test_bad, test_ids, test_info = [], [], {}, {}
    for i, row in enumerate(questions):
        if not isinstance(row.get('question'), str):
            raise ValueError(f'{task}: missing question text at row {i}')
        match = test_media(root, row.get('video_name'))
        answer = answers[a_index[typed_id(row['question_id'])]]
        if 'video_name' in answer and answer['video_name'] != row['video_name']:
            raise ValueError(f'{task}: question/answer video identity differs at row {i}')
        if not match['same_file']:
            test_bad.append({'index': i, 'id': row['question_id'], 'video_name': row['video_name'], **match})
        else:
            test_ok.append(i)
            test_info[i] = match
            test_ids[i] = identity(Path(match['video_llava']))
    chosen_train = sample_indices(train_ok, train_target, seed, task, 'train')
    chosen_test = sample_indices(test_ok, test_target, seed, task, 'test')
    chosen_answers = [a_index[typed_id(questions[i]['question_id'])] for i in chosen_test]
    # Consumers chunk both files independently and pair by position, so write matched answers in Q order.
    output = {'train.json': [train[i] for i in chosen_train],
              'test_questions.json': [questions[i] for i in chosen_test],
              'test_answers.json': [answers[i] for i in chosen_answers]}
    train_video_set, test_video_set = set(train_ids.values()), set(test_ids.values())
    source_overlap = train_video_set & test_video_set
    selected_overlap = {train_ids[i] for i in chosen_train} & {test_ids[i] for i in chosen_test}
    overlap_details = []
    for shared in sorted(source_overlap):
        train_rows = [{'index': i, 'id': train[i]['id'], 'video': train[i]['video']}
                      for i, value in train_ids.items() if value == shared]
        test_rows = [{'index': i, 'id': questions[i]['question_id'], 'video_name': questions[i]['video_name']}
                     for i, value in test_ids.items() if value == shared]
        overlap_details.append({'train_rows': train_rows, 'test_rows': test_rows})
    manifest = {
        'task': task, 'source': {'train': train_src, 'test_questions': question_src, 'test_answers': answer_src},
        'requested': {'train': train_target, 'test': test_target},
        'eligible': {'train': len(train_ok), 'test': len(test_ok)},
        'selected': {'train': [{'source_index': i, 'id': train[i]['id'], 'video': train[i]['video'],
                                'resolved_video': str(relative_media(root, train[i]['video']).resolve())}
                               for i in chosen_train],
                     'test': [{'question_index': i, 'answer_index': ai, 'id': questions[i]['question_id'],
                               'video_name': questions[i]['video_name'], 'media': test_info[i]}
                              for i, ai in zip(chosen_test, chosen_answers)]},
        'excluded_missing_or_conflicting_media': {'train': train_bad, 'test': test_bad},
        'duplicate_pairs': {'train_exact': repeated_pairs(train, 'video', human_question),
                            'train_normalized': repeated_pairs(train, 'video', human_question, normalized=True),
                            'test_exact': repeated_pairs(questions, 'video_name', lambda row: row.get('question')),
                            'test_normalized': repeated_pairs(questions, 'video_name', lambda row: row.get('question'), normalized=True)},
        'video_overlap': {'source_available_unique_files': len(source_overlap), 'sample_unique_files': len(selected_overlap),
                          'source_train_rows': sum(value in source_overlap for value in train_ids.values()),
                          'source_test_rows': sum(value in source_overlap for value in test_ids.values()),
                          'source_overlap_details': overlap_details,
                          'sample_train_indices': [i for i in chosen_train if train_ids[i] in selected_overlap],
                          'sample_test_indices': [i for i in chosen_test if test_ids[i] in selected_overlap]},
        'answer_video_identity': 'checked when present; raw answer rows normally omit video identity',
        'status': 'sample_ready' if len(chosen_train) == train_target and len(chosen_test) == test_target else 'sample_shortfall',
        'auxiliary': {'replay': 'blocked_pending_sample_provenance', 'router': 'blocked_pending_router_policy'},
    }
    return output, manifest


def write_json(path: Path, value: Any) -> dict[str, Any]:
    content = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    return {'path': str(path), 'sha256': digest(content), 'bytes': len(content)}


def promote_noreplace(source: Path, destination: Path) -> None:
    library = ctypes.CDLL('libc.so.6', use_errno=True)
    native = getattr(library, 'renameat2', None)
    if native is not None:
        native.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        native.restype = ctypes.c_int
        status = native(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    else:
        uname = os.uname()
        number = {'x86_64': 316, 'aarch64': 276}.get(uname.machine)
        if uname.sysname != 'Linux' or number is None or ctypes.sizeof(ctypes.c_long) != 8:
            raise OSError('unsupported no-clobber rename platform')
        call = library.syscall
        call.argtypes = (ctypes.c_long, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        call.restype = ctypes.c_long
        status = call(number, -100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if status:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number), str(destination))


def build(configs: Path, root: Path, destination: Path, seed: int, train_target: int, test_target: int) -> Path:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'will not overwrite existing dataset: {destination}')
    if train_target <= 0 or test_target <= 0:
        raise ValueError('sample targets must be positive')
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f'.{destination.name}.staging-{uuid.uuid4().hex}'
    stage.mkdir()
    domains = []
    try:
        for task in TASKS:
            config_path = configs / f'{task}.json'
            config_bytes = config_path.read_bytes()
            cfg = json.loads(config_bytes)
            output, entry = prepare_domain(task, cfg, root, seed, train_target, test_target)
            entry['source_config'] = {'path': str(config_path), 'sha256': digest(config_bytes)}
            entry['outputs'] = {}
            for name, rows in output.items():
                metadata = write_json(stage / task / name, rows)
                metadata['path'] = str(destination / task / name)
                metadata['rows'] = len(rows)
                entry['outputs'][name] = metadata
            domains.append(entry)
            print(json.dumps({'task': task, 'eligible': entry['eligible'], 'status': entry['status']}), flush=True)
        manifest = {'schema_version': 1, 'seed': seed, 'task_order': TASKS, 'root': str(root),
                    'selection': 'sha256-derived per-task/split random sample; selected source indices sorted',
                    'media_policy': 'exclude missing or different first-match files across both legacy loaders; do not rewrite media fields',
                    'dataset_status': 'shortfall' if any(x['status'] != 'sample_ready' for x in domains) else 'sample_ready',
                    'domains': domains, 'note': 'Sample QA only; no media copied; no overlap filtering; '
                    'replay/router provenance and model loaders remain pending.'}
        write_json(stage / 'manifest.json', manifest)
        promote_noreplace(stage, destination)
    except Exception as error:
        raise RuntimeError(f'build failed; unpromoted staging retained at {stage}: {error}') from error
    return destination / 'manifest.json'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configs', type=Path, default=find_repo() / 'configs/data_configs/CL-VISTA')
    parser.add_argument('--video-root', type=Path, default=Path('/data/mcitlib/CL-VISTA'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261108)
    parser.add_argument('--train-target', type=int, default=3000)
    parser.add_argument('--test-target', type=int, default=1000)
    args = parser.parse_args()
    try:
        output = args.output.resolve()
        project = find_repo()
        for forbidden in (project / 'data/validation_3k_1k', project / 'data/validation_aux'):
            if output == forbidden or forbidden in output.parents:
                raise ValueError(f'output must not be inside existing LLaVA sample data: {output}')
        output = build(args.configs.resolve(), args.video_root.resolve(), output,
                       args.seed, args.train_target, args.test_target)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f'video sample: {error}', file=sys.stderr)
        return 2
    print(json.dumps({'manifest': str(output)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
