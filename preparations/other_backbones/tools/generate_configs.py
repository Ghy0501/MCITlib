#!/usr/bin/env python3
"""Generate an offline configuration inventory, never executable model commands."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

VIDEO_TASKS = ('counting', 'space', 'traffic', 'movie', 'gui', 'science', 'sports', 'star')
UCIT_TASKS = ('ImageNet-R', 'ArxivQA', 'VizWiz', 'IconQA', 'CLEVR-Math', 'Flickr30k')
DCL_TASKS = ('RS', 'Med', 'AD', 'Sci', 'Fin')
LEGACY_METHODS = ('CL-MoE', 'DISCO', 'HiDe', 'KeepLoRA', 'LoRA-FT', 'MR-LoRA',
                  'MoELoRA', 'ModalPrompt', 'OLoRA', 'RegLoRA', 'Replay', 'SMoLoRA')
MODELS = {'InternVL': ('internvl', ('UCIT', 'MLLM-DCL')),
          'Video-LLaVA': ('videollava', ('CL-VISTA',)),
          'VideoLLaMA2': ('videollama2', ('CL-VISTA',))}
ROUTER_MAPPING = {'F': {'domain': 'counting', 'task_id': 1}, 'G': {'domain': 'space', 'task_id': 2},
                  'D': {'domain': 'traffic', 'task_id': 3}, 'A': {'domain': 'movie', 'task_id': 4},
                  'C': {'domain': 'gui', 'task_id': 5}, 'E': {'domain': 'science', 'task_id': 6},
                  'H': {'domain': 'sports', 'task_id': 7}, 'B': {'domain': 'star', 'task_id': 8}}
SMOLORA_ASSET_SHA = {
    'ins_emb_multi_ucit.pkl': '2a4595386ca010a104d3dd2109f2da52c6cabe562aa5d6304e3a469880d4bc37',
    'ins_emb_multi_dcl.pkl': 'ea251206f6ef970a2207e150810e416c134246c3904cffac1b2c91abfcf01721',
    'ins_emb_multi.pkl': 'c8a769183da3b2b41e5b08d09e928fd539e14717d490d2a61ed819ed17ff82ef',
}
REFERENCE_EVIDENCE_ROOT = Path(__file__).resolve().parents[1] / 'reference_evidence'
REFERENCE_EVIDENCE_MANIFEST_SHA = 'c83c4397dda50b0e8e26469ed2a9be5e12462af8eb2801d198450a0f160ef4cf'


def load_reference_evidence() -> dict[str, bytes]:
    """Read immutable source snapshots as data, without importing or executing them."""
    try:
        raw = (REFERENCE_EVIDENCE_ROOT / 'manifest.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != REFERENCE_EVIDENCE_MANIFEST_SHA:
            raise ValueError('reference evidence manifest SHA mismatch')
        manifest = json.loads(raw)
        if manifest.get('schema_version') != 1 or not isinstance(manifest.get('files'), dict):
            raise ValueError('invalid reference evidence manifest')
        contents = {}
        for relative, entry in manifest['files'].items():
            path = Path(relative)
            if path.is_absolute() or '..' in path.parts or not relative.endswith('.txt'):
                raise ValueError(f'unsafe reference evidence path: {relative}')
            if entry.get('source_relative_path') + '.txt' != relative:
                raise ValueError(f'reference evidence source path mismatch: {relative}')
            data = (REFERENCE_EVIDENCE_ROOT / path).read_bytes()
            if len(data) != entry.get('bytes') or hashlib.sha256(data).hexdigest() != entry.get('sha256'):
                raise ValueError(f'reference evidence SHA mismatch: {relative}')
            contents[relative] = data
        return contents
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f'cannot read reference evidence: {error}') from error


def find_repo() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / 'mcitlib').is_dir() and (parent / 'configs/train_configs').is_dir():
            return parent
    raise ValueError('cannot find MCITlib project root; pass --project-root')


def read(path: Path) -> tuple[dict[str, Any], str]:
    content = path.read_bytes()
    data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError(f'expected JSON object: {path}')
    return data, hashlib.sha256(content).hexdigest()


def portable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: portable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [portable(x) for x in value]
    if not isinstance(value, str):
        return value
    substitutions = [('/your_ckpts_path/MCITlib_v3/LLaVA/SMoLoRA/', '${PROJECT_ROOT}/assets/smolora/'),
                     ('/your_ckpts_path/MCITlib_v3/InternVL/SMoLoRA/', '${PROJECT_ROOT}/assets/smolora/'),
                     ('/your_ckpts_path/MCITlib_v3/Video-LLaVA/SMoLoRA/', '${PREPARATIONS_ROOT}/assets/Video-LLaVA/smolora/'),
                     ('/your_ckpts_path/MCITlib_v3/VideoLLaMA2/SMoLoRA/', '${PREPARATIONS_ROOT}/assets/VideoLLaMA2/smolora/'),
                     ('/opt/mcitlib-example/MCITlib/checkpoints/', '${CKPT_ROOT}/MCITlib/'),
                     ('/data/mcitlib/ckpt_VISTA/', '${CKPT_ROOT}/MCITlib/'),
                     ('./MCITlib/', '${CKPT_ROOT}/MCITlib/'),
                     ('/opt/mcitlib-example/clip-vit-large-patch14-336', '${MODEL_ROOT}/clip-vit-large-patch14-336'),
                     ('/your_model_path/Qwen3-30B-A3B-Instruct-2507', '${JUDGE_MODEL}'),
                     ('/your_model_path/Internvl-chat-7b', '${MODEL_ROOT}/InternVL-Chat-ViT-6B-Vicuna-7B'),
                     ('/your_model_path', '${MODEL_ROOT}'), ('/your_data_path/CL-VISTA', '${CLVISTA_ROOT}'),
                     ('/your_data_path', '${DATA_ROOT}'), ('/your_ckpts_path', '${CKPT_ROOT}'),
                     ('/your_results_path', '${RESULTS_ROOT}'), ('./results/', '${RESULTS_ROOT}/')]
    for old, new in substitutions:
        value = value.replace(old, new)
    return value


def sample_training(original: dict[str, Any], method: str, gpu_count: int) -> dict[str, Any]:
    if gpu_count not in (2, 4):
        raise ValueError('only two- or four-GPU nominal plans are available')
    for key in ('gpu_num', 'epoch', 'batch_size', 'grad_acc', 'lr'):
        if key not in original or isinstance(original[key], bool) or not isinstance(original[key], (int, float)):
            raise ValueError(f'legacy main-training configuration lacks numeric {key}')
    output = copy.deepcopy(portable(original))
    output.update(gpu_num=gpu_count, batch_size=1, grad_acc=64 // gpu_count,
                  lr=0.0002, epoch=10 if method == 'ModalPrompt' else 1)
    return output


def source(path: Path, project: Path) -> dict[str, Any]:
    value, fingerprint = read(path)
    return {'path': str(path.relative_to(project)), 'sha256': fingerprint, 'config': portable(value)}


def dataset_plan(project: Path, benchmark: str, tasks: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    sample_manifest_path = project / 'configs/validation_3k_1k/manifest.json'
    image_manifest, image_manifest_sha = read(sample_manifest_path) if sample_manifest_path.is_file() else ({}, None)
    aux_manifest_path = project / 'data/validation_aux/manifest.json'
    image_aux, image_aux_sha = read(aux_manifest_path) if aux_manifest_path.is_file() else ({}, None)
    image_sample_bound = bool(image_manifest_sha and image_aux.get('sample_manifest_sha256') == image_manifest_sha)
    video_manifest_path = project / 'data/validation_3k_1k_video/manifest.json'
    video_manifest, video_sha = read(video_manifest_path) if video_manifest_path.is_file() else ({}, None)
    video_aux_path = project / 'data/validation_3k_1k_video_aux/manifest.json'
    video_aux, video_aux_sha = read(video_aux_path) if video_aux_path.is_file() else ({}, None)
    for number, name in enumerate(tasks, 1):
        if benchmark == 'CL-VISTA':
            original = source(project / 'configs/data_configs/CL-VISTA' / f'{name}.json', project)
            full_auxiliary = {}
            for role in ('replay', 'router'):
                configured = original['config'].get(f'{role}_path')
                resolved = (configured.replace('${CLVISTA_ROOT}', '/data/mcitlib/CL-VISTA')
                            if isinstance(configured, str) else '')
                exists = Path(resolved).is_file()
                full_auxiliary[role] = {'configured_path': configured, 'exists': exists,
                                        'status': 'present_unverified' if exists else 'missing_source',
                                        'basis': 'original raw CL-VISTA data configuration'}
            sample = {'train_path': f'${{VIDEO_SAMPLE_ROOT}}/{name}/train.json',
                      'test_q_path': f'${{VIDEO_SAMPLE_ROOT}}/{name}/test_questions.json',
                      'test_a_path': f'${{VIDEO_SAMPLE_ROOT}}/{name}/test_answers.json',
                      'train_folder': '${CLVISTA_ROOT}', 'test_folder': '${CLVISTA_ROOT}',
                      'replay_path': None, 'router_path': None}
            meta = next((x for x in video_manifest.get('domains', []) if x.get('task') == name), None)
            available = bool(meta and meta.get('status') == 'sample_ready')
            if available:
                for filename in ('train.json', 'test_questions.json', 'test_answers.json'):
                    path = project / 'data/validation_3k_1k_video' / name / filename
                    info = meta['outputs'][filename]
                    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != info['sha256']:
                        available = False
            aux_meta = next((x for x in video_aux.get('domains', []) if x.get('task') == name), None)
            aux_bound = bool(aux_meta and video_sha and video_aux.get('base_manifest_sha256') == video_sha and
                             aux_meta.get('status') == 'source_qa_recovered_and_hash_verified')
            for role in ('replay', 'router'):
                path = project / 'data/validation_3k_1k_video_aux' / name / f'{role}.json'
                info = aux_meta.get(role, {}) if aux_meta else {}
                role_ok = bool(aux_bound and path.is_file() and
                               hashlib.sha256(path.read_bytes()).hexdigest() == info.get('sha256'))
                sample[f'{role}_path'] = f'${{VIDEO_AUX_ROOT}}/{name}/{role}.json' if role_ok else None
                aux_bound = aux_bound and role_ok
            rows.append({'task': name, 'full': original, 'full_auxiliary': full_auxiliary, 'sample': sample,
                         'sample_status': 'verified_manifest_hashes' if available else 'not_built_or_hash_mismatch',
                         'sample_manifest': {'path': str(video_manifest_path), 'sha256': video_sha},
                         'auxiliary_manifest': {'path': str(video_aux_path), 'sha256': video_aux_sha},
                         'auxiliary_status': ('hashes_match_manifest_backend_pending' if aux_bound else
                                              'replay_and_router_pending_source_validation')})
        else:
            folder = 'ucit' if benchmark == 'UCIT' else 'mllm_dcl'
            sample_dir = project / 'data/validation_3k_1k' / folder / name
            original = source(project / 'configs/data_configs' / benchmark / f'{name}.json', project)
            full = copy.deepcopy(original['config'])
            annotation_root = '${UCIT_ANNOT_ROOT}' if benchmark == 'UCIT' else '${DCL_MEDIA_ROOT}'
            full_root = Path('/data/mcitlib/MLLM-CL/UCIT') if benchmark == 'UCIT' else Path('/data/mcitlib/data/Domain_data')
            full['train_path'] = f'{annotation_root}/{name}/train.json'
            full['test_path'] = f'{annotation_root}/{name}/test.json'
            full_auxiliary = {}
            for role, relative in (('replay', f'{name}/replay.json'),
                                   ('router', f'router_json_20/task{number}.json')):
                exists = (full_root / relative).is_file()
                candidate = f'{annotation_root}/{relative}'
                full[f'{role}_path'] = candidate if exists else None
                count = None
                status = 'present_unverified' if exists else 'missing_source'
                if role == 'router' and exists:
                    try:
                        records = json.loads((full_root / relative).read_text(encoding='utf-8'))
                        if isinstance(records, list):
                            count = len(records)
                            status = 'nonempty_source' if count else 'blocked_empty_source'
                        else:
                            status = 'blocked_invalid_source'
                    except (OSError, ValueError):
                        status = 'blocked_invalid_source'
                full_auxiliary[role] = {'configured_path': candidate if exists else None,
                                        'expected_path': candidate, 'exists': exists, 'rows': count,
                                        'status': status,
                                        'basis': ('configs/paths.yaml stage mapping; task1 uses the same checked directory'
                                                  if role == 'router' else 'legacy domain replay path')}
            for key in ('train_folder', 'test_folder'):
                if key in full:
                    full[key] = '${UCIT_MEDIA_ROOT}' if benchmark == 'UCIT' else f'${{DCL_MEDIA_ROOT}}/{name}'
            if 'router_folder' in full:
                full['router_folder'] = '${UCIT_MEDIA_ROOT}' if benchmark == 'UCIT' else '${DCL_MEDIA_ROOT}'
            if 'anno_path' in full:
                full['anno_path'] = f'{annotation_root}/{name}/val_coco_type_3000.json'
            sample = copy.deepcopy(full)
            sample.update(train_path=f'${{IMAGE_SAMPLE_ROOT}}/{folder}/{name}/train.json',
                          test_path=f'${{IMAGE_SAMPLE_ROOT}}/{folder}/{name}/test.json')
            entry = next((x for x in image_aux.get('tasks', []) if x.get('benchmark') == folder and x.get('name') == name), None)
            verified_aux = bool(image_sample_bound and entry)
            for role in ('replay', 'router'):
                path = project / 'data/validation_aux' / folder / name / f'{role}.json'
                expected = entry.get(f'{role}_output_sha256') if entry else None
                role_ok = bool(expected and path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected)
                verified_aux = verified_aux and role_ok
                sample[f'{role}_path'] = f'${{IMAGE_AUX_ROOT}}/{folder}/{name}/{role}.json' if role_ok and image_sample_bound else None
            sample['router_folder'] = '${UCIT_MEDIA_ROOT}' if benchmark == 'UCIT' else '${DCL_MEDIA_ROOT}'
            if 'anno_path' in sample:
                sample['anno_path'] = f'${{IMAGE_SAMPLE_ROOT}}/{folder}/{name}/val_coco_type_3000.json'
            manifest_task = next((x for x in image_manifest.get('tasks', [])
                                  if x['benchmark'] == folder and x['name'] == name), {})
            manifest_splits = manifest_task.get('splits', {})
            available = all(split in manifest_splits and (sample_dir / f'{split}.json').is_file() and
                            hashlib.sha256((sample_dir / f'{split}.json').read_bytes()).hexdigest() ==
                            manifest_splits[split]['output_sha256'] for split in ('train', 'test'))
            rows.append({'task': name, 'full': {'source_config': original, 'resolved_config': full,
                                               'source_evidence': {split: {key: meta.get(key) for key in
                                                   ('source', 'source_sha256', 'source_count', 'sample_count', 'output', 'output_sha256')}
                                                   for split, meta in manifest_splits.items()},
                                               'main_files_exist': {key: (full_root / name / f'{key}.json').is_file()
                                                                    for key in ('train', 'test')}},
                         'full_auxiliary': full_auxiliary, 'sample': sample,
                         'sample_status': 'verified_manifest_hashes' if available else 'missing_or_hash_mismatch',
                         'manifest': {'path': str(sample_manifest_path), 'sha256': image_manifest_sha},
                         'auxiliary_manifest': {'path': str(aux_manifest_path), 'sha256': image_aux_sha},
                         'auxiliary_status': 'hashes_match_manifest_backend_pending' if verified_aux else 'blocked_missing_or_conflicting_aux'})
    return rows


def general_inventory(project: Path) -> list[dict[str, Any]]:
    rows = []
    deferred = {'mvbench': '${MVBENCH_ROOT}', 'longvideobench': '${LONGVIDEOBENCH_ROOT}',
                'videomme': '${VIDEOMME_ROOT}', 'mmvu': '${MMVU_ROOT}', 'nextqa': '${NEXTQA_ROOT}',
                'mmbench-video': '${MMBENCH_VIDEO_ROOT}'}
    for path in sorted((project / 'configs/data_configs/General_benchmark').glob('*.json')):
        content, fingerprint = read(path)
        checks = ({} if path.stem.lower() in deferred else
                  {key: Path(value.replace('/your_data_path', '/data/mcitlib')).exists()
                   for key, value in content.items() if isinstance(value, str) and value.startswith('/your_data_path/')})
        rows.append({'name': path.stem, 'source': str(path), 'sha256': fingerprint, 'config': portable(content),
                     'root_placeholder': deferred.get(path.stem.lower()), 'local_paths_exist': checks,
                     'status': 'deferred_by_user' if path.stem.lower() in deferred else 'available_unvalidated' if checks and all(checks.values())
                     else 'missing_or_unvalidated', 'downloaded': False})
    local = project.parents[1] / 'MCITlib_video/configs/data_configs/CVU'
    for name in ('mvbench', 'longvideobench', 'videomme'):
        path = local / f'{name}.json'
        if path.is_file():
            content, fingerprint = read(path)
            rows.append({'name': f'local_legacy_{name}', 'source': str(path), 'sha256': fingerprint,
                         'config': content, 'local_paths_exist': {}, 'root_placeholder': deferred[name],
                         'status': 'deferred_by_user',
                         'downloaded': False})
    return rows


def video_router_protocol(_project: Path, model: str, evidence: dict[str, bytes] | None = None) -> dict[str, Any]:
    evidence = load_reference_evidence() if evidence is None else evidence
    extractor = f'{model}/MR-LoRA/scripts/extract_router_result.py.txt'
    inference = (f'{model}/MR-LoRA/videollava/eval/video/run_inference_video_qa_router.py.txt' if model == 'Video-LLaVA'
                 else f'{model}/MR-LoRA/videollama2/eval/inference_video_qa_router.py.txt')
    extractor_bytes, inference_bytes = evidence[extractor], evidence[inference]
    tree = ast.parse(extractor_bytes)
    mappings = [ast.literal_eval(node.value) for node in ast.walk(tree) if isinstance(node, ast.Assign) and
                any(isinstance(target, ast.Name) and target.id == 'cl_mapping' for target in node.targets)]
    prompt = inference_bytes.decode('utf-8')
    descriptions = ('A: A movie expert', 'B: A reasoning expert', 'C: A GUI', 'D: A traffic',
                    'E: A science', 'F: A video counting', 'G: A spatial', 'H: A sports')
    matches = len(mappings) == 1 and mappings[0] == ROUTER_MAPPING and all(text in prompt for text in descriptions)
    return {'letter_to_expert_stage': ROUTER_MAPPING, 'task_order_letters': ['F', 'G', 'D', 'A', 'C', 'E', 'H', 'B'],
            'extractor': f'reference_evidence/{extractor}', 'extractor_sha256': hashlib.sha256(extractor_bytes).hexdigest(),
            'inference': f'reference_evidence/{inference}', 'inference_sha256': hashlib.sha256(inference_bytes).hexdigest(),
            'prompt_and_extractor_mapping': 'matched' if matches else 'mismatch_blocked',
            'readiness': 'blocked_cross_domain_join_and_typed_id_audit',
            'note': 'Legacy extractor reads selected expert-domain results and silently skips missing IDs; '
                    'do not use its partial denominator as a routed score.'}


def merged_output_evidence(_project: Path, model: str, method: str,
                           snapshots: dict[str, bytes] | None = None) -> dict[str, Any]:
    if method not in ('KeepLoRA', 'RegLoRA'):
        return {}
    snapshots = load_reference_evidence() if snapshots is None else snapshots
    evidence = {}
    for kind, filename in (('first', 'Task1.sh'), ('later', 'Taskn.sh')):
        path = f'{model}/{method}/scripts/MCITlib/Train/{filename}.txt'
        raw = snapshots[path]
        declared = b'SAVE_PATH="${OUTPUT_DIR}_merged"' in raw and b'merge_lora_weights.py' in raw
        evidence[kind] = {'source': f'reference_evidence/{path}', 'sha256': hashlib.sha256(raw).hexdigest(),
                          'status': 'declared_in_legacy_script' if declared else 'missing_producer'}
    return evidence


def generate(project: Path) -> dict[str, Any]:
    reference_evidence = load_reference_evidence()
    model_root = project.parents[1]
    variables = {'PROJECT_ROOT': str(project), 'MODEL_ROOT': str(model_root),
                 'DATA_ROOT': '/data/mcitlib', 'CLVISTA_ROOT': '/data/mcitlib/CL-VISTA',
                 'UCIT_ANNOT_ROOT': '/data/mcitlib/MLLM-CL/UCIT',
                 'UCIT_MEDIA_ROOT': '/data/mcitlib/MLLM-CL/UCIT/datasets',
                 'DCL_MEDIA_ROOT': '/data/mcitlib/data/Domain_data',
                 'IMAGE_SAMPLE_ROOT': str(project / 'data/validation_3k_1k'),
                 'IMAGE_AUX_ROOT': str(project / 'data/validation_aux'),
                 'VIDEO_SAMPLE_ROOT': str(project / 'data/validation_3k_1k_video'),
                 'VIDEO_AUX_ROOT': str(project / 'data/validation_3k_1k_video_aux'),
                 'PREPARATIONS_ROOT': str(project / 'preparations/other_backbones'),
                 'JUDGE_MODEL': '/data/mcitlib/models/Qwen3-30B-A3B-Instruct-2507',
                 'CKPT_ROOT': '${PROJECT_ROOT}/ckpts/offline_validation/${RUN_ID}',
                 'RESULTS_ROOT': '${PROJECT_ROOT}/results/offline_validation/${RUN_ID}',
                 'LOG_ROOT': '${PROJECT_ROOT}/logs/offline_validation/${RUN_ID}',
                 'RUN_ID': '${CALLER_MUST_SPECIFY_RUN_ID}'}
    weight_paths = {'InternVL': ['InternVL-Chat-ViT-6B-Vicuna-7B', 'InternViT-6B-224px', 'clip-vit-large-patch14-336'],
                    'Video-LLaVA': ['Video-LLaVA-7B', 'LanguageBind_Video_merge', 'LanguageBind_Image'],
                    'VideoLLaMA2': ['VideoLLaMA2-7B', 'clip-vit-large-patch14-336']}
    models: dict[str, Any] = {}
    variants = []
    combinations = set()
    for model, (model_name, benchmarks) in MODELS.items():
        original = source(project / 'configs/model_configs' / f'{model_name}.json', project)
        overrides = ({'mm_image_tower': '${MODEL_ROOT}/LanguageBind_Image',
                      'mm_video_tower': '${MODEL_ROOT}/LanguageBind_Video_merge'} if model == 'Video-LLaVA' else
                     {'mm_vision_tower': '${MODEL_ROOT}/clip-vit-large-patch14-336'} if model == 'VideoLLaMA2' else {})
        models[model] = {'legacy_model_config': original, 'loader_config_overrides': overrides,
                         'model_assets': {name: {'path': str(model_root / name),
                                                 'exists': (model_root / name / 'config.json').is_file()}
                                          for name in weight_paths[model]},
                         'loader_status': 'pending_not_integrated',
                         'note': 'Overrides are explicit plans; original model files are not modified.'}
        for benchmark in benchmarks:
            tasks = VIDEO_TASKS if benchmark == 'CL-VISTA' else UCIT_TASKS if benchmark == 'UCIT' else DCL_TASKS
            datasets = dataset_plan(project, benchmark, tasks)
            for method in LEGACY_METHODS:
                base = project / 'configs/train_configs' / method / model / benchmark
                if not base.is_dir():
                    continue
                combinations.add((model, method))
                stages = []
                for number, task in enumerate(tasks, 1):
                    train_path, eval_path = base / 'train' / f'task{number}.json', base / 'eval' / f'task{number}.json'
                    original_train, train_sha = read(train_path)
                    eval_info = source(eval_path, project)
                    sample_eval_4 = copy.deepcopy(eval_info['config'])
                    sample_eval_2 = copy.deepcopy(eval_info['config'])
                    if 'gpu_num' in sample_eval_4:
                        sample_eval_4['gpu_num'], sample_eval_2['gpu_num'] = 4, 2
                    full_train = portable(original_train)
                    sample_train_4 = sample_training(original_train, method, 4)
                    sample_train_2 = sample_training(original_train, method, 2)
                    corrections = []
                    if (model, method, benchmark, number) == ('InternVL', 'ModalPrompt', 'UCIT', 2):
                        old = '${CKPT_ROOT}/MCITlib/UCIT/InternVL/ModalPrompt/Task1_internvl'
                        new = '${CKPT_ROOT}/MCITlib/UCIT/InternVL/ModalPrompt/Task2_internvl'
                        next_train, _ = read(base / 'train/task3.json')
                        if (full_train.get('output_dir') != old or full_train.get('previous_model') != old or
                                eval_info['config'].get('model_path') != new or portable(next_train).get('previous_model') != new):
                            raise ValueError('ModalPrompt UCIT task2 path correction preconditions changed')
                        for train in (full_train, sample_train_4, sample_train_2):
                            train['output_dir'] = new
                        corrections.append({'field': 'output_dir', 'before': old, 'after': new,
                                            'reason': 'task2 duplicated task1 output; task2 eval and task3 previous_model specify Task2',
                                            'source_train_sha256': train_sha})
                    external_inputs = []
                    if 'ins_emb' in full_train:
                        portable_asset = full_train['ins_emb']
                        if portable_asset.startswith('${PROJECT_ROOT}/assets/smolora/'):
                            asset = project / 'assets/smolora' / Path(portable_asset).name
                        elif portable_asset.startswith('${PREPARATIONS_ROOT}/assets/'):
                            asset = Path(__file__).resolve().parents[1] / portable_asset.removeprefix('${PREPARATIONS_ROOT}/')
                        else:
                            raise ValueError(f'unrecognized SMoLoRA input asset {portable_asset}')
                        if not asset.is_file():
                            raise ValueError(f'missing SMoLoRA input asset {asset}')
                        fingerprint = hashlib.sha256(asset.read_bytes()).hexdigest()
                        if fingerprint != SMOLORA_ASSET_SHA.get(asset.name):
                            raise ValueError(f'SMoLoRA input differs from verified source: {asset}')
                        external_inputs.append({'field': 'ins_emb', 'original_config_value': original_train['ins_emb'],
                                                'path': portable_asset, 'sha256': fingerprint,
                                                'source_basis': 'byte-matched original MCITlib_v3_ref SMoLoRA asset'})
                    stages.append({'stage': number, 'task': task,
                        'source_train': {'path': str(train_path.relative_to(project)), 'sha256': train_sha},
                        'full_train': full_train, 'full_eval': eval_info,
                        'sample_train_4gpu': sample_train_4,
                        'sample_train_2gpu': sample_train_2,
                        'path_corrections': corrections, 'external_inputs': external_inputs,
                        'sample_eval_4gpu': sample_eval_4, 'sample_eval_2gpu': sample_eval_2,
                        'sample_epoch_policy': 'explicit_modalprompt_10' if method == 'ModalPrompt' else 'default_1',
                        'resource_status': 'planned_not_gpu_verified'})
                auxiliary = []
                for directory in sorted(base.iterdir()):
                    if directory.is_dir() and directory.name not in ('train', 'eval'):
                        for path in sorted(directory.glob('task*.json')):
                            entry = source(path, project)
                            corrections = []
                            if ((model, method, benchmark, directory.name, path.name) ==
                                    ('InternVL', 'KeepLoRA', 'UCIT', 'train_pre', 'task1.json')):
                                old = '${MODEL_ROOT}/internvl-v1.5-7b'
                                new = models[model]['legacy_model_config']['config']['model_name'].replace(
                                    '${model_root}', '${MODEL_ROOT}')
                                dcl_path = project / 'configs/train_configs/KeepLoRA/InternVL/MLLM-DCL/train_pre/task1.json'
                                current = entry['config'].get('model_path')
                                if (current not in (old, new) or
                                        source(dcl_path, project)['config'].get('model_path') != new):
                                    raise ValueError('KeepLoRA UCIT initial backbone correction preconditions changed')
                                if current == old:
                                    entry['config']['model_path'] = new
                                    corrections.append({'field': 'model_path', 'before': old, 'after': new,
                                                        'reason': 'initial gradients must use the selected training backbone, as in DCL',
                                                        'source_sha256': entry['sha256']})
                            auxiliary.append({'role': directory.name, **entry, 'path_corrections': corrections})
                legacy_method = project.parent / model / method
                if method in ('Replay', 'MR-LoRA') and benchmark != 'CL-VISTA':
                    aux_status = ('source_hashes_verified_backend_pending' if all(
                        row['auxiliary_status'] == 'hashes_match_manifest_backend_pending' for row in datasets)
                        else 'blocked_missing_or_conflicting_aux')
                elif method == 'Replay' and benchmark == 'CL-VISTA':
                    aux_status = ('source_hashes_verified_backend_pending' if all(
                        row['auxiliary_status'] == 'hashes_match_manifest_backend_pending' for row in datasets)
                        else 'blocked_pending_router_source_validation')
                elif method == 'MR-LoRA':
                    aux_status = ('router_input_hashes_verified_join_blocked' if all(
                        row['auxiliary_status'] == 'hashes_match_manifest_backend_pending' for row in datasets)
                        else 'blocked_pending_validation')
                else:
                    aux_status = 'not_applicable'
                if method in ('Replay', 'MR-LoRA'):
                    role = 'replay' if method == 'Replay' else 'router'
                    statuses = {row['full_auxiliary'][role]['status'] for row in datasets}
                    if 'missing_source' in statuses:
                        full_aux_status = f'blocked_missing_full_{role}_files'
                    elif 'blocked_empty_source' in statuses:
                        full_aux_status = f'blocked_empty_full_{role}_source'
                    elif 'blocked_invalid_source' in statuses:
                        full_aux_status = f'blocked_invalid_full_{role}_source'
                    else:
                        full_aux_status = 'source_paths_exist_backend_pending'
                    if (method == 'MR-LoRA' and benchmark == 'CL-VISTA' and
                            full_aux_status == 'source_paths_exist_backend_pending'):
                        full_aux_status = 'router_inputs_exist_join_blocked'
                else:
                    full_aux_status = 'not_applicable'
                variant = {'model': model, 'method': method, 'benchmark': benchmark, 'tasks': list(tasks),
                                 'datasets': datasets, 'stages': stages, 'auxiliary_configs_unmodified': auxiliary,
                                 'legacy_method_dir': str(legacy_method), 'legacy_method_exists': legacy_method.is_dir(),
                                 'sample_auxiliary': aux_status,
                                 'full_auxiliary': full_aux_status,
                                 'merged_output_evidence': merged_output_evidence(project, model, method, reference_evidence),
                                 'execution_readiness': 'blocked_loader_and_entry_not_integrated',
                                 'note': 'Offline plans only. No CLI integration, decoder, hardware or scoring validation implied.'}
                if method == 'MR-LoRA' and benchmark == 'CL-VISTA':
                    variant['router_protocol'] = video_router_protocol(project, model, reference_evidence)
                variants.append(variant)
    expected = {'InternVL': 12, 'Video-LLaVA': 12, 'VideoLLaMA2': 11}
    observed = {model: sum(1 for entry in combinations if entry[0] == model) for model in MODELS}
    if observed != expected:
        raise ValueError(f'legacy combination inventory differs: expected {expected}, got {observed}')
    return {'schema_version': 1, 'purpose': 'offline preparation only; no model execution',
            'reference_evidence_manifest': {'path': 'reference_evidence/manifest.json',
                                            'sha256': REFERENCE_EVIDENCE_MANIFEST_SHA},
            'variables': variables, 'models': models, 'method_backbone_combinations': len(combinations),
            'benchmark_variants': len(variants), 'coverage': observed, 'variants': variants,
            'sample_policy': {'base_learning_rate': 0.0002, 'nominal_global_batch': 64,
                              'default_main_epochs': 1, 'modalprompt_main_epochs': 10,
                              'modalprompt_reason': '3000-example sample budget; explicit override for video full epochs of 1',
                              'router_and_method_specific_parameters': 'preserve legacy auxiliary files',
                              'resource_status': 'plans only; not validated on hardware'},
            'mllm_acl': 'InternVL legacy full configs remain in source tree; no sample generated or scheduled',
            'general_benchmarks': general_inventory(project)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=find_repo())
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = generate(args.project_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(args.output.resolve()), 'coverage': result['coverage'],
                      'benchmark_variants': result['benchmark_variants']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
