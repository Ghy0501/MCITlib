import copy
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))


def load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sampler = load('video_sample')
configgen = load('generate_configs')
auxiliary = load('video_aux')
materializer = load('materialize')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def setup_domain(tmp_path):
    root = tmp_path / 'data'
    root.mkdir()
    (root / 'a.mp4').write_bytes(b'a')
    (root / 'b.mp4').write_bytes(b'b')
    (root / 'conflict.avi').write_bytes(b'avi')
    (root / 'conflict.webm').write_bytes(b'webm')
    train = [{'id': 1, 'video': 'a.mp4', 'conversations': [{'from': 'human', 'value': 'question'}]},
             {'id': '1', 'video': 'a.mp4', 'conversations': [{'from': 'human', 'value': 'question'}]},
             {'id': 2, 'video': 'missing.mp4', 'conversations': [{'from': 'human', 'value': 'question2'}]}]
    questions = [{'question_id': 1, 'video_name': 'a', 'question': 'q1'},
                 {'question_id': '1', 'video_name': 'b', 'question': 'q2'},
                 {'question_id': 3, 'video_name': 'conflict', 'question': 'q3'},
                 {'question_id': 4, 'video_name': 'missing', 'question': 'q4'}]
    answers = [{'question_id': '1', 'answer': 's'}, {'question_id': 1, 'answer': 'i'},
               {'question_id': 4, 'answer': 'm'}, {'question_id': 3, 'answer': 'c'}]
    for name, value in [('train.json', train), ('q.json', questions), ('a.json', answers)]:
        save(root / name, value)
    cfg = {'train_path': '/your_data_path/CL-VISTA/train.json',
           'test_q_path': '/your_data_path/CL-VISTA/q.json', 'test_a_path': '/your_data_path/CL-VISTA/a.json'}
    return root, cfg, train, questions, answers


def test_paired_sampling_preserves_typed_ids_rows_and_overlaps(tmp_path):
    root, cfg, train, questions, answers = setup_domain(tmp_path)
    output, info = sampler.prepare_domain('counting', cfg, root, 99, 3, 3)
    output2, info2 = sampler.prepare_domain('counting', cfg, root, 99, 3, 3)
    assert output == output2 and info == info2
    assert output['train.json'] == train[:2]
    assert output['test_questions.json'] == questions[:2]
    assert output['test_answers.json'] == [answers[1], answers[0]]
    assert type(output['test_questions.json'][0]['question_id']) is int
    assert type(output['test_questions.json'][1]['question_id']) is str
    assert info['status'] == 'sample_shortfall'
    assert info['video_overlap']['sample_unique_files'] == 1
    assert info['duplicate_pairs']['train_exact']['groups'] == 1
    assert info['duplicate_pairs']['train_normalized']['groups'] == 1
    assert len(info['excluded_missing_or_conflicting_media']['train']) == 1
    assert {x['reason'] for x in info['excluded_missing_or_conflicting_media']['test']} == {
        'different_first_match', 'missing_common_video'}


def test_duplicate_or_mismatched_id_is_rejected(tmp_path):
    root, cfg, _, questions, answers = setup_domain(tmp_path)
    answers[1]['question_id'] = '1'
    save(root / 'a.json', answers)
    with pytest.raises(ValueError, match='duplicate'):
        sampler.prepare_domain('counting', cfg, root, 1, 2, 2)
    answers[1]['question_id'] = 'not-int-1'
    save(root / 'a.json', answers)
    with pytest.raises(ValueError, match='question/answer IDs differ'):
        sampler.prepare_domain('counting', cfg, root, 1, 2, 2)
    assert questions[0]['question_id'] == 1


def test_seed_and_video_path_safety(tmp_path):
    choices = list(range(100))
    assert sampler.sample_indices(choices, 10, 7, 'space', 'train') == sampler.sample_indices(choices, 10, 7, 'space', 'train')
    assert sampler.sample_indices(choices, 10, 7, 'space', 'train') != sampler.sample_indices(choices, 10, 8, 'space', 'train')
    with pytest.raises(ValueError, match='relative path'):
        sampler.relative_media(tmp_path, '../other.mp4')
    with pytest.raises(ValueError, match='relative path'):
        sampler.relative_media(tmp_path, '/absolute.mp4')


def test_build_writes_hashed_manifest_and_refuses_overwrite(tmp_path, monkeypatch):
    root, cfg, _, _, _ = setup_domain(tmp_path)
    configs = tmp_path / 'configs'
    save(configs / 'counting.json', cfg)
    monkeypatch.setattr(sampler, 'TASKS', ('counting',))
    destination = tmp_path / 'sample'
    manifest_path = sampler.build(configs, root, destination, 7, 3, 3)
    manifest = json.loads(manifest_path.read_text())
    entry = manifest['domains'][0]
    result_path = destination / 'counting/train.json'
    assert entry['outputs']['train.json']['sha256'] == sampler.digest(result_path.read_bytes())
    assert entry['outputs']['train.json']['path'] == str(result_path)
    assert manifest['dataset_status'] == 'shortfall'
    with pytest.raises(FileExistsError):
        sampler.build(configs, root, destination, 7, 3, 3)


def test_atomic_promotion_never_replaces_existing_directory(tmp_path):
    old, new = tmp_path / 'stage', tmp_path / 'output'
    old.mkdir()
    new.mkdir()
    (old / 'new').write_text('new')
    (new / 'existing').write_text('keep')
    with pytest.raises(OSError):
        sampler.promote_noreplace(old, new)
    assert (new / 'existing').read_text() == 'keep'
    dest = tmp_path / 'fresh'
    sampler.promote_noreplace(old, dest)
    assert (dest / 'new').read_text() == 'new'


def test_sample_profile_changes_only_authorized_fields():
    original = {'gpu_num': 2, 'epoch': 1, 'batch_size': 4, 'grad_acc': 2, 'lr': 0.0001,
                'rank': 32, 'video_frames': 8, 'special_lr': 0.004, 'model': '/your_model_path/foo'}
    normal = configgen.sample_training(original, 'LoRA-FT', 4)
    modal = configgen.sample_training(original, 'ModalPrompt', 2)
    assert normal['gpu_num'] * normal['batch_size'] * normal['grad_acc'] == 64
    assert modal['gpu_num'] * modal['batch_size'] * modal['grad_acc'] == 64
    assert (normal['batch_size'], normal['grad_acc']) == (1, 16)
    assert (modal['batch_size'], modal['grad_acc']) == (1, 32)
    assert normal['epoch'] == 1 and modal['epoch'] == 10 and original['epoch'] == 1
    assert normal['lr'] == modal['lr'] == 0.0002
    for key in ('rank', 'video_frames', 'special_lr'):
        assert normal[key] == modal[key] == original[key]
    assert normal['model'] == '${MODEL_ROOT}/foo'


def test_materialized_47_variants_have_unique_roots_and_no_placeholders(tmp_path):
    plan = Path(__file__).resolve().parents[1] / 'configs/offline-plan.json'
    if not plan.is_file():
        pytest.skip('local integration plan was not generated')
    destination = tmp_path / 'effective'
    materializer.materialize(plan, destination, 'cpu-only-test', 'sample', 4,
                             {'PREPARATIONS_ROOT': str(plan.parents[1])})
    manifest = json.loads((destination / 'manifest.json').read_text())
    assert len(manifest['namespace_roots']) == 47
    for name in ('CKPT_ROOT', 'RESULTS_ROOT', 'LOG_ROOT'):
        assert len({item[name] for item in manifest['namespace_roots']}) == 47
    for namespace in manifest['namespace_roots']:
        base = destination / namespace['model'] / namespace['method'] / namespace['benchmark']
        outputs = []
        for path in sorted((base / 'train').glob('task*.json')):
            train = json.loads(path.read_text())
            evaluation = json.loads((base / 'eval' / path.name).read_text())
            output = Path(train['output_dir'])
            assert output.is_relative_to(Path(namespace['CKPT_ROOT']))
            assert output not in outputs
            assert evaluation['model_path'] in (str(output), str(output) + '_merged')
            assert Path(evaluation['result_path']).is_relative_to(Path(namespace['RESULTS_ROOT']))
            if 'previous_model' in train:
                previous = train['previous_model'].removesuffix('_merged')
                assert Path(previous) in outputs
            outputs.append(output)
        for path in (base / 'aux').rglob('*.json'):
            cfg = json.loads(path.read_text())
            if 'output_dir' in cfg:
                assert Path(cfg['output_dir']).is_relative_to(Path(namespace['CKPT_ROOT']))
            if 'result_path' in cfg:
                assert Path(cfg['result_path']).is_relative_to(Path(namespace['RESULTS_ROOT']))
    blocked = {(v['model'], v['method'], v['benchmark']) for v in manifest['variants'] if v['missing_producers']}
    assert blocked == {('InternVL', 'RegLoRA', 'UCIT'), ('InternVL', 'RegLoRA', 'MLLM-DCL'),
                       ('VideoLLaMA2', 'RegLoRA', 'CL-VISTA')}
    assert all(not v['execution_allowed'] for v in manifest['variants'])
    image_router = next(v for v in manifest['variants'] if (v['model'], v['method'], v['benchmark']) ==
                        ('InternVL', 'MR-LoRA', 'UCIT'))
    assert image_router['full_auxiliary'] == 'blocked_empty_full_router_source'
    assert image_router['full_auxiliary_files']['ImageNet-R']['router']['rows'] == 0
    for path in destination.rglob('*.json'):
        text = path.read_text()
        assert '/your_' not in text and '${' not in text


def test_plan_binds_verified_aux_but_blocks_video_router_scoring():
    path = Path(__file__).resolve().parents[1] / 'configs/offline-plan.json'
    if not path.is_file():
        pytest.skip('local integration plan was not generated')
    plan = json.loads(path.read_text())
    assert plan['coverage'] == {'InternVL': 12, 'Video-LLaVA': 12, 'VideoLLaMA2': 11}
    assert len(plan['variants']) == 47
    for variant in plan['variants']:
        assert variant['execution_readiness'] == 'blocked_loader_and_entry_not_integrated'
        for dataset in variant['datasets']:
            assert dataset['sample_status'] == 'verified_manifest_hashes'
            assert dataset['auxiliary_status'] == 'hashes_match_manifest_backend_pending'
            assert dataset['sample']['replay_path'] and dataset['sample']['router_path']
        if variant['benchmark'] == 'CL-VISTA' and variant['method'] == 'MR-LoRA':
            assert variant['sample_auxiliary'] == 'router_input_hashes_verified_join_blocked'
            protocol = variant['router_protocol']
            assert protocol['task_order_letters'] == ['F', 'G', 'D', 'A', 'C', 'E', 'H', 'B']
            assert protocol['prompt_and_extractor_mapping'] == 'matched'
            assert protocol['readiness'] == 'blocked_cross_domain_join_and_typed_id_audit'
    image_replay = next(v for v in plan['variants'] if v['model'] == 'InternVL'
                        and v['method'] == 'Replay' and v['benchmark'] == 'UCIT')
    assert image_replay['full_auxiliary'] == 'blocked_missing_full_replay_files'
    for number, dataset in enumerate(image_replay['datasets'], 1):
        assert dataset['full']['resolved_config']['replay_path'] is None
        assert dataset['full']['resolved_config']['router_path'] == f'${{UCIT_ANNOT_ROOT}}/router_json_20/task{number}.json'
        assert dataset['full_auxiliary']['router']['exists']
    image_router = next(v for v in plan['variants'] if v['model'] == 'InternVL'
                        and v['method'] == 'MR-LoRA' and v['benchmark'] == 'UCIT')
    assert image_router['full_auxiliary'] == 'blocked_empty_full_router_source'
    assert image_router['datasets'][0]['full_auxiliary']['router']['rows'] == 0
    assert image_router['datasets'][0]['full_auxiliary']['router']['status'] == 'blocked_empty_source'


def test_original_checkpoint_aliases_and_two_corrected_chains():
    assert configgen.portable('/opt/mcitlib-example/MCITlib/checkpoints/MLLM-DCL/InternVL/ModalPrompt/Task1') == (
        '${CKPT_ROOT}/MCITlib/MLLM-DCL/InternVL/ModalPrompt/Task1')
    assert configgen.portable('./MCITlib/Video-LLaVA/KeepLoRA/Task1_videollava_lora') == (
        '${CKPT_ROOT}/MCITlib/Video-LLaVA/KeepLoRA/Task1_videollava_lora')
    path = Path(__file__).resolve().parents[1] / 'configs/offline-plan.json'
    if not path.is_file():
        pytest.skip('local integration plan was not generated')
    plan = json.loads(path.read_text())
    variant = next(v for v in plan['variants'] if (v['model'], v['method'], v['benchmark']) ==
                   ('InternVL', 'ModalPrompt', 'UCIT'))
    stage1, stage2, stage3 = variant['stages'][:3]
    assert stage2['full_train']['output_dir'] != stage1['full_train']['output_dir']
    assert stage2['full_train']['output_dir'] == stage2['full_eval']['config']['model_path']
    assert stage2['full_train']['output_dir'] == stage3['full_train']['previous_model']
    assert stage2['path_corrections'][0]['before'] == stage1['full_train']['output_dir']
    assert stage2['path_corrections'][0]['after'] == stage2['full_train']['output_dir']
    variant = next(v for v in plan['variants'] if (v['model'], v['method'], v['benchmark']) ==
                   ('InternVL', 'KeepLoRA', 'UCIT'))
    pre = next(a for a in variant['auxiliary_configs_unmodified'] if a['role'] == 'train_pre'
               and a['path'].endswith('/task1.json'))
    assert pre['config']['model_path'] == plan['models']['InternVL']['legacy_model_config']['config']['model_name']
    assert pre['path_corrections'][0]['before'] == '${MODEL_ROOT}/internvl-v1.5-7b'


def test_unscoped_writes_and_unproduced_references_are_rejected():
    path = Path(__file__).resolve().parents[1] / 'configs/offline-plan.json'
    if not path.is_file():
        pytest.skip('local integration plan was not generated')
    plan = json.loads(path.read_text())
    variant = next(v for v in plan['variants'] if (v['model'], v['method'], v['benchmark']) ==
                   ('InternVL', 'ModalPrompt', 'UCIT'))
    scope = dict(plan['variables'])
    scope.update(RUN_ID='test', PREPARATIONS_ROOT=str(path.parents[1]))
    base = Path(variant['model']) / variant['method'] / variant['benchmark']
    for key in ('CKPT_ROOT', 'RESULTS_ROOT', 'LOG_ROOT'):
        scope[key] = str(Path(materializer.expand(scope[key], scope)) / base)
    model = plan['models']['InternVL']
    bad = copy.deepcopy(variant)
    bad['stages'][0]['sample_train_4gpu']['output_dir'] = '/tmp/outside-scope'
    with pytest.raises(ValueError, match='escapes its variant root'):
        materializer.audit_variant(bad, model, scope, 'sample', 4)
    bad = copy.deepcopy(variant)
    bad['stages'][1]['sample_train_4gpu']['output_dir'] = bad['stages'][0]['sample_train_4gpu']['output_dir']
    with pytest.raises(ValueError, match='duplicated'):
        materializer.audit_variant(bad, model, scope, 'sample', 4)
    bad = copy.deepcopy(variant)
    bad['stages'][1]['sample_train_4gpu']['previous_model'] = '${CKPT_ROOT}/missing-task'
    with pytest.raises(ValueError, match='no planned producer'):
        materializer.audit_variant(bad, model, scope, 'sample', 4)


def test_generation_uses_bundled_evidence_without_external_legacy_repositories(tmp_path):
    actual = configgen.find_repo()
    project = tmp_path / 'isolated' / 'mcitlib'
    project.mkdir(parents=True)
    for directory in ('configs', 'data', 'assets'):
        (project / directory).symlink_to(actual / directory, target_is_directory=True)
    assert not any((project.parent / model).exists() for model in configgen.MODELS)
    plan = configgen.generate(project)
    assert len(plan['variants']) == 47
    assert all(not item['legacy_method_exists'] for item in plan['variants'])
    video_mr = [v for v in plan['variants'] if v['benchmark'] == 'CL-VISTA' and v['method'] == 'MR-LoRA']
    assert len(video_mr) == 2
    assert all(v['router_protocol']['prompt_and_extractor_mapping'] == 'matched' for v in video_mr)
    assert all(v['router_protocol']['extractor'].startswith('reference_evidence/') for v in video_mr)
    image_reg = next(v for v in plan['variants'] if (v['model'], v['method'], v['benchmark']) ==
                     ('InternVL', 'RegLoRA', 'UCIT'))
    assert image_reg['merged_output_evidence']['first']['status'] == 'missing_producer'
    assert plan['reference_evidence_manifest']['sha256'] == configgen.REFERENCE_EVIDENCE_MANIFEST_SHA


def test_tampered_reference_evidence_or_manifest_refuses_generation(tmp_path, monkeypatch):
    source = configgen.REFERENCE_EVIDENCE_ROOT
    altered = tmp_path / 'altered'
    shutil.copytree(source, altered)
    evidence = altered / 'Video-LLaVA/MR-LoRA/scripts/extract_router_result.py.txt'
    evidence.write_bytes(evidence.read_bytes() + b'\n')
    monkeypatch.setattr(configgen, 'REFERENCE_EVIDENCE_ROOT', altered)
    with pytest.raises(ValueError, match='reference evidence SHA mismatch'):
        configgen.generate(tmp_path / 'no-legacy-project')
    altered_manifest = tmp_path / 'altered-manifest'
    shutil.copytree(source, altered_manifest)
    manifest = altered_manifest / 'manifest.json'
    manifest.write_bytes(manifest.read_bytes() + b'\n')
    monkeypatch.setattr(configgen, 'REFERENCE_EVIDENCE_ROOT', altered_manifest)
    with pytest.raises(ValueError, match='reference evidence manifest SHA mismatch'):
        configgen.generate(tmp_path / 'no-legacy-project')


def test_aux_replay_recovers_original_qa_not_router_label(tmp_path, monkeypatch):
    monkeypatch.setattr(auxiliary.base, 'TASKS', ('counting', 'space'))
    root, configs, sample = tmp_path / 'data', tmp_path / 'configs', tmp_path / 'sample'
    root.mkdir()
    history = []
    domains = []
    for task, letter in (('counting', 'F'), ('space', 'G')):
        name = task.capitalize()
        video = f'{name}/v.mp4'
        (root / name).mkdir()
        (root / video).write_bytes(b'video')
        train = [{'id': str(i) if i % 2 else i, 'video': video,
                  'conversations': [{'from': 'human', 'value': f'question-{i}'},
                                    {'from': 'gpt', 'value': f'original-{task}-{i}'}]} for i in range(100)]
        router = [{'id': row['id'], 'video': video, 'conversations': [{'from': 'human', 'value': 'router'},
                  {'from': 'gpt', 'value': letter}]} for row in train]
        prior_routers = [item for _, _, old_router in history for item in old_router]
        prior_train = [item for _, old_train, _ in history for item in old_train]
        save(root / task / 'train.json', train)
        save(root / task / 'router.json', router + prior_routers)
        save(root / task / 'replay.json', train + prior_train)
        cfg = {key: f'/your_data_path/CL-VISTA/{task}/{file}' for key, file in
               [('train_path', 'train.json'), ('router_path', 'router.json'), ('replay_path', 'replay.json')]}
        save(configs / f'{task}.json', cfg)
        save(sample / task / 'train.json', train[:3])
        domains.append({'task': task, 'source': {'train': {'sha256': sampler.digest((root / task / 'train.json').read_bytes())}},
                        'outputs': {'train.json': {'sha256': sampler.digest((sample / task / 'train.json').read_bytes())}},
                        'selected': {'train': [{'source_index': i} for i in range(3)]}})
        history.append((task, train, router))
    save(sample / 'manifest.json', {'task_order': ['counting', 'space'], 'domains': domains})
    output = tmp_path / 'auxiliary'
    manifest = auxiliary.build(configs, root, sample, output)
    info = json.loads(manifest.read_text())
    rows = json.loads((output / 'space/replay.json').read_text())
    assert len(rows) == 103 and info['domains'][1]['replay']['rows'] == 103
    assert all(row['conversations'][-1]['value'].startswith('original-') for row in rows)
    assert json.loads((output / 'space/router.json').read_text())[0]['conversations'][-1]['value'] == 'G'
    with pytest.raises(FileExistsError):
        auxiliary.build(configs, root, sample, output)
    corrupt = json.loads((root / 'space/replay.json').read_text())
    corrupt[-1]['conversations'][-1]['value'] = 'F'
    save(root / 'space/replay.json', corrupt)
    with pytest.raises(RuntimeError, match='prior replay does not equal recovered original QA'):
        auxiliary.build(configs, root, sample, tmp_path / 'rejected')
    assert not (tmp_path / 'rejected').exists()
