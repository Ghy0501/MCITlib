# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""CPU-only tests for explicit DCL inference and routing validation repairs."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from mcitlib.config.schema import PathSet
from mcitlib.dataflow import find_gaps, produced_paths
from mcitlib.eval.metrics import metrics_from_path
from mcitlib.methods._mrlora.router_results.dcl_validation import (
    EXPERTS,
    SCORERS,
    route_predictions,
    select_expert,
)
from mcitlib.paths import PathResolver
from mcitlib.runner import build_sequence_commands
from mcitlib.validation_batch import Queue, make_plan, verified_dcl_sample_count, worker_commands


class FixedRandom:
    def __init__(self, value: int):
        self.value = value
        self.calls = 0

    def randint(self, first: int, last: int) -> int:
        assert (first, last) == (0, 4)
        self.calls += 1
        return self.value


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def option(command, flag: str) -> str:
    return command.argv[command.argv.index(flag) + 1]


def test_original_router_selection_priority_and_random_fallback() -> None:
    rng = FixedRandom(0)
    assert select_expert("AB", rng) == (4, "A", None)
    assert select_expert(" A. ", rng) == (4, "A", None)
    assert select_expert("F", rng) == (0, "F", None)
    assert rng.calls == 0
    for value, expected in enumerate((4, 3, 1, 2, 0)):
        rng = FixedRandom(value)
        assert select_expert("a", rng) == (expected, None, "no_matching_label")
        assert rng.calls == 1
    rng = FixedRandom(4)
    assert select_expert("A, B", rng) == (0, None, "multiple_non_label_characters")
    assert rng.calls == 1


def test_route_then_original_scorers_build_complete_15_cell_matrix(tmp_path: Path, capsys) -> None:
    root = tmp_path / "DCL_router"
    for domain in EXPERTS:
        ids = list(range(4)) if domain == "RS" else [f"{domain}-{i}" for i in range(4)]
        annotations = [{"question_id": qid, "answer": "yes", "text": f"question {qid}",
                        "image": "AI2D_same_image.png"} for qid in ids]
        annotation_path = tmp_path / "data" / f"{domain}.json"
        annotation_path.parent.mkdir(exist_ok=True)
        annotation_path.write_text(json.dumps(annotations), encoding="utf-8")
        expert_paths = []
        for expert in EXPERTS:
            path = tmp_path / "experts" / domain / expert / "merge.jsonl"
            rows = [{"question_id": qid, "prompt": f"question {qid}", "text": "yes",
                     "model_id": expert} for qid in ids]
            if expert == "Fin":
                rows[1]["text"] = "no"
            if expert == "RS":
                rows[2]["text"] = "Unanswerable"
            write_jsonl(path, rows)
            expert_paths.append(path)
        domain_index = EXPERTS.index(domain)
        for stage in range(domain_index + 1, 6):
            router_path = tmp_path / "raw" / domain / f"task{stage}" / "merge.jsonl"
            write_jsonl(router_path, [{"question_id": qid, "text": label}
                                     for qid, label in zip(ids, ("E", "A", "F", "a"))])
            directory = root / "each_dataset" / domain / f"MR-LoRA-router-task{stage}"
            out = directory / "routed_predictions.jsonl"
            rng = FixedRandom(2)  # standalone candidate order index 2 is Med
            summary = route_predictions(annotation_path, router_path, expert_paths, out, expected_count=4, rng=rng)
            selected = [json.loads(line) for line in out.read_text().splitlines()]
            assert [row["question_id"] for row in selected] == ids
            assert [row["model_id"] for row in selected] == ["RS", "Fin", "RS", "Med"]
            assert selected[-1]["routing_validation"]["fallback_reason"] == "no_matching_label"
            assert selected[-1]["routing_validation"]["expert_file"] == str(expert_paths[1])
            assert summary["samples"] == 4 and summary["random_fallbacks"] == 1 and rng.calls == 1
            assert router_path.read_text().count("\n") == 4
            scorer = importlib.import_module(f"mcitlib.methods._mrlora.eval_runners.{SCORERS[domain]}")
            scorer.args = SimpleNamespace(output_dir=str(directory))
            scorer.eval_single(str(annotation_path), str(out))
            assert "Samples: 4" in (directory / "Result.text").read_text()
    result = metrics_from_path(root, tasks=list(EXPERTS))
    assert result["matrix"] == [[50.0] * length for length in range(1, 6)]
    assert result["metrics"]["maa"] == 50.0
    capsys.readouterr()


def test_routing_rejects_missing_duplicate_type_and_short_sample(tmp_path: Path) -> None:
    annotation = tmp_path / "annotation.json"
    annotation.write_text(json.dumps([{"question_id": 1}, {"question_id": 2}]))
    router = tmp_path / "router.jsonl"
    write_jsonl(router, [{"question_id": 1, "text": "A"}, {"question_id": 2, "text": "E"}])
    experts = []
    for name in EXPERTS:
        path = tmp_path / f"{name}.jsonl"
        write_jsonl(path, [{"question_id": i, "text": "x", "prompt": "q"} for i in (1, 2)])
        experts.append(path)
    out = tmp_path / "out" / "routed_predictions.jsonl"
    with pytest.raises(ValueError, match="expected 3 annotations, found 2"):
        route_predictions(annotation, router, experts, out, expected_count=3)
    assert not out.exists()
    experts[4].unlink()
    with pytest.raises(FileNotFoundError):
        route_predictions(annotation, router, experts, out, expected_count=2)
    write_jsonl(experts[4], [{"question_id": i, "text": "x", "prompt": "q"} for i in (1, 2)])
    write_jsonl(router, [{"question_id": 1, "text": "A"}, {"question_id": 3, "text": "E"}])
    with pytest.raises(ValueError, match="coverage differs"):
        route_predictions(annotation, router, experts, out, expected_count=2)
    write_jsonl(router, [{"question_id": 1, "text": "A"}, {"question_id": 2, "text": "E"}])
    write_jsonl(experts[1], [{"question_id": 1, "text": "x", "prompt": "q"}])
    with pytest.raises(ValueError, match="coverage differs"):
        route_predictions(annotation, router, experts, out, expected_count=2)
    write_jsonl(experts[1], [{"question_id": 1, "text": "x", "prompt": "q"}] * 2)
    with pytest.raises(ValueError, match="duplicate question_id"):
        route_predictions(annotation, router, experts, out, expected_count=2)
    write_jsonl(experts[1], [{"question_id": str(i), "text": "x", "prompt": "q"} for i in (1, 2)])
    with pytest.raises(ValueError, match="question_id type differs"):
        route_predictions(annotation, router, experts, out, expected_count=2)
    write_jsonl(experts[1], [{"question_id": i, "text": "x", "prompt": "q"} for i in (1, 2)])
    stale_hash = hashlib.sha256(annotation.read_bytes()).hexdigest()
    annotation.write_text(json.dumps([{"question_id": 1}, {"question_id": 3}]))
    with pytest.raises(ValueError, match="annotation hash differs"):
        route_predictions(annotation, router, experts, out, expected_count=2, expected_sha256=stale_hash)
    assert not out.exists()


def _paths(repo_root: Path) -> PathResolver:
    return PathResolver(PathSet(Path("/audit/model"), Path("/audit/data"), Path("/audit/ckpt"), repo_root,
                                Path("/audit/results"), repo_root / "logs"))


@pytest.mark.parametrize("four", [False, True])
def test_explicit_repairs_cover_mr_and_keep_plans_and_dependencies(repo_root: Path, four: bool) -> None:
    paths = _paths(repo_root)
    args = {"paths": paths, "validation_fixes": True, "expected_eval_samples": 1000,
            "gpus": "0,1,2,3" if four else "0,1"}
    if four:
        args["execution_profile"] = "four_gpu_validation"
    config = repo_root / "configs"
    ordinary = build_sequence_commands(config, "llava", "mrlora", "mllm_dcl", **args)
    router = build_sequence_commands(config, "llava", "mrlora", "mllm_dcl", sequence="router", **args)
    keep = build_sequence_commands(config, "llava", "keeplora", "mllm_dcl", only_eval=True, **args)
    expected_chunks = 4 if four else 2
    expert = [item for item in ordinary if item.metadata.get("validation_fix") == "mrlora_dcl_expert_inference"]
    joins = [item for item in router if item.metadata.get("validation_fix") == "mrlora_dcl_router_selection"]
    scores = [item for item in router if item.metadata.get("validation_fix") == "mrlora_dcl_router_scoring"]
    med = [item for item in keep if item.metadata.get("validation_fix") == "keeplora_dcl_stage2_med"]
    assert len(expert) == 25 * expected_chunks and len(joins) == len(scores) == 15
    assert len(med) == expected_chunks + 1
    keep_cells = {(item.task_id, Path(option(item, "--output-dir")).parent.name)
                  for item in keep if item.kind == "eval" and "--output-dir" in item.argv}
    assert keep_cells == {(stage, domain) for stage in range(1, 6) for domain in EXPERTS[:stage]}
    expert_groups: dict[tuple[int, str], list] = {}
    answer_files = set()
    for item in expert:
        domain = Path(option(item, "--question-file")).parent.name
        key = (item.task_id, domain)
        expert_groups.setdefault(key, []).append(item)
        answer = option(item, "--answers-file")
        assert answer not in answer_files
        answer_files.add(answer)
        assert f"/{domain}/MR-LoRA-task{item.task_id}/{expected_chunks}_" in answer
        assert option(item, "--model-path").endswith(f"/Task{item.task_id}_llava_lora")
        assert option(item, "--question-file") == f"/audit/data/MLLM-CL/{domain}/test.json"
    assert set(expert_groups) == {(stage, domain) for stage in range(1, 6) for domain in EXPERTS}
    for items in expert_groups.values():
        assert {option(item, "--chunk-idx") for item in items} == {str(i) for i in range(expected_chunks)}
    expert_merges = [item for item in ordinary if item.metadata.get("merge_chunks")]
    assert len(expert_merges) == 25
    assert len({option(item, "--output") for item in expert_merges}) == 25
    for item in expert_merges:
        directory = Path(option(item, "--output")).parent
        expected_files = [str(directory / f"{expected_chunks}_{index}.jsonl") for index in range(expected_chunks)]
        assert item.argv[item.argv.index("--inputs") + 1:] == expected_files
    scored = set()
    for item in scores:
        output_dir = Path(option(item, "--output-dir"))
        domain = output_dir.parent.name
        key = (item.task_id, domain)
        assert key not in scored
        scored.add(key)
        assert option(item, "--annotation-file") == f"/audit/data/MLLM-CL/{domain}/test.json"
        assert option(item, "--result-file") == str(output_dir / "routed_predictions.jsonl")
    assert scored == {(stage, domain) for stage in range(1, 6) for domain in EXPERTS[:stage]}
    router_merges = [item for item in router if item.metadata.get("merge_chunks")]
    assert len(router_merges) == len({option(item, "--output") for item in router_merges}) == 15
    med_inference = [item for item in med if "--answers-file" in item.argv]
    assert len({option(item, "--answers-file") for item in med_inference}) == expected_chunks
    assert {option(item, "--chunk-idx") for item in med_inference} == {str(i) for i in range(expected_chunks)}
    assert all(option(item, "--model-path").endswith("/Task2_llava_lora_merged") for item in med_inference)
    assert all(option(item, "--question-file") == "/audit/data/MLLM-CL/Med/test.json" for item in med_inference)
    assert all(option(item, "--expected-count") == "1000" for item in joins)
    assert all("--discover" not in item.argv for item in ordinary + router)
    roots = paths.external_roots("mllm_dcl")
    internal = (str(paths.paths.ckpt_root), str(paths.results_root))
    assert not find_gaps(ordinary, roots, internal_roots=internal)
    produced = produced_paths(ordinary)
    assert not find_gaps(router, roots, tuple(produced), internal)
    join = joins[0]
    router_merge = option(join, "--result-file")
    expert_merges = join.argv[join.argv.index("--inputs") + 1:]
    assert len(expert_merges) == 5
    for missing in expert_merges:
        available = (produced - {missing}) | {router_merge}
        assert any(gap.flag == "--inputs" and gap.path == missing for gap in find_gaps([join], roots, tuple(available), internal))
    assert any(gap.flag == "--result-file" and gap.path == router_merge
               for gap in find_gaps([join], roots, tuple(produced), internal))
    for task in range(1, 6):
        standalone = build_sequence_commands(config, "llava", "mrlora", "mllm_dcl", only_eval=True,
                                             sequence="router", task_id=task, **args)
        assert [item.as_dict() for item in standalone] == [item.as_dict() for item in router if item.task_id == task and item.kind != "train"]
    old = build_sequence_commands(config, "llava", "mrlora", "mllm_dcl", only_eval=True)
    assert sum("--discover" in item.argv for item in old) == 25
    old_keep = build_sequence_commands(config, "llava", "keeplora", "mllm_dcl", task_id=2, only_eval=True)
    assert not any("/Med/KeepLoRA-task2/" in token for item in old_keep for token in item.argv)


def test_queue_binds_count_and_metrics_path_only_to_verified_sample_manifest(tmp_path: Path,
                                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / "configs/validation_3k_1k/manifest.json"
    manifest.parent.mkdir(parents=True)
    tasks = []
    mappings = []
    for domain in EXPERTS:
        sample = tmp_path / "data/validation_3k_1k/mllm_dcl" / domain / "test.json"
        sample.parent.mkdir(parents=True)
        sample.write_text(json.dumps([{"question_id": 1}, {"question_id": 2}]))
        source = f"${{data_root}}/MLLM-CL/{domain}/test.json"
        target = f"${{repo_root}}/data/validation_3k_1k/mllm_dcl/{domain}/test.json"
        tasks.append({"benchmark": "mllm_dcl", "name": domain, "splits": {"test": {
            "source": source, "output": target, "sample_count": 2,
            "output_sha256": hashlib.sha256(sample.read_bytes()).hexdigest()}}})
        mappings.append({"from": source, "to": target})
    manifest.write_text(json.dumps({"tasks": tasks}))
    profile = tmp_path / "paths.yaml"
    profile.write_text(yaml.safe_dump({"data_root": "data", "ckpt_root": "ckpts", "results_root": "results",
                                       "logs_root": "logs", "path_mappings": {"mllm_dcl": mappings}}))
    resolver = PathResolver.from_file(profile, repo_root=tmp_path)
    assert verified_dcl_sample_count(tmp_path, resolver) == 2
    plan = make_plan(tmp_path, profile, "dcl-test", None, None, "0,1,2,3", 29873, 0,
                     execution_profile="four_gpu_validation", job_select=["mrlora/mllm_dcl"])
    job = plan["jobs"][0]
    assert job.expected_eval_samples == 2
    assert set(job.expected_eval_hashes) == set(EXPERTS)
    assert all(argv[argv.index("--expected-eval-samples") + 1] == "2" for _, argv in job.doctors)
    assert all(json.loads(argv[argv.index("--expected-eval-hashes") + 1]) == job.expected_eval_hashes for _, argv in job.doctors)
    assert job.worker[job.worker.index("--expected-eval-samples") + 1] == "2"
    assert json.loads(job.worker[job.worker.index("--expected-eval-hashes") + 1]) == job.expected_eval_hashes
    assert job.metrics[job.metrics.index("--run-dir") + 1] == str(job.metrics_result_dir)
    assert str(job.metrics_result_dir).endswith("/mrlora/mllm_dcl/mrlora/DCL_router")
    queue = Queue(plan)
    monkeypatch.setattr(queue, "run_job", lambda item: queue.state["jobs"][item.key].update(status="completed"))
    assert queue.execute() == 0
    state = json.loads(queue.state_path.read_text())["jobs"]["mrlora/mllm_dcl"]
    assert state["metrics_result_dir"] == str(job.metrics_result_dir)
    assert state["expected_eval_samples"] == 2 and state["expected_eval_hashes"] == job.expected_eval_hashes
    import mcitlib.runner as runner
    import mcitlib.validation_batch as batch

    monkeypatch.setattr(batch, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(batch, "worker_commands", lambda *args, **kwargs: [])
    monkeypatch.setattr(runner, "execute_commands", lambda *args, **kwargs: [])
    args = SimpleNamespace(paths=job.paths_file, run_dir=job.directory, method="mrlora", benchmark="mllm_dcl",
                           gpus="0,1,2,3", master_port=29873, execution_profile="four_gpu_validation",
                           expected_eval_samples=2, expected_eval_hashes=job.expected_eval_hashes, min_free_gib=0)
    assert batch.run_worker(args) == 0
    commands = json.loads((job.directory / "commands.json").read_text())
    for key in ("metrics_result_dir", "expected_eval_samples", "expected_eval_hashes"):
        assert commands[key] == state[key]
    partial = yaml.safe_load(profile.read_text())
    partial["path_mappings"]["mllm_dcl"].pop()
    profile.write_text(yaml.safe_dump(partial))
    with pytest.raises(ValueError, match="only part"):
        verified_dcl_sample_count(tmp_path, PathResolver.from_file(profile, repo_root=tmp_path))
    sample = tmp_path / "data/validation_3k_1k/mllm_dcl/RS/test.json"
    sample.write_text("[]")
    with pytest.raises(ValueError, match="hash differs"):
        verified_dcl_sample_count(tmp_path, resolver)


def test_sample_profile_without_manifest_is_rejected(tmp_path: Path) -> None:
    profile = tmp_path / "paths.yaml"
    profile.write_text(yaml.safe_dump({"data_root": "data", "path_mappings": {"common": [
        {"from": "${data_root}/source", "to": "${repo_root}/data/validation_3k_1k/target"}]}}))
    with pytest.raises(FileNotFoundError, match="requires its manifest"):
        verified_dcl_sample_count(tmp_path, PathResolver.from_file(profile, repo_root=tmp_path))


def test_worker_and_doctor_share_expected_count(repo_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import mcitlib.doctor as doctor

    paths = _paths(repo_root)
    hashes = {domain: str(index) * 64 for index, domain in enumerate(EXPERTS)}
    commands = worker_commands(repo_root, paths, "mrlora", "mllm_dcl", "0,1,2,3", 29873,
                               "four_gpu_validation", expected_eval_samples=1000, expected_eval_hashes=hashes)
    joins = [item for item in commands if item.metadata.get("validation_fix") == "mrlora_dcl_router_selection"]
    assert len(joins) == 15 and all(option(item, "--expected-count") == "1000" for item in joins)
    for item in joins:
        domain = Path(option(item, "--annotation-file")).parent.name
        assert option(item, "--annotation-sha256") == hashes[domain]
    monkeypatch.setattr(PathResolver, "from_file", classmethod(lambda cls, *args, **kwargs: paths))
    for name in ("check_environment", "check_vendored_imports", "check_inputs", "check_prerequisites"):
        monkeypatch.setattr(doctor, name, lambda *args: None)
    monkeypatch.setattr(doctor, "_keeplora_config_inputs", lambda report, values: (values, ()))
    captured = []
    monkeypatch.setattr(doctor, "check_dataflow", lambda report, values, *args: captured.extend(values))
    doctor.run_doctor("mrlora", "mllm_dcl", None, repo_root, skip_gpu=True, skip_entrypoints=True,
                      sequence="router", validation_fixes=True, execution_profile="four_gpu_validation",
                      gpus="0,1,2,3", master_port=29873, expected_eval_samples=1000, expected_eval_hashes=hashes)
    assert [item.as_dict() for item in captured] == [item.as_dict() for item in commands if item.sequence == "router"]


def test_cpu_doctor_masks_gpu_only_for_entrypoint_imports(repo_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import mcitlib.doctor as doctor

    monkeypatch.setattr(doctor, "check_environment", lambda *args: None)
    monkeypatch.setattr(doctor, "check_vendored_imports", lambda *args: None)
    monkeypatch.setattr(doctor, "check_inputs", lambda *args: None)
    monkeypatch.setattr(doctor, "check_prerequisites", lambda *args: None)
    monkeypatch.setattr(doctor, "_keeplora_config_inputs", lambda report, commands: (commands, ()))
    checked: dict[str, list] = {}
    monkeypatch.setattr(doctor, "check_dataflow", lambda report, commands, *args: checked.update(plan=commands))
    monkeypatch.setattr(doctor, "check_entry_points", lambda report, commands, *args: checked.update(entry=commands))
    doctor.run_doctor("mrlora", "mllm_dcl", None, repo_root, skip_gpu=True, skip_entrypoints=False,
                      sequence="router", validation_fixes=True, execution_profile="four_gpu_validation",
                      gpus="0,1,2,3", expected_eval_samples=1000)
    assert all(command.env["CUDA_VISIBLE_DEVICES"] == "" for command in checked["entry"])
    assert any(command.env.get("CUDA_VISIBLE_DEVICES") == "0,1,2,3" for command in checked["plan"])
