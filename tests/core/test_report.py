# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""CPU-only validation queue reporting regression tests."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mcitlib.eval.metrics import continual_metrics
from mcitlib.registry import get_benchmark
from mcitlib.reporting import build_report, generate_report

TASKS = list(get_benchmark("ucit").tasks)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _json(path: Path, value: object) -> Path:
    return _write(path, json.dumps(value))


def _fixture(tmp_path: Path, status: str = "running") -> tuple[Path, Path, dict]:
    project = tmp_path / "project"
    run = tmp_path / "run"
    results = tmp_path / "results"
    config = _write(project / "configs" / "paths.yaml", "input: ${repo_root}/data/sample-set/ucit/ImageNet-R/train.json\n")
    sample_dir = project / "configs" / "sample-set"
    recipe = _json(sample_dir / "recipe.json", {"version": "fixture", "seed": 5})
    _json(sample_dir / "manifest.json", {"version": "fixture", "seed": 5,
        "recipe_sha256": hashlib.sha256(recipe.read_bytes()).hexdigest(),
        "tasks": [{"benchmark": "ucit", "name": "ImageNet-R", "splits": {
            "train": {"sample_count": 3, "source": "${data_root}/UCIT/ImageNet-R/train.json", "output": "${repo_root}/data/sample-set/ucit/ImageNet-R/train.json"},
            "test": {"sample_count": 2, "source": "${data_root}/UCIT/ImageNet-R/test.json", "output": "${repo_root}/data/sample-set/ucit/ImageNet-R/test.json"}}}]})
    job_dir = run / "jobs" / "lora_ft" / "ucit"
    effective = _write(job_dir / "paths.effective.yaml", """path_mappings:
  ucit:
    - from: ${data_root}/UCIT/ImageNet-R/train.json
      to: ${repo_root}/data/sample-set/ucit/ImageNet-R/train.json
    - from: ${data_root}/UCIT/ImageNet-R/test.json
      to: ${repo_root}/data/sample-set/ucit/ImageNet-R/test.json
""")
    job = {"method": "lora_ft", "benchmark": "ucit", "status": status, "directory": str(job_dir),
           "effective_paths": str(effective), "environment": "llava", "python": "/example/python",
           "started_at": "2026-01-01T00:00:00+00:00"}
    state = {"run_id": "run", "status": "running", "current": "lora_ft/ucit" if status == "running" else None,
             "project": str(project), "source_paths": str(config), "roots": {"results_root": str(results),
             "logs_root": str(run), "ckpt_root": str(tmp_path / "ckpts")},
             "summary": {status: 1}, "jobs": {"lora_ft/ucit": job}}
    _json(run / "state.json", state)
    return run, results / "lora_ft" / "ucit" / "lora_ft" / "UCIT", state


def _result(root: Path, stage: int, task: str, text: str, label: str = "LoRA") -> Path:
    return _write(root / "each_dataset" / task / f"{label}-task{stage}" / "Result.text", text)


def _complete(root: Path) -> list[list[float]]:
    matrix = []
    for row in range(len(TASKS)):
        values = []
        for col in range(row + 1):
            value = float(80 - row * 2 + col)
            values.append(value)
            _result(root, row + 1, TASKS[col], f"Samples: 2\nAccuracy: {value}%\n")
        matrix.append(values)
    return matrix


def test_running_partial_results_samples_and_output_formats(tmp_path: Path) -> None:
    run, result, state = _fixture(tmp_path)
    state["jobs"]["lora_ft/ucit"].update(phase="sequence", task=2, step={"index": 3, "status": "started", "task": 2})
    _json(run / "state.json", state)
    event = {"index": 3, "status": "started", "task": 2, "kind": "eval"}
    _write(run / "jobs/lora_ft/ucit/events.jsonl", json.dumps(event) + "\n" + '{"status":')
    _result(result, 1, TASKS[0], "Samples: 2\nAccuracy: 80.6%\n")
    _result(result, 2, TASKS[0], "Samples: 2\nAccuracy: 79.0%\n")
    report = build_report(run)
    job = report["jobs"][0]
    assert report["summary"] == {"running": 1}
    assert job["matrix"][:2] == [[80.6], [79.0, None]]
    assert job["observed_samples"][:2] == [[2], [2, None]]
    assert all(value is None for value in job["metrics"].values())
    assert job["planned_samples"][TASKS[0]]["train"]["sample_count"] == 3
    assert job["planned_samples"][TASKS[1]] is None
    assert any("incomplete final event" in w for w in job["warnings"])
    assert report["manifest"]["sample"]["recipe_hash_matches"] is True
    assert report["manifest"]["state_file"]["sha256_at_start"] == hashlib.sha256((run / "state.json").read_bytes()).hexdigest()
    outputs = generate_report(run, tmp_path / "out")
    md = Path(outputs["markdown"]).read_text()
    assert "| 1 | 80.60 |" in md and "N/A" in md and "not establish formal reproduction" in md
    with Path(outputs["csv"]).open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["mft"] == "N/A"
    assert row["available_cells"] == "2" and row["required_cells"] == "21"
    with Path(outputs["cells_csv"]).open(newline="") as handle:
        cells = list(csv.DictReader(handle))
    assert len(cells) == 21
    assert cells[0]["score"] == "80.6" and cells[0]["samples"] == "2"
    assert cells[2]["score"] == "N/A" and cells[2]["availability"] == "unavailable"


def test_complete_matrix_uses_original_metrics_and_requires_every_cell(tmp_path: Path) -> None:
    run, result, state = _fixture(tmp_path, "completed")
    matrix = _complete(result)
    job = build_report(run)["jobs"][0]
    assert job["matrix_complete"]
    assert job["matrix"] == matrix
    assert job["metrics"] == continual_metrics(matrix)
    assert job["metrics"]["bwt"] < 0
    _result(result, 6, TASKS[2], "Accuracy: NaN\n")
    job = build_report(run)["jobs"][0]
    assert job["matrix"][5][2] is None and not job["matrix_complete"]
    assert all(value is None for value in job["metrics"].values())
    assert any("completed status" in w for w in job["warnings"])


def test_completed_metrics_file_and_conflicting_results(tmp_path: Path) -> None:
    run, result, _ = _fixture(tmp_path, "completed")
    matrix = _complete(result)
    metrics_path = run / "jobs/lora_ft/ucit/metrics.json"
    _json(metrics_path, {"matrix": matrix, "tasks": TASKS, "metrics": continual_metrics(matrix)})
    assert build_report(run)["jobs"][0]["metrics"] == continual_metrics(matrix)
    matrix[1][0] = -10
    _json(metrics_path, {"matrix": matrix, "tasks": TASKS})
    job = build_report(run)["jobs"][0]
    assert job["matrix"][1][0] is None
    assert all(value is None for value in job["metrics"].values())
    assert any("disagrees" in warning for warning in job["warnings"])
    _result(result, 1, TASKS[0], "Accuracy: 50%\n", label="other")
    job = build_report(run)["jobs"][0]
    assert job["matrix"][0][0] is None
    assert any("ambiguous matrix cell" in warning for warning in job["warnings"])


def test_dcl_five_stage_results_and_contradictory_doctor(tmp_path: Path) -> None:
    run, _, state = _fixture(tmp_path, "completed")
    job = state["jobs"].pop("lora_ft/ucit")
    job.update(benchmark="mllm_dcl", directory=str(run / "jobs/lora_ft/mllm_dcl"))
    state["jobs"]["lora_ft/mllm_dcl"] = job
    _json(run / "state.json", state)
    tasks = list(get_benchmark("mllm_dcl").tasks)
    result = tmp_path / "results/lora_ft/mllm_dcl/lora_ft/DCL"
    matrix = []
    for row, _ in enumerate(tasks):
        values = []
        for column in range(row + 1):
            value = float(row * 10 + column)
            values.append(value)
            _result(result, row + 1, tasks[column], f"Samples: 2\nAccuracy: {value}%\n")
        matrix.append(values)
    job = build_report(run)["jobs"][0]
    assert job["tasks"] == tasks and job["matrix"] == matrix
    assert job["metrics"] == continual_metrics(matrix)
    assert job["matrix"][0][0] == 0.0  # A measured zero stays zero.
    _json(run / "jobs/lora_ft/mllm_dcl/doctor-router.json", {"checks": [{"status": "FAIL", "name": "router", "detail": "missing"}]})
    job = build_report(run)["jobs"][0]
    assert job["status"] == "completed" and job["matrix_complete"]
    assert all(value is None for value in job["metrics"].values())
    assert any("failed doctor" in warning for warning in job["warnings"])


def _mrlora_dcl_results(tmp_path: Path) -> tuple[Path, Path, Path, dict]:
    run, _, state = _fixture(tmp_path, "completed")
    job = state["jobs"].pop("lora_ft/ucit")
    job.update(method="mrlora", benchmark="mllm_dcl", directory=str(run / "jobs/mrlora/mllm_dcl"))
    state["jobs"]["mrlora/mllm_dcl"] = job
    _json(run / "state.json", state)
    base = tmp_path / "results/mrlora/mllm_dcl/mrlora"
    return run, base / "DCL", base / "DCL_router", state


def _dcl_matrix(root: Path, value: float, label: str) -> list[list[float]]:
    tasks = list(get_benchmark("mllm_dcl").tasks)
    matrix = []
    for row in range(len(tasks)):
        values = []
        for column in range(row + 1):
            cell = value + row + column
            values.append(cell)
            _result(root, row + 1, tasks[column], f"Samples: 2\nAccuracy: {cell}%\n", label)
        matrix.append(values)
    return matrix


def test_mrlora_dcl_explicit_router_result_selection(tmp_path: Path) -> None:
    run, expert, router, state = _mrlora_dcl_results(tmp_path)
    _dcl_matrix(expert, 10, "MR-LoRA")
    expected = _dcl_matrix(router, 70, "MR-LoRA-router")
    hashes = {"RS": "a" * 64, "Med": "b" * 64}
    state["jobs"]["mrlora/mllm_dcl"].update(metrics_result_dir=str(router), expected_eval_samples=1000,
                                             expected_eval_hashes=hashes)
    _json(run / "state.json", state)
    _json(run / "jobs/mrlora/mllm_dcl/commands.json", {"metrics_result_dir": str(router), "commands": [],
          "expected_eval_samples": 1000, "expected_eval_hashes": hashes})
    report = build_report(run)
    job = report["jobs"][0]
    assert job["result_source"] == "router" and job["results_dir"] == str(router)
    assert job["expert_results_dir"] == str(expert)
    sample = report["manifest"]["jobs"]["mrlora/mllm_dcl"]["sample_expectation"]
    assert sample["state_expected_eval_hashes"] == sample["commands_expected_eval_hashes"] == hashes
    assert sample["state_expected_eval_samples"] == sample["commands_expected_eval_samples"] == 1000
    assert job["matrix"] == expected and job["metrics"] == continual_metrics(expected)
    assert all(str(router) in paths[0] for row in job["matrix_sources"] for paths in row)
    metrics_file = run / "jobs/mrlora/mllm_dcl/metrics.json"
    _json(metrics_file, {"source": str(router), "tasks": list(get_benchmark("mllm_dcl").tasks), "matrix": expected})
    assert build_report(run)["jobs"][0]["metrics"] == continual_metrics(expected)
    outputs = generate_report(run, tmp_path / "out")
    markdown = Path(outputs["markdown"]).read_text()
    assert "explicitly recorded router results" in markdown and "not a strictly online protocol" in markdown
    with Path(outputs["csv"]).open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["result_source"] == "router" and row["results_dir"] == str(router)


def test_mrlora_dcl_unmarked_legacy_does_not_claim_router(tmp_path: Path) -> None:
    run, expert, router, _ = _mrlora_dcl_results(tmp_path)
    expected = _dcl_matrix(expert, 10, "MR-LoRA")
    _dcl_matrix(router, 70, "MR-LoRA-router")
    report = build_report(run)
    job = report["jobs"][0]
    assert job["result_source"] == "legacy" and job["matrix"] == expected
    assert job["results_dir"] == str(expert) and job["expert_results_dir"] is None
    outputs = generate_report(run, tmp_path / "out")
    assert "does not identify a separate router directory" in Path(outputs["markdown"]).read_text()


def test_mrlora_dcl_missing_router_never_uses_expert_or_stale_metrics(tmp_path: Path) -> None:
    run, expert, router, state = _mrlora_dcl_results(tmp_path)
    expert_matrix = _dcl_matrix(expert, 10, "MR-LoRA")
    _dcl_matrix(router, 70, "MR-LoRA-router")
    (router / "each_dataset/RS/MR-LoRA-router-task2/Result.text").unlink()
    state["jobs"]["mrlora/mllm_dcl"]["metrics_result_dir"] = str(router)
    _json(run / "state.json", state)
    _json(run / "jobs/mrlora/mllm_dcl/metrics.json", {"source": str(expert),
          "tasks": list(get_benchmark("mllm_dcl").tasks), "matrix": expert_matrix})
    job = build_report(run)["jobs"][0]
    assert job["result_source"] == "router" and job["matrix"][1][0] is None
    assert not job["matrix_complete"] and all(value is None for value in job["metrics"].values())
    assert any("source does not match" in warning for warning in job["warnings"])


def test_mrlora_dcl_command_only_and_conflicting_markers(tmp_path: Path) -> None:
    run, expert, router, state = _mrlora_dcl_results(tmp_path)
    _dcl_matrix(expert, 10, "MR-LoRA")
    expected = _dcl_matrix(router, 70, "MR-LoRA-router")
    commands_file = run / "jobs/mrlora/mllm_dcl/commands.json"
    _json(commands_file, {"metrics_result_dir": str(router), "commands": []})
    job = build_report(run)["jobs"][0]
    assert job["result_source"] == "router" and job["matrix"] == expected
    state["jobs"]["mrlora/mllm_dcl"]["metrics_result_dir"] = str(expert)
    _json(run / "state.json", state)
    job = build_report(run)["jobs"][0]
    assert job["result_source"] == "invalid" and job["results_dir"] is None
    assert all(cell is None for row in job["matrix"] for cell in row)
    assert all(value is None for value in job["metrics"].values())
    assert any("invalid or conflicting" in warning for warning in job["warnings"])


def test_mrlora_dcl_conflicting_sample_expectations_suppress_metrics(tmp_path: Path) -> None:
    run, expert, router, state = _mrlora_dcl_results(tmp_path)
    _dcl_matrix(expert, 10, "MR-LoRA")
    _dcl_matrix(router, 70, "MR-LoRA-router")
    state["jobs"]["mrlora/mllm_dcl"].update(metrics_result_dir=str(router), expected_eval_samples=1000,
                                             expected_eval_hashes={"RS": "state-hash"})
    _json(run / "state.json", state)
    _json(run / "jobs/mrlora/mllm_dcl/commands.json", {"metrics_result_dir": str(router), "commands": [],
          "expected_eval_samples": 900, "expected_eval_hashes": {"RS": "other-hash"}})
    job = build_report(run)["jobs"][0]
    assert job["matrix_complete"] and all(value is None for value in job["metrics"].values())
    assert len([warning for warning in job["warnings"] if "differs between state and commands" in warning]) == 2


def test_mrlora_ucit_legacy_is_not_routed_and_explicit_router_never_falls_back(tmp_path: Path) -> None:
    run, _, state = _fixture(tmp_path, "completed")
    raw = state["jobs"].pop("lora_ft/ucit")
    raw.update(method="mrlora", directory=str(run / "jobs/mrlora/ucit"))
    state["jobs"]["mrlora/ucit"] = raw
    _json(run / "state.json", state)
    root = tmp_path / "results/mrlora/ucit/mrlora"
    expert = root / "UCIT"
    router = root / "UCIT_router"
    _complete(expert)
    job = build_report(run)["jobs"][0]
    assert job["result_source"] == "legacy" and job["matrix_complete"]
    assert all(value is None for value in job["metrics"].values())
    assert any("do not establish routed evaluation" in warning for warning in job["warnings"])
    for row in range(len(TASKS)):
        for column in range(row + 1):
            _result(router, row + 1, TASKS[column], f"Samples: 2\nAccuracy: {90 - row - column}%\n", "MR-LoRA-router")
    raw["metrics_result_dir"] = str(router)
    _json(run / "state.json", state)
    job = build_report(run)["jobs"][0]
    assert job["result_source"] == "router" and job["matrix"][0][0] == 90
    assert job["expert_results_dir"] == str(expert) and job["metrics"]["mft"] is not None
    (router / "each_dataset/ImageNet-R/MR-LoRA-router-task2/Result.text").unlink()
    job = build_report(run)["jobs"][0]
    assert job["matrix"][1][0] is None and all(value is None for value in job["metrics"].values())


def test_status_disagreement_and_failed_router_diagnosis(tmp_path: Path) -> None:
    run, result, state = _fixture(tmp_path, "pending")
    _complete(result)
    state["status"] = "completed"
    state["summary"] = {"completed": 1}
    state["jobs"]["lora_ft/ucit"]["started_at"] = None
    _json(run / "state.json", state)
    _write(run / "jobs/lora_ft/ucit/events.jsonl", '{"status":"failed","error":"stale"}\n')
    job_report = build_report(run)
    job = job_report["jobs"][0]
    assert job["status"] == "pending" and job["matrix"][0][0] is None
    assert all(value is None for value in job["metrics"].values())
    assert any("suppressed" in warning for warning in job["warnings"])
    assert any("summary" in warning for warning in job_report["warnings"])
    assert any("completed queue" in warning for warning in job_report["warnings"])
    state["jobs"]["lora_ft/ucit"].update(status="blocked", error="missing | config, please\ninspect")
    state["summary"] = {"blocked": 1}
    state["status"] = "completed_with_failures"
    _json(run / "state.json", state)
    _json(run / "jobs/lora_ft/ucit/doctor-router.json", {"failed": True, "checks": [
        {"status": "FAIL", "name": f"input-{i}", "detail": "missing"} for i in range(25)]})
    outputs = generate_report(run, tmp_path / "out")
    job = json.loads(Path(outputs["json"]).read_text())["jobs"][0]
    assert len(job["doctor_failures"]) == 25
    assert job["failure_log"].endswith("doctor-router.json")
    assert job["matrix"][0][0] is None
    assert "missing \\| config, please inspect" in Path(outputs["markdown"]).read_text()
    with Path(outputs["csv"]).open(newline="") as handle:
        assert next(csv.DictReader(handle))["error"] == "missing | config, please\ninspect"


@pytest.mark.parametrize("status", ["running", "failed", "stopped", "blocked_disk", "not_run"])
def test_aggregate_requires_completed_status(tmp_path: Path, status: str) -> None:
    run, result, state = _fixture(tmp_path, status)
    _complete(result)
    job = build_report(run)["jobs"][0]
    assert job["status"] == status
    assert job["matrix_complete"] is (status not in {"not_run", "blocked_disk"})
    assert all(value is None for value in job["metrics"].values())


def test_sample_counts_require_effective_mapping(tmp_path: Path) -> None:
    run, _, _ = _fixture(tmp_path)
    _write(run / "jobs/lora_ft/ucit/paths.effective.yaml", "path_mappings: {}\n")
    report = build_report(run)
    sample = report["jobs"][0]["planned_samples"][TASKS[0]]
    assert sample["train"] is None and sample["test"] is None
    assert report["manifest"]["sample_counts"]["ucit"][TASKS[0]]["train"]["sample_count"] == 3
    assert any("do not all match" in w for w in report["jobs"][0]["warnings"])
    _write(run / "jobs/lora_ft/ucit/paths.effective.yaml", """path_mappings:
  mllm_dcl:
    - from: ${data_root}/UCIT/ImageNet-R/test.json
      to: ${repo_root}/data/sample-set/ucit/ImageNet-R/test.json
  ucit:
    - from: ${data_root}/different/train.json
      to: ${repo_root}/data/sample-set/ucit/ImageNet-R/train.json
""")
    sample = build_report(run)["jobs"][0]["planned_samples"][TASKS[0]]
    assert sample["train"] is None and sample["test"] is None


def test_changed_state_is_reported_and_first_state_retained(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, _, _ = _fixture(tmp_path)
    original = Path.read_bytes
    reads = 0
    def read_bytes(path: Path) -> bytes:
        nonlocal reads
        value = original(path)
        if path == run / "state.json":
            reads += 1
            if reads == 2:
                return value + b" "
        return value
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    report = build_report(run)
    assert report["manifest"]["state_file"]["changed_during_collection"] is True
    assert report["jobs"][0]["status"] == "running"
    assert any("changed during collection" in w for w in report["warnings"])


def test_results_beyond_state_task_are_suppressed(tmp_path: Path) -> None:
    run, result, state = _fixture(tmp_path)
    state["jobs"]["lora_ft/ucit"]["task"] = 1
    _json(run / "state.json", state)
    _result(result, 1, TASKS[0], "Accuracy: 70%\n")
    _result(result, 2, TASKS[0], "Accuracy: 72%\n")
    job = build_report(run)["jobs"][0]
    assert job["matrix"][:2] == [[70.0], [None, None]]
    assert any("newer than the state task" in warning for warning in job["warnings"])


def test_recorded_resources_and_run_lineage(tmp_path: Path) -> None:
    run, _, state = _fixture(tmp_path)
    previous = tmp_path / "previous-run"
    _json(previous / "state.json", {"status": "completed", "jobs": {"other/ucit": {"status": "completed"}}})
    receipt = tmp_path / "transition.json"
    _json(receipt, {"schema_version": 1, "status": "applied", "previous_run": str(previous),
                    "old_run_id": "old", "new_run_id": "run", "execution_profile": "four_gpu_validation"})
    state.update(execution_profile="four_gpu_validation", previous_run=str(previous),
                 transition_receipt=str(receipt), related_runs=["another-run"])
    _json(run / "state.json", state)
    metadata = {"execution_profile": "four_gpu_validation"}
    commands = [{"kind": "train", "task_id": 1, "sequence": "train", "metadata": {**metadata, "effective_batch": 64},
                 "argv": ["deepspeed", "--include", "localhost:0,1,2,3", "--per_device_train_batch_size", "8",
                          "--gradient_accumulation_steps", "2"]}]
    commands += [{"kind": "eval", "task_id": 1, "sequence": "train", "metadata": {**metadata, "parallel_group": "dataset"},
                  "argv": ["python", "--num-chunks", "4", "--chunk-idx", str(i)],
                  "env": {"CUDA_VISIBLE_DEVICES": str(3 - i)}} for i in range(4)]
    commands += [{"kind": "extra", "task_id": 1, "metadata": {**metadata, "gradient_preparation_shards": 2},
                  "argv": ["python", "-m", "mcitlib.methods._keeplora.extract_gradients",
                           "--num-chunks", "2", "--chunk-idx", str(i)],
                  "env": {"CUDA_VISIBLE_DEVICES": str(i)}} for i in range(2)]
    commands.append({"kind": "extra", "task_id": 1, "metadata": {**metadata, "gradient_preparation_shards": 2},
                     "argv": ["python", "-m", "mcitlib.methods._keeplora.extract_gradients", "--num-chunks", "2", "--merge-only"]})
    _json(run / "jobs/lora_ft/ucit/commands.json", {"execution_profile": "four_gpu_validation", "commands": commands})
    report = build_report(run)
    assert report["execution_profile"] == "four_gpu_validation"
    assert report["status"] == "running" and len(report["jobs"]) == 1
    assert report["lineage"]["previous_run"] == str(previous)
    assert report["lineage"]["related_runs"] == ["another-run"]
    assert report["lineage"]["transition_receipt"]["sha256"] == hashlib.sha256(receipt.read_bytes()).hexdigest()
    assert report["lineage"]["receipt_summary"]["status"] == "applied"
    resources = report["jobs"][0]["resources"]
    training = resources["training"][0]
    assert (training["world_size"], training["per_device_batch_size"], training["gradient_accumulation_steps"],
            training["effective_batch_size"]) == (4, 8, 2, 64)
    assert [(item["num_chunks"], item["chunk_index"], item["cuda_visible_devices"]) for item in resources["evaluation"]] == [
        (4, 0, "3"), (4, 1, "2"), (4, 2, "1"), (4, 3, "0")]
    assert [item["num_chunks"] for item in resources["gradient_preparation"]] == [2, 2, 2]
    assert resources["gradient_preparation"][-1]["merge_only"] is True
    outputs = generate_report(run, tmp_path / "out")
    markdown = Path(outputs["markdown"]).read_text()
    assert "plans, not proof of execution" in markdown and "2 chunks" in markdown and "4 chunks" in markdown
    with Path(outputs["csv"]).open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["train_world_size"] == "4" and row["effective_batch_size"] == "64"
    assert row["eval_num_chunks"] == "4" and row["gradient_preparation_num_chunks"] == "2"


def test_old_run_and_unavailable_or_conflicting_command_resources(tmp_path: Path) -> None:
    run, _, _ = _fixture(tmp_path)
    report = build_report(run)
    assert report["execution_profile"] is None and report["lineage"]["previous_run"] is None
    assert report["jobs"][0]["resources"]["training"] is None
    commands = [{"kind": "train", "task_id": 1, "argv": ["deepspeed", "--include", "localhost:0,1",
                 "--per_device_train_batch_size", "16", "--gradient_accumulation_steps", "2"]},
                {"kind": "train", "task_id": 2, "argv": ["deepspeed", "--include", "localhost:0,1",
                 "--per_device_train_batch_size", "16"]},
                {"kind": "train", "task_id": 3, "metadata": {"effective_batch": 999},
                 "argv": ["deepspeed", "--include", "localhost:0,1", "--per_device_train_batch_size", "16",
                          "--gradient_accumulation_steps", "2"]}]
    _json(run / "jobs/lora_ft/ucit/commands.json", {"commands": commands})
    job = build_report(run)["jobs"][0]
    resources = job["resources"]["training"]
    assert resources[0]["world_size"] == 2 and resources[0]["effective_batch_size"] == 64
    assert resources[1]["gradient_accumulation_steps"] is None and resources[1]["effective_batch_size"] is None
    assert resources[2]["effective_batch_size"] is None
    assert any("effective batch disagrees" in warning for warning in job["warnings"])


def test_both_clis_and_output_protection(tmp_path: Path) -> None:
    run, _, _ = _fixture(tmp_path, "pending")
    for module in ("mcitlib.reporting", "mcitlib.cli"):
        output = tmp_path / module
        args = [sys.executable, "-m", module]
        if module.endswith("cli"):
            args.append("report")
        args.extend(["--run-dir", str(run), "--output-dir", str(output)])
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        completed = subprocess.run(args, env=env, capture_output=True, text=True, check=True)
        paths = json.loads(completed.stdout)
        assert all(Path(path).is_file() for path in paths.values())
    with pytest.raises(ValueError, match="outside the run"):
        generate_report(run, run / "reports")
    output = tmp_path / "link-output"
    output.mkdir()
    state = (run / "state.json").read_bytes()
    (output / "report.json").symlink_to(run / "state.json")
    generate_report(run, output)
    assert (run / "state.json").read_bytes() == state
    assert not (output / "report.json").is_symlink()

