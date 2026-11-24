# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""CPU-only checks for reuse and final UCIT router evidence."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from mcitlib.methods._mrlora.router_results.ucit_validation import EXPERTS
from mcitlib.methods._mrlora.ucit_eval import _copy_verified, _finish, _validate_reused_cell, main


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_reuse_requires_original_ids_chunks_commands_and_denominator(tmp_path: Path) -> None:
    annotation = tmp_path / "test.json"
    annotation.write_text(json.dumps([{"question_id": 1}, {"question_id": 2}]), encoding="utf-8")
    checkpoint, model_base = tmp_path / "Task1_llava_lora", tmp_path / "base"
    folder = tmp_path / "ImageNet-R" / "MR-LoRA-task1"
    rows = [{"question_id": n, "text": "answer", "model_id": checkpoint.name} for n in (1, 2)]
    for index in range(4):
        _jsonl(folder / f"4_{index}.jsonl", rows[index:index + 1])
    merge = folder / "merge.jsonl"
    merge.write_bytes(b"".join((folder / f"4_{i}.jsonl").read_bytes() for i in range(4)))
    (folder / "Result.text").write_text("Samples: 2\nAccuracy: 50.00%\n", encoding="utf-8")
    commands = [{"sequence": "router", "task_id": 1,
                 "argv": ["python", "--model-path", str(checkpoint), "--model-base", str(model_base),
                          "--question-file", str(annotation), "--answers-file", str(folder / f"4_{i}.jsonl"),
                          "--temperature", "0", "--conv-mode", "vicuna_v1", "--num-chunks", "4"]}
                for i in range(4)]
    events = {i: {"status": "completed", "return_code": 0, "command": cmd} for i, cmd in enumerate(commands)}
    result = _validate_reused_cell(folder, "ImageNet-R", 1, annotation, 2, commands, events,
                                   checkpoint, model_base)
    assert result["samples"] == 2 and len(result["completed_inference_indices"]) == 4
    events[3]["status"] = "failed"
    with pytest.raises(ValueError, match="missing successful recorded inference"):
        _validate_reused_cell(folder, "ImageNet-R", 1, annotation, 2, commands, events, checkpoint, model_base)
    events[3]["status"] = "completed"
    _jsonl(merge, list(reversed(rows)))
    with pytest.raises(ValueError, match="IDs or order differ"):
        _validate_reused_cell(folder, "ImageNet-R", 1, annotation, 2, commands, events, checkpoint, model_base)
    _jsonl(merge, rows)
    (folder / "4_0.jsonl").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="merged predictions differ"):
        _validate_reused_cell(folder, "ImageNet-R", 1, annotation, 2, commands, events, checkpoint, model_base)


def test_copy_checks_hash_and_never_overwrites_source_or_destination(tmp_path: Path) -> None:
    source, target = tmp_path / "old.jsonl", tmp_path / "new" / "merge.jsonl"
    source.write_text("original\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="source changed"):
        _copy_verified(source, target, "0" * 64)
    assert not target.exists()
    _copy_verified(source, target, digest)
    with pytest.raises(ValueError, match="overwrite"):
        _copy_verified(source, target, digest)
    assert source.read_bytes() == target.read_bytes() == b"original\n"


def test_verified_metrics_require_all_domains_scorer_ids_and_denominators(tmp_path: Path) -> None:
    results, output = tmp_path / "UCIT_router", tmp_path / "metrics.json"
    hashes = {name: "a" * 64 for name in EXPERTS}
    hashes.update({"VizWiz:caption": "b" * 64, "Flickr30k:caption": "c" * 64})
    folders = []
    for stage in range(1, len(EXPERTS) + 1):
        for domain in EXPERTS[:stage]:
            folder = results / "each_dataset" / domain / f"MR-LoRA-router-task{stage}"
            folder.mkdir(parents=True)
            folders.append(folder)
            (folder / "Result.text").write_text("Samples: 2\nAccuracy: 50.00%\n", encoding="utf-8")
            stats = {"samples": 2, "expected_samples": 2, "annotation_sha256": hashes[domain],
                     "expected_annotation_sha256": hashes[domain], "expert_pool": list(EXPERTS),
                     "retrospective_six_expert_pool": True, "caption_alignment_verified": True}
            (folder / "routing_stats.json").write_text(json.dumps(stats), encoding="utf-8")
            _jsonl(folder / "routed_predictions.jsonl", [{"question_id": 1}, {"question_id": 2}])
            _jsonl(folder / "Result.json", [{"question_id": 1}, {"question_id": 2}])
    (folders[-1] / "Result.text").write_text("Samples: 1\nAccuracy: 50%\n", encoding="utf-8")
    with pytest.raises(ValueError, match="scorer denominator"):
        _finish(results, output, 2, hashes)
    assert not output.exists()
    (folders[-1] / "Result.text").write_text("Samples: 2\nAccuracy: 50%\n", encoding="utf-8")
    _jsonl(folders[-1] / "Result.json", [{"question_id": 2}, {"question_id": 1}])
    with pytest.raises(ValueError, match="scorer question IDs"):
        _finish(results, output, 2, hashes)
    assert not output.exists()
    _jsonl(folders[-1] / "Result.json", [{"question_id": 1}, {"question_id": 2}])
    result = _finish(results, output, 2, hashes)
    assert result["source"] == str(results) and result["metrics"]["mfn"] == 50
    with pytest.raises(FileExistsError):
        _finish(results, output, 2, hashes)


def test_router_generation_keeps_original_token_limit_without_extra_stopping_criterion(repo_root: Path) -> None:
    source = repo_root / "mcitlib/methods/_mrlora/eval_runners/model_ucit_router.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "generate"]
    assert len(calls) == 1
    options = {item.arg for item in calls[0].keywords}
    assert "max_new_tokens" in options and "stopping_criteria" not in options


def test_execute_requires_passing_doctor_for_exact_reviewed_plan(tmp_path: Path, monkeypatch) -> None:
    from mcitlib.methods._mrlora import ucit_eval

    plan, doctor = tmp_path / "plan.json", tmp_path / "doctor.json"
    plan.write_text("{}\n", encoding="utf-8")
    digest = hashlib.sha256(plan.read_bytes()).hexdigest()
    monkeypatch.setattr(ucit_eval, "_reviewed_plan", lambda _: {"commands": [], "logs_dir": str(tmp_path)})
    def never_execute(*args, **kwargs):
        pytest.fail("invalid doctor evidence must not execute commands")
    monkeypatch.setattr(ucit_eval, "execute_commands", never_execute)
    for evidence in ({"failed": True, "plan_sha256": digest},
                     {"failed": False, "plan_sha256": "0" * 64}, {}):
        doctor.write_text(json.dumps(evidence), encoding="utf-8")
        assert main(["execute", "--plan-file", str(plan), "--doctor-report", str(doctor)]) == 2
