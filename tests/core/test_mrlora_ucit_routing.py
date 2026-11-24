# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""CPU-only checks for UCIT routing choice and sample alignment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mcitlib.config.schema import PathSet
from mcitlib.dataflow import find_gaps, produced_paths
from mcitlib.methods._mrlora.router_results.ucit_validation import EXPERTS, route_predictions, select_expert
from mcitlib.paths import PathResolver
from mcitlib.runner import build_sequence_commands


class FixedRandom:
    def __init__(self, index: int):
        self.index = index
        self.calls = 0

    def randint(self, first: int, last: int) -> int:
        assert (first, last) == (0, 5)
        self.calls += 1
        return self.index


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _inputs(tmp_path: Path) -> tuple[Path, Path, list[Path], Path]:
    annotation = tmp_path / "questions.json"
    annotation.write_text(json.dumps([{"id": "first", "image": "images/a.jpg", "answer": "yes"},
                                      {"id": "second", "image": "images/b.jpg", "answer": "no"}]))
    reference = tmp_path / "captions.json"
    reference.write_text(json.dumps({"images": [{"id": 1, "file_name": "a.jpg"}, {"id": 2, "file_name": "b.jpg"}],
                                     "annotations": [{"image_id": 1, "caption": "a"}, {"image_id": 2, "caption": "b"}]}))
    router = tmp_path / "router.jsonl"
    _jsonl(router, [{"question_id": "second", "text": "F"}, {"question_id": "first", "text": "AB"}])
    experts = []
    for index, name in enumerate(EXPERTS):
        path = tmp_path / name / "merge.jsonl"
        _jsonl(path, [{"question_id": "second", "text": f"{index}-second", "prompt": "q2"},
                      {"question_id": "first", "text": f"{index}-first", "prompt": "q1"}])
        experts.append(path)
    return annotation, router, experts, reference


def test_ucit_original_six_expert_choice_and_random_fallback() -> None:
    rng = FixedRandom(5)
    assert select_expert(" AB ", rng) == (0, "A", None)
    assert select_expert("F.", rng) == (5, "F", None)
    assert select_expert("bad", rng) == (5, None, "multiple_non_label_characters")
    assert rng.calls == 1


def test_ucit_joins_by_id_and_preserves_annotation_caption_order(tmp_path: Path) -> None:
    annotation, router, experts, reference = _inputs(tmp_path)
    output, normalized, stats = (tmp_path / "out" / name for name in ("selected.jsonl", "questions.json", "stats.json"))
    report = route_predictions(annotation, router, experts, output, normalized, stats, expected_count=2,
                               expected_sha256=hashlib.sha256(annotation.read_bytes()).hexdigest(),
                               caption_reference=reference, caption_sha256=hashlib.sha256(reference.read_bytes()).hexdigest())
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["question_id"] for row in rows] == ["first", "second"]
    assert [row["text"] for row in rows] == ["0-first", "5-second"]
    assert [row["question_id"] for row in json.loads(normalized.read_text())] == ["first", "second"]
    assert report["samples"] == 2 and report["caption_alignment_verified"]
    assert report["selected_experts"]["ImageNet-R"] == report["selected_experts"]["Flickr30k"] == 1


def test_ucit_rejects_missing_duplicate_wrong_type_and_misaligned_caption(tmp_path: Path) -> None:
    annotation, router, experts, reference = _inputs(tmp_path)
    output, normalized, stats = (tmp_path / "out" / name for name in ("selected.jsonl", "questions.json", "stats.json"))
    with pytest.raises(ValueError, match="expected 3 questions"):
        route_predictions(annotation, router, experts, output, normalized, stats, expected_count=3)
    ref = json.loads(reference.read_text())
    ref["images"].reverse()
    reference.write_text(json.dumps(ref))
    with pytest.raises(ValueError, match="caption image IDs"):
        route_predictions(annotation, router, experts, output, normalized, stats, caption_reference=reference)
    assert not output.exists()
    _jsonl(experts[0], [{"question_id": "first", "text": "x"}, {"question_id": "first", "text": "y"}])
    with pytest.raises(ValueError, match="duplicate question identifier"):
        route_predictions(annotation, router, experts, output, normalized, stats)
    _jsonl(experts[0], [{"question_id": "first", "text": "x"}])
    with pytest.raises(ValueError, match="expert question ID coverage"):
        route_predictions(annotation, router, experts, output, normalized, stats)


def test_ucit_fallback_is_traceable_and_wrong_hash_fails(tmp_path: Path) -> None:
    annotation, router, experts, _ = _inputs(tmp_path)
    _jsonl(router, [{"question_id": "first", "text": "not a label"}, {"question_id": "second", "text": "F"}])
    output, normalized, stats = (tmp_path / "out" / name for name in ("selected.jsonl", "questions.json", "stats.json"))
    with pytest.raises(ValueError, match="SHA-256 differs"):
        route_predictions(annotation, router, experts, output, normalized, stats, expected_sha256="0" * 64)
    assert not output.exists()
    info = route_predictions(annotation, router, experts, output, normalized, stats, rng=FixedRandom(2))
    first = json.loads(output.read_text().splitlines()[0])
    assert first["text"] == "2-first" and first["routing_validation"]["expert"] == "VizWiz"
    assert first["routing_validation"]["status"] == "random_fallback"
    assert info["random_fallbacks"] == 1


@pytest.mark.parametrize("four", [False, True])
def test_ucit_plan_uses_six_experts_and_real_router_with_no_unproduced_inputs(repo_root: Path, four: bool) -> None:
    paths = PathResolver(PathSet(Path("/audit/model"), Path("/audit/data"), Path("/audit/ckpt"), repo_root,
                                 Path("/audit/results"), repo_root / "logs"))
    hashes = {name: "a" * 64 for name in EXPERTS}
    hashes.update({"VizWiz:caption": "b" * 64, "Flickr30k:caption": "c" * 64})
    args = {"paths": paths, "validation_fixes": True, "expected_eval_samples": 1000,
            "expected_eval_hashes": hashes, "gpus": "0,1,2,3" if four else "0,1"}
    if four:
        args["execution_profile"] = "four_gpu_validation"
    config = repo_root / "configs"
    experts = build_sequence_commands(config, "llava", "mrlora", "ucit", **args)
    router = build_sequence_commands(config, "llava", "mrlora", "ucit", sequence="router", **args)
    n = 4 if four else 2
    inference = [item for item in experts if item.metadata.get("validation_fix") == "mrlora_ucit_expert_inference"]
    router_inference = [item for item in router if item.metadata.get("validation_fix") == "mrlora_ucit_router_inference"]
    joins = [item for item in router if item.metadata.get("validation_fix") == "mrlora_ucit_router_selection"]
    scores = [item for item in router if item.metadata.get("validation_fix") == "mrlora_ucit_router_scoring"]
    assert len(inference) == 36 * n and len(router_inference) == 21 * n and len(joins) == len(scores) == 21
    def option(item, flag):
        return item.argv[item.argv.index(flag) + 1]
    outputs = {option(item, "--answers-file") for item in inference}
    assert len(outputs) == len(inference)
    for item in inference:
        answer = Path(option(item, "--answers-file"))
        domain = answer.parent.parent.name
        assert domain in EXPERTS and answer.parent.name == f"MR-LoRA-task{item.task_id}"
        assert option(item, "--model-path").endswith(f"/Task{item.task_id}_llava_lora")
        assert Path(option(item, "--question-file")).name == "test_3000.json"
    assert len({option(item, "--answers-file") for item in router_inference}) == len(router_inference)
    for item in router_inference:
        answer = Path(option(item, "--answers-file"))
        assert answer.parent.parent.name in EXPERTS[:item.task_id]
        assert "UCIT_router_raw" in answer.parts
        assert option(item, "--model-path").endswith(f"/Router_task{item.task_id}_llava_lora")
    for item in joins:
        domain = Path(option(item, "--output")).parent.parent.name
        assert option(item, "--expected-count") == "1000" and option(item, "--annotation-sha256") == hashes[domain]
        inputs = item.argv[item.argv.index("--inputs") + 1:]
        assert len(inputs) == 6 and len(set(inputs)) == 6
        assert all(f"/UCIT/each_dataset/{domain}/MR-LoRA-task{stage}/merge.jsonl" in inputs[stage - 1] for stage in range(1, 7))
        if domain in {"VizWiz", "Flickr30k"}:
            assert option(item, "--caption-sha256") == hashes[f"{domain}:caption"]
    assert len({option(item, "--output-dir") for item in scores}) == 21
    roots = paths.external_roots("ucit")
    internal = (str(paths.paths.ckpt_root), str(paths.results_root))
    assert not find_gaps(experts, roots, internal_roots=internal)
    available = produced_paths(experts)
    assert not find_gaps(router, roots, tuple(available), internal)
    join = joins[0]
    missing = option(join, "--inputs")
    assert any(gap.flag == "--inputs" and gap.path == missing
               for gap in find_gaps([join], roots, tuple(available - {missing}), internal))
    score = next(item for item in scores if "--annotation-file" in item.argv and option(item, "--annotation-file").endswith("scorer_annotations.json"))
    assert any(gap.flag == "--annotation-file" for gap in find_gaps([score], roots, internal_roots=internal))
