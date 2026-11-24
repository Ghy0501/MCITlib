# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Select DCL expert predictions using router labels, with strict sample alignment.

The original MR-LoRA expert pool is ordered RS, Med, AD, Sci, Fin while its
router prompt labels them E, C, D, B, A (and F also selects RS). This module
writes selected prediction records for the existing task scorer; it does not
calculate scores itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

EXPERTS = ("RS", "Med", "AD", "Sci", "Fin")
LABEL_TO_EXPERT = {"A": 4, "B": 3, "C": 1, "D": 2, "E": 0, "F": 0}
PROMPT_ORDER = (4, 3, 1, 2, 0)
SCORERS = {"RS": "eval_ai2d", "Med": "eval_pvqa", "AD": "eval_ai2d",
           "Sci": "eval_sci", "Fin": "eval_finvis"}


def _key(record: Any, source: Path) -> str:
    if not isinstance(record, dict) or record.get("question_id") is None:
        raise ValueError(f"{source}: every record must contain question_id")
    value = record["question_id"]
    if not isinstance(value, (str, int)) or isinstance(value, bool) or not str(value):
        raise ValueError(f"{source}: invalid question_id {value!r}")
    return str(value)


def _indexed(records: list[Any], source: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for record in records:
        key = _key(record, source)
        if key in result:
            raise ValueError(f"{source}: duplicate question_id {key!r}")
        result[key] = record
    if not result:
        raise ValueError(f"{source}: no records")
    return result


def _jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{number}: blank JSONL record")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: record must be an object")
            records.append(value)
    return records


def _atomic_write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=".mrlora-", delete=False) as handle:
            name = handle.name
            handle.write(contents)
        os.replace(name, path)
    finally:
        if name is not None and os.path.exists(name):
            os.unlink(name)


def select_expert(raw: str, rng: Any = random) -> tuple[int, str | None, str | None]:
    """Reproduce standalone choose_ans, including its random fallback and F alias."""
    text = raw.strip()
    if sum(char not in "ABCDEF" for char in text) <= 1:
        for label in "ABCDEF":
            if label in text:
                return LABEL_TO_EXPERT[label], label, None
        reason = "no_matching_label"
    else:
        reason = "multiple_non_label_characters"
    return PROMPT_ORDER[rng.randint(0, 4)], None, reason


def route_predictions(annotation_file: str | Path, router_file: str | Path, expert_files: list[str | Path],
                      output: str | Path, stats_file: str | Path | None = None,
                      expected_count: int | None = None, expected_sha256: str | None = None,
                      rng: Any = random) -> dict[str, Any]:
    """Choose one expert answer per annotation; never silently drop a question.

    Missing/duplicate IDs or incomplete files fail. Invalid labels use the
    standalone implementation's random fallback and are recorded. Original
    expert question_id types are kept.
    """
    annotation_file, router_file, output = Path(annotation_file), Path(router_file), Path(output)
    expert_files = [Path(path) for path in expert_files]
    if len(expert_files) != len(EXPERTS):
        raise ValueError("exactly five expert prediction files are required in RS, Med, AD, Sci, Fin order")
    stats = Path(stats_file) if stats_file is not None else output.parent / "routing_stats.json"
    inputs = {path.resolve() for path in [annotation_file, router_file, *expert_files]}
    if output.resolve() in inputs or stats.resolve() in inputs or output.resolve() == stats.resolve():
        raise ValueError("routed output and statistics file must differ from every input and each other")
    annotation_bytes = annotation_file.read_bytes()
    annotation_hash = hashlib.sha256(annotation_bytes).hexdigest()
    if expected_sha256 is not None and annotation_hash != expected_sha256:
        raise ValueError(f"{annotation_file}: annotation hash differs from the verified sample manifest")
    annotations_raw = json.loads(annotation_bytes)
    if not isinstance(annotations_raw, list):
        raise ValueError(f"{annotation_file}: annotations must be a list")
    annotations = _indexed(annotations_raw, annotation_file)
    if expected_count is not None and (expected_count < 1 or len(annotations) != expected_count):
        raise ValueError(f"{annotation_file}: expected {expected_count} annotations, found {len(annotations)}")
    router_records = _jsonl(router_file)
    router = _indexed(router_records, router_file)
    expected = set(annotations)
    if set(router) != expected:
        raise ValueError(f"{router_file}: question_id coverage differs from annotations "
                         f"(missing {len(expected - set(router))}, extra {len(set(router) - expected)})")
    experts = []
    for path in expert_files:
        items = _indexed(_jsonl(path), path)
        if set(items) != expected:
            raise ValueError(f"{path}: question_id coverage differs from annotations "
                             f"(missing {len(expected - set(items))}, extra {len(set(items) - expected)})")
        for item in items.values():
            if not isinstance(item.get("text"), str) or not isinstance(item.get("prompt"), str):
                raise ValueError(f"{path}: expert records must contain string text and prompt")
            key = _key(item, path)
            expected_id = annotations[key]["question_id"]
            if type(item["question_id"]) is not type(expected_id) or item["question_id"] != expected_id:
                raise ValueError(f"{path}: question_id type differs from annotations for {key!r}")
        experts.append(items)
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for item in router_records:
        qid = _key(item, router_file)
        raw = item.get("text")
        if not isinstance(raw, str):
            raise ValueError(f"{router_file}: router text must be a string for question_id {qid!r}")
        index, label, fallback_reason = select_expert(raw, rng)
        prediction = dict(experts[index][qid])
        prediction["routing_validation"] = {"status": "random_fallback" if fallback_reason else "selected",
                                              "label": label, "expert": EXPERTS[index],
                                              "expert_file": str(expert_files[index]),
                                              "fallback_reason": fallback_reason}
        counts[EXPERTS[index]] += 1
        if fallback_reason:
            counts["fallback"] += 1
        prediction["agent_selection"] = raw
        selected.append(prediction)
    summary = {"samples": len(selected), "selected_experts": {name: counts[name] for name in EXPERTS},
               "random_fallbacks": counts["fallback"], "fallback_policy": "standalone choose_ans random.randint(0,4)",
               "expected_samples": expected_count, "annotation_sha256": annotation_hash,
               "expected_annotation_sha256": expected_sha256, "expert_pool": list(EXPERTS),
               "retrospective_five_expert_pool": True}
    _atomic_write(output, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
    _atomic_write(stats, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-file", required=True)
    parser.add_argument("--result-file", required=True, help="router merge.jsonl")
    parser.add_argument("--output", required=True, help="selected expert predictions in JSONL")
    parser.add_argument("--stats-file", required=True)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--annotation-sha256")
    parser.add_argument("--inputs", nargs=5, required=True, metavar=("RS", "MED", "AD", "SCI", "FIN"))
    args = parser.parse_args(argv)
    try:
        result = route_predictions(args.annotation_file, args.result_file, args.inputs, args.output, args.stats_file,
                                   args.expected_count, args.annotation_sha256)
    except (OSError, ValueError, TypeError) as error:
        print(f"mrlora DCL routing: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
