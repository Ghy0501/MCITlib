# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""CL-VISTA-specific replay, router selection and Octopus sample budgets."""
from __future__ import annotations

from random import Random
from typing import Any

ROUTER_LABELS = {
    "F": "counting", "G": "space", "D": "traffic", "A": "movie",
    "C": "gui", "E": "science", "H": "sports", "B": "star",
}


def typed_question_id(row: dict) -> tuple[type, Any]:
    value = row["question_id"]
    if type(value) not in {str, int} or value == "":
        raise ValueError(f"Invalid CL-VISTA question_id: {value!r}")
    return type(value), value


def prediction_index(rows):
    index = {}
    for row in rows:
        identity = typed_question_id(row)
        if identity in index:
            raise ValueError(f"Duplicate CL-VISTA question ID: {identity[1]!r}")
        index[identity] = row
    return index


def routed_predictions(questions, router_rows, expert_rows):
    """Select an expert prediction for each held-out question by ID inside its cell.

    The available experts are explicit. Incomplete coverage is returned with the
    expected denominator so callers can retain the native predictions and report
    a diagnostic failure without silently excluding questions from scoring.
    """
    expected = prediction_index(questions)
    router = prediction_index(router_rows)
    if router.keys() - expected.keys():
        raise ValueError("Router predictions include IDs outside the evaluation cell")
    indexed_experts = {expert: prediction_index(rows) for expert, rows in expert_rows.items()}
    if any(rows.keys() - expected.keys() for rows in indexed_experts.values()):
        raise ValueError("Expert predictions include IDs outside the evaluation cell")
    selected = []
    missing = []
    for qid, question in expected.items():
        decision = router.get(qid)
        if decision is None:
            missing.append({"question_id": qid[1], "reason": "router_prediction_missing"})
            continue
        raw_label = decision["pred"]
        label = raw_label.strip() if isinstance(raw_label, str) else ""
        expert = ROUTER_LABELS.get(label)
        if expert not in indexed_experts:
            missing.append({"question_id": qid[1], "reason": "expert_unavailable", "router_output": raw_label})
            continue
        row = indexed_experts[expert].get(qid)
        if row is None:
            missing.append({"question_id": qid[1], "reason": "expert_prediction_missing", "expert": expert})
            continue
        if row["question"] != question["question"]:
            raise ValueError(f"Expert question text differs for ID {qid[1]!r}")
        selected.append({**row, "expert": expert, "router_label": label})
    return selected, {"expected": len(expected), "selected": len(selected), "missing": missing,
                      "expert_pool": sorted(indexed_experts)}


def scaled_auxiliary_sample(sample_rows, full_train_count, full_budget, *, seed):
    """Scale an Octopus Stage2 or GPWC draw by this task's retain ratio."""
    count = len(sample_rows)
    if not 0 < count <= full_train_count or type(full_budget) is not int or full_budget < 0:
        raise ValueError("The task must bind sample/full training sizes and a nonnegative original auxiliary budget")
    if full_budget == 0:
        return [], {"full_train": full_train_count, "sample_train": count, "full_budget": 0,
                    "sample_count": 0, "seed": seed, "indices": []}
    rounded = (2 * full_budget * count + full_train_count) // (2 * full_train_count)
    target = min(count, max(1, rounded))
    indices = Random(seed).sample(range(count), target)
    return [sample_rows[index] for index in indices], {
        "full_train": full_train_count, "sample_train": count, "full_budget": full_budget,
        "sample_count": target, "seed": seed, "indices": indices,
    }


def hash_video_qa_inputs(paths, media_root):
    """Validate source-relative video paths while preserving original QA rows."""
    import hashlib
    import json
    from pathlib import Path

    root = Path(media_root).resolve(strict=True)
    cached = set()
    datasets, hashes = [], []
    for filename in paths:
        payload = Path(filename).read_bytes()
        rows = json.loads(payload)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Expected a nonempty JSON annotation array: {filename}")
        for row in rows:
            video = Path(row["video"])
            if video.is_absolute() or ".." in video.parts:
                raise ValueError(f"Unbound CL-VISTA video path: {video}")
            if video not in cached:
                path = (root / video).resolve(strict=True)
                if root not in path.parents or not path.is_file():
                    raise ValueError(f"Video escapes the specified CL-VISTA root: {video}")
                cached.add(video)
        datasets.append(rows)
        hashes.append(hashlib.sha256(payload).hexdigest())
    if not datasets:
        raise ValueError("At least the current CL-VISTA training annotation is required")
    return datasets, hashes
