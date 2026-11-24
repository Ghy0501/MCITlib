# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Join UCIT router choices to six independently evaluated experts."""
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

EXPERTS = ("ImageNet-R", "ArxivQA", "VizWiz", "IconQA", "CLEVR-Math", "Flickr30k")


def select_expert(raw: str, rng: Any = random) -> tuple[int, str | None, str | None]:
    """Preserve the standalone UCIT choose_ans priority and random fallback."""
    text = raw.strip()
    if sum(char not in "ABCDEF" for char in text) <= 1:
        for index, label in enumerate("ABCDEF"):
            if label in text:
                return index, label, None
        reason = "no_matching_label"
    else:
        reason = "multiple_non_label_characters"
    return rng.randint(0, 5), None, reason


def _identifier(record: Any, source: Path) -> str | int:
    if not isinstance(record, dict):
        raise ValueError(f"{source}: record must be an object")
    qid = record.get("question_id")
    alternate = record.get("id")
    if qid is None:
        qid = alternate
    elif alternate is not None and (type(qid) is not type(alternate) or qid != alternate):
        raise ValueError(f"{source}: id and question_id disagree")
    if not isinstance(qid, (str, int)) or isinstance(qid, bool) or str(qid) == "":
        raise ValueError(f"{source}: missing or invalid question identifier")
    return qid


def _index(records: list[Any], source: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        qid = str(_identifier(record, source))
        if qid in indexed:
            raise ValueError(f"{source}: duplicate question identifier {qid!r}")
        indexed[qid] = record
    if not indexed:
        raise ValueError(f"{source}: no records")
    return indexed


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{number}: blank record")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected object")
            rows.append(value)
    return rows


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=".ucit-", delete=False) as handle:
            name = handle.name
            handle.write(text)
        os.replace(name, path)
    finally:
        if name is not None and os.path.exists(name):
            os.unlink(name)


def _verified_bytes(path: Path, expected_hash: str | None) -> bytes:
    data = path.read_bytes()
    if expected_hash is not None and hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError(f"{path}: input SHA-256 differs from recorded sample manifest")
    return data


def validate_caption_alignment(annotations: list[dict[str, Any]], reference: Any, path: Path) -> None:
    """The existing caption scorer numbers predictions by order, not question ID."""
    if not isinstance(reference, dict) or not isinstance(reference.get("images"), list):
        raise ValueError(f"{path}: caption reference images are unavailable")
    images = reference["images"]
    if len(images) != len(annotations):
        raise ValueError(f"{path}: caption reference image count differs from questions")
    image_ids = set()
    for index, (question, image) in enumerate(zip(annotations, images), 1):
        if not isinstance(image, dict) or image.get("id") != index:
            raise ValueError(f"{path}: caption image IDs must follow question order 1..N")
        if not question.get("image") or not image.get("file_name") or Path(str(question["image"])).name != Path(str(image["file_name"])).name:
            raise ValueError(f"{path}: caption image filename differs at question {index}")
        image_ids.add(index)
    refs = reference.get("annotations")
    if not isinstance(refs, list) or any(not isinstance(item, dict) or item.get("image_id") not in image_ids for item in refs):
        raise ValueError(f"{path}: invalid caption references")
    counts = Counter(item["image_id"] for item in refs)
    if any(counts[image_id] == 0 for image_id in image_ids):
        raise ValueError(f"{path}: some caption images have no references")


def route_predictions(annotation_file: str | Path, router_file: str | Path, expert_files: list[str | Path],
                      output: str | Path, normalized_annotation: str | Path, stats_file: str | Path,
                      *, expected_count: int | None = None, expected_sha256: str | None = None,
                      caption_reference: str | Path | None = None, caption_sha256: str | None = None,
                      rng: Any = random) -> dict[str, Any]:
    """Select predictions in input-question order; require six complete candidate files."""
    annotation_path, router_path = Path(annotation_file), Path(router_file)
    experts = [Path(path) for path in expert_files]
    output_path, normalized_path, stats_path = Path(output), Path(normalized_annotation), Path(stats_file)
    reference_path = Path(caption_reference) if caption_reference is not None else None
    if len(experts) != len(EXPERTS):
        raise ValueError("exactly six experts in UCIT benchmark order are required")
    inputs = [annotation_path, router_path, *experts] + ([reference_path] if reference_path else [])
    destinations = [output_path.resolve(), normalized_path.resolve(), stats_path.resolve()]
    if len(set(destinations)) != len(destinations) or any(dest in {path.resolve() for path in inputs} for dest in destinations):
        raise ValueError("output files must be distinct from each other and all inputs")
    data = _verified_bytes(annotation_path, expected_sha256)
    annotations = json.loads(data)
    if not isinstance(annotations, list):
        raise ValueError(f"{annotation_path}: expected a list")
    source = _index(annotations, annotation_path)
    if expected_count is not None and (type(expected_count) is not int or expected_count < 1 or len(source) != expected_count):
        raise ValueError(f"{annotation_path}: expected {expected_count} questions, found {len(source)}")
    if reference_path:
        validate_caption_alignment(annotations, json.loads(_verified_bytes(reference_path, caption_sha256)), reference_path)
    elif caption_sha256 is not None:
        raise ValueError("caption reference hash was provided without a reference")
    expected = set(source)
    router = _index(_read_jsonl(router_path), router_path)
    if set(router) != expected:
        raise ValueError(f"{router_path}: router question ID coverage differs from annotations")
    candidate_maps = []
    for path in experts:
        candidate = _index(_read_jsonl(path), path)
        if set(candidate) != expected:
            raise ValueError(f"{path}: expert question ID coverage differs from annotations")
        for key, item in candidate.items():
            original = _identifier(source[key], annotation_path)
            actual = _identifier(item, path)
            if type(actual) is not type(original) or actual != original:
                raise ValueError(f"{path}: question ID type or value differs for {key!r}")
            if not isinstance(item.get("text"), str):
                raise ValueError(f"{path}: expert answer text is missing for {key!r}")
        candidate_maps.append(candidate)
    selected = []
    normalized = []
    counts: Counter[str] = Counter()
    for question in annotations:
        original = _identifier(question, annotation_path)
        key = str(original)
        router_item = router[key]
        if type(_identifier(router_item, router_path)) is not type(original):
            raise ValueError(f"{router_path}: question ID type differs for {key!r}")
        raw = router_item.get("text")
        if not isinstance(raw, str):
            raise ValueError(f"{router_path}: router text is missing for {key!r}")
        chosen, label, reason = select_expert(raw, rng)
        prediction = dict(candidate_maps[chosen][key])
        prediction["question_id"] = original
        prediction["agent_selection"] = raw
        prediction["routing_validation"] = {"status": "random_fallback" if reason else "selected",
                                            "label": label, "expert": EXPERTS[chosen],
                                            "expert_file": str(experts[chosen]), "fallback_reason": reason}
        selected.append(prediction)
        normalized.append({**question, "question_id": original})
        counts[EXPERTS[chosen]] += 1
        if reason:
            counts["fallback"] += 1
    summary = {"samples": len(selected), "selected_experts": {name: counts[name] for name in EXPERTS},
               "random_fallbacks": counts["fallback"], "fallback_policy": "standalone choose_ans random.randint(0,5)",
               "expected_samples": expected_count, "annotation_sha256": hashlib.sha256(data).hexdigest(),
               "expected_annotation_sha256": expected_sha256, "caption_reference": str(reference_path) if reference_path else None,
               "caption_alignment_verified": bool(reference_path), "expert_pool": list(EXPERTS),
               "retrospective_six_expert_pool": True}
    _write(output_path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
    _write(normalized_path, json.dumps(normalized, ensure_ascii=False) + "\n")
    _write(stats_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-file", required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--normalized-annotation", required=True)
    parser.add_argument("--stats-file", required=True)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--annotation-sha256")
    parser.add_argument("--caption-reference")
    parser.add_argument("--caption-sha256")
    parser.add_argument("--inputs", nargs=6, required=True)
    args = parser.parse_args(argv)
    try:
        result = route_predictions(args.annotation_file, args.result_file, args.inputs, args.output,
                                   args.normalized_annotation, args.stats_file, expected_count=args.expected_count,
                                   expected_sha256=args.annotation_sha256, caption_reference=args.caption_reference,
                                   caption_sha256=args.caption_sha256)
    except (OSError, ValueError, TypeError) as exc:
        print(f"mrlora UCIT routing: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
