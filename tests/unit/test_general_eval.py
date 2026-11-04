# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""CPU fixtures for the four general-benchmark scoring protocols."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mcitlib.eval.general import main, merge_jsonl, score_mmbench, score_mme, score_pope, score_seed


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_pope_categories_and_binary_metrics(tmp_path: Path) -> None:
    questions = [
        {"question_id": 1, "category": "random"}, {"question_id": 2, "category": "random"},
        {"question_id": 3, "category": "random"}, {"question_id": 4, "category": "random"},
    ]
    answers = [{"question_id": i, "text": text} for i, text in enumerate(("Yes", "No.", "No", "Yes"), 1)]
    labels = [{"label": value} for value in ("yes", "yes", "no", "no")]
    write_jsonl(tmp_path / "questions.jsonl", questions)
    write_jsonl(tmp_path / "answers.jsonl", answers)
    annotations = tmp_path / "annotations"
    write_jsonl(annotations / "coco_pope_random.json", labels)
    result = score_pope(tmp_path / "questions.jsonl", tmp_path / "answers.jsonl", annotations)["categories"]["random"]
    assert result["accuracy"] == 50
    assert result["tp"] == 1 and result["tn"] == 1
    assert result["f1"] == pytest.approx(0.5)


def test_mme_accuracy_plus_and_group_totals(tmp_path: Path) -> None:
    questions = []
    answers = []
    for image in ("img1", "img2"):
        for number in (1, 2):
            qid = f"existence/{image}.jpg/{number}"
            questions.append({"question_id": qid, "text": "Question?", "label": "yes"})
            answers.append({"question_id": qid, "text": "no" if qid == "existence/img2.jpg/2" else "yes"})
    write_jsonl(tmp_path / "questions.jsonl", questions)
    write_jsonl(tmp_path / "answers.jsonl", answers)
    result = score_mme(tmp_path / "questions.jsonl", tmp_path / "answers.jsonl", tmp_path)
    assert result["tasks"]["existence"] == 125
    assert result["perception"] == 125
    assert result["cognition"] == 0
    assert not result["complete"]


def test_mmbench_offline_extraction_and_merge(tmp_path: Path) -> None:
    annotations = tmp_path / "mmbench.tsv"
    annotations.write_text("index\tanswer\tA\tB\tquestion\n1\tA\tcat\tdog\tq1\n2\tB\tcat\tdog\tq2\n", encoding="utf-8")
    write_jsonl(tmp_path / "a.jsonl", [{"question_id": 1, "text": "A. cat"}])
    write_jsonl(tmp_path / "b.jsonl", [{"question_id": 2, "text": "unknown"}])
    merged = tmp_path / "merged.jsonl"
    assert merge_jsonl([tmp_path / "a.jsonl", tmp_path / "b.jsonl"], merged) == 2
    result = score_mmbench(annotations, merged)
    assert result["accuracy"] == 50
    assert result["unparsed"] == 1
    output = tmp_path / "score.json"
    main(["score", "--benchmark", "mmbench", "--annotation-file", str(annotations),
          "--result-file", str(merged), "--output", str(output)])
    assert json.loads(output.read_text())["accuracy"] == 50


def test_seed_all_image_video_and_question_type_accuracy(tmp_path: Path) -> None:
    annotations = {
        "question_type": {"scene": 1},
        "questions": [
            {"question_id": 1, "question_type_id": 1, "data_type": "image", "answer": "A"},
            {"question_id": 2, "question_type_id": 1, "data_type": "image", "answer": "B"},
            {"question_id": 3, "question_type_id": 1, "data_type": "video", "answer": "A"},
        ],
    }
    (tmp_path / "seed.json").write_text(json.dumps(annotations), encoding="utf-8")
    write_jsonl(tmp_path / "answers.jsonl", [
        {"question_id": 1, "text": "A"}, {"question_id": 2, "text": "A"}, {"question_id": 3, "text": "A"},
    ])
    result = score_seed(tmp_path / "seed.json", tmp_path / "answers.jsonl")
    assert result["all"] == pytest.approx(200 / 3)
    assert result["img"] == 50
    assert result["vid"] == 100
    assert result["question_types"]["scene"] == pytest.approx(200 / 3)
