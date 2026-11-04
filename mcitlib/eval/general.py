# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Offline scoring utilities for POPE, MME, MMBench, and SEED-Bench."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

MME_TASKS = {
    "perception": ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"],
    "cognition": ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"],
}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def merge_jsonl(inputs: list[str | Path], output: str | Path) -> int:
    rows = [row for path in inputs for row in read_jsonl(path)]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _pope_prediction(text: str) -> int:
    text = text.split(".", 1)[0].replace(",", "")
    return 0 if any(word in text.split(" ") for word in ("No", "not", "no")) else 1


def score_pope(question_file: str | Path, result_file: str | Path, annotation_dir: str | Path) -> dict[str, Any]:
    questions = {row["question_id"]: row for row in read_jsonl(question_file)}
    answers = read_jsonl(result_file)
    output: dict[str, Any] = {"categories": {}}
    for annotation in sorted(Path(annotation_dir).glob("coco_pope_*.json")):
        category = annotation.name[10:-5]
        current = [row for row in answers if questions[row["question_id"]]["category"] == category]
        labels = [0 if row["label"] == "no" else 1 for row in read_jsonl(annotation)]
        if len(current) != len(labels):
            raise ValueError(f"POPE {category}: {len(current)} predictions for {len(labels)} labels")
        predictions = [_pope_prediction(row["text"]) for row in current]
        tp = sum(pred == 1 and label == 1 for pred, label in zip(predictions, labels))
        tn = sum(pred == 0 and label == 0 for pred, label in zip(predictions, labels))
        fp = sum(pred == 1 and label == 0 for pred, label in zip(predictions, labels))
        fn = sum(pred == 0 and label == 1 for pred, label in zip(predictions, labels))
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        output["categories"][category] = {
            "samples": len(current), "accuracy": 100 * _safe_divide(tp + tn, len(current)),
            "precision": precision, "recall": recall,
            "f1": _safe_divide(2 * precision * recall, precision + recall),
            "yes_ratio": _safe_divide(predictions.count(1), len(predictions)),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        }
    if not output["categories"]:
        raise ValueError(f"no coco_pope_*.json files in {annotation_dir}")
    return output


def _mme_ground_truth(root: Path) -> dict[tuple[str, str, str], str]:
    result = {}
    for category_dir in root.iterdir():
        if not category_dir.is_dir():
            continue
        qa_path = category_dir / "questions_answers_YN" if (category_dir / "images").is_dir() else category_dir
        if not qa_path.is_dir():
            continue
        for path in qa_path.glob("*.txt"):
            for line in path.read_text(encoding="utf-8").splitlines():
                question, answer = line.split("\t")
                result[(category_dir.name, path.name, question)] = answer
    return result


def _mme_answer(question: dict[str, Any], ground_truth: dict[tuple[str, str, str], str], category: str, file: str) -> str:
    for key in ("label", "answer"):
        if key in question:
            return str(question[key]).lower()
    text = question["text"].replace("Answer the question using a single word or phrase.", "").strip()
    if "Please answer yes or no." not in text:
        text += " Please answer yes or no."
    if (category, file, text) not in ground_truth:
        text = text.replace(" Please answer yes or no.", "  Please answer yes or no.")
    return ground_truth[(category, file, text)].lower()


def _mme_prediction(text: str) -> str:
    text = text.lower().strip()
    if text in {"yes", "no"}:
        return text
    prefix = text[:4]
    if "yes" in prefix:
        return "yes"
    if "no" in prefix:
        return "no"
    return "other"


def score_mme(question_file: str | Path, result_file: str | Path, annotation_dir: str | Path) -> dict[str, Any]:
    questions = read_jsonl(question_file)
    predictions = {str(row["question_id"]): _mme_prediction(row["text"]) for row in read_jsonl(result_file)}
    ground_truth = _mme_ground_truth(Path(annotation_dir))
    groups: dict[str, dict[str, list[tuple[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for question in questions:
        question_id = str(question["question_id"])
        category = question_id.split("/")[0]
        file = question_id.split("/")[-1].split(".")[0] + ".txt"
        answer = _mme_answer(question, ground_truth, category, file)
        if answer not in {"yes", "no"}:
            raise ValueError(f"MME ground truth is not yes/no: {answer}")
        prediction = predictions.get(question_id, "other")
        groups[category][file].append((answer, prediction))
    task_scores = {}
    for category, images in groups.items():
        pairs = [pair for items in images.values() for pair in items]
        if any(len(items) != 2 for items in images.values()):
            raise ValueError(f"MME {category}: every image must have exactly two questions")
        accuracy = _safe_divide(sum(answer == prediction for answer, prediction in pairs), len(pairs))
        accuracy_plus = _safe_divide(
            sum(all(answer == prediction for answer, prediction in items) for items in images.values()), len(images)
        )
        task_scores[category] = 100 * (accuracy + accuracy_plus)
    output: dict[str, Any] = {"tasks": task_scores, "complete": set(task_scores) == set(MME_TASKS["perception"] + MME_TASKS["cognition"])}
    for group, tasks in MME_TASKS.items():
        output[group] = math.fsum(task_scores.get(task, 0.0) for task in tasks)
    return output


def _extract_mmbench_choice(text: str, options: list[str]) -> str | None:
    stripped = text.strip()
    match = re.match(r"^\(?([A-Da-d])\)?(?:\s|[).,;:!?]|$)", stripped)
    if match:
        return match.group(1).upper()
    for index, option in enumerate(options):
        if stripped.casefold() == str(option).strip().casefold():
            return chr(ord("A") + index)
    match = re.search(r"(?:answer|option|choice)\s*(?:is|:)?\s*\(?([A-Da-d])\)?\b", stripped, re.I)
    return match.group(1).upper() if match else None


def score_mmbench(annotation_file: str | Path, result_file: str | Path) -> dict[str, Any]:
    with Path(annotation_file).open(encoding="utf-8", newline="") as handle:
        annotations = {str(row["index"]): row for row in csv.DictReader(handle, delimiter="\t")}
    predictions = {str(row["question_id"]): row for row in read_jsonl(result_file)}
    correct = missing = unparsed = 0
    for index, annotation in annotations.items():
        prediction = predictions.get(index)
        if prediction is None:
            missing += 1
            continue
        options = [annotation[chr(ord("A") + i)] for i in range(4) if annotation.get(chr(ord("A") + i))]
        choice = _extract_mmbench_choice(prediction["text"], options)
        if choice is None:
            unparsed += 1
        elif choice == str(annotation["answer"]).strip().upper():
            correct += 1
    total = len(annotations)
    return {
        "samples": total, "accuracy": 100 * _safe_divide(correct, total), "correct": correct,
        "missing": missing, "unparsed": unparsed,
        "protocol": "offline deterministic option extraction; not an online MMBench submission",
    }


def score_seed(annotation_file: str | Path, result_file: str | Path) -> dict[str, Any]:
    data = json.loads(Path(annotation_file).read_text(encoding="utf-8"))
    rows = read_jsonl(result_file)
    predictions = {str(row["question_id"]): row.get("text", "") for row in rows}
    type_names = {identifier: name for name, identifier in data["question_type"].items()}
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for question in data["questions"]:
        data_type = question.get("data_type", "unknown")
        type_id = question["question_type_id"]
        prediction = predictions.get(str(question["question_id"]), "")
        correct = prediction == question.get("answer")
        for key in ("all", data_type, str(type_id)):
            counts[key][0] += int(correct)
            counts[key][1] += 1
    def accuracy(key: str) -> float:
        correct, total = counts[key]
        return 100 * _safe_divide(correct, total)
    return {
        "all": accuracy("all"), "img": accuracy("image"), "vid": accuracy("video"),
        "question_types": {type_names.get(int(key), key): accuracy(key) for key in counts if key.isdigit()},
    }


def score_general(benchmark: str, **kwargs: Any) -> dict[str, Any]:
    if benchmark == "pope":
        return score_pope(kwargs["question_file"], kwargs["result_file"], kwargs["annotation_dir"])
    if benchmark == "mme":
        return score_mme(kwargs["question_file"], kwargs["result_file"], kwargs["annotation_dir"])
    if benchmark == "mmbench":
        return score_mmbench(kwargs["annotation_file"], kwargs["result_file"])
    if benchmark == "seed":
        return score_seed(kwargs["annotation_file"], kwargs["result_file"])
    raise ValueError(f"unknown general benchmark: {benchmark}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    merge = sub.add_parser("merge")
    merge.add_argument("--inputs", nargs="+", required=True)
    merge.add_argument("--output", required=True)
    score = sub.add_parser("score")
    score.add_argument("--benchmark", choices=("pope", "mme", "mmbench", "seed"), required=True)
    score.add_argument("--question-file")
    score.add_argument("--result-file", required=True)
    score.add_argument("--annotation-file")
    score.add_argument("--annotation-dir")
    score.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "merge":
        count = merge_jsonl(args.inputs, args.output)
        result = {"output": args.output, "samples": count}
    else:
        values = {key: value for key, value in vars(args).items() if key != "benchmark"}
        result = score_general(args.benchmark, **values)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
