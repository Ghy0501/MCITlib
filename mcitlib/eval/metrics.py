# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Paper-compatible MFT/MFN/MAA/BWT from a lower-triangular accuracy matrix."""
from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from typing import Any, Sequence


def _validate_matrix(matrix: Sequence[Sequence[Any]]) -> list[list[float]]:
    if not matrix:
        raise ValueError("accuracy matrix is empty")
    result: list[list[float]] = []
    for index, row in enumerate(matrix):
        if len(row) != index + 1:
            raise ValueError(
                f"matrix must be lower triangular: row {index} has {len(row)} values, expected {index + 1}"
            )
        converted = [float(value) for value in row]
        if not all(math.isfinite(value) for value in converted):
            raise ValueError(f"matrix row {index} contains a non-finite value")
        result.append(converted)
    return result


def continual_metrics(matrix: Sequence[Sequence[Any]]) -> dict[str, float]:
    """Compute the four metrics using percentages and the paper's matrix layout.

    A[i][j] is measured after learning task i; rows and columns are zero-based.
    """
    values = _validate_matrix(matrix)
    task_count = len(values)
    final = values[-1]
    bwt = 0.0 if task_count == 1 else math.fsum(
        final[j] - values[j][j] for j in range(task_count - 1)
    ) / (task_count - 1)
    return {
        "mft": math.fsum(values[index][index] for index in range(task_count)) / task_count,
        "mfn": math.fsum(final) / task_count,
        "maa": math.fsum(math.fsum(row) / len(row) for row in values) / task_count,
        "bwt": bwt,
    }


def _parse_result_text(path: Path) -> float:
    text = path.read_text(encoding="utf-8")
    for label in ("Accuracy", "Average"):
        match = re.search(rf"(?m)^{re.escape(label)}:\s*(-?\d+(?:\.\d+)?)\s*%?\s*$", text)
        if match:
            return float(match.group(1))
    match = re.search(r"Final_score_dict:\s*(\{.*\})", text)
    if match:
        values = ast.literal_eval(match.group(1))
        if "Final Score Norm" in values:
            return float(values["Final Score Norm"])
    raise ValueError(f"no supported accuracy value in {path}")


def matrix_from_results(root: str | Path, tasks: Sequence[str] | None = None) -> tuple[list[list[float]], list[str]]:
    """Recover A[i][j] from CoIN result directories, requiring every j <= i."""
    root = Path(root)
    for filename in ("matrix.json", "accuracy_matrix.json"):
        candidate = root / filename
        if candidate.is_file():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            matrix = payload["matrix"] if isinstance(payload, dict) else payload
            return _validate_matrix(matrix), list(payload.get("tasks", [])) if isinstance(payload, dict) else []
    search_root = root / "each_dataset" if (root / "each_dataset").is_dir() else root
    cells: dict[tuple[int, int], float] = {}
    task_names = list(tasks or [])
    for result in sorted(search_root.rglob("Result.text")):
        parts = result.parts
        stage_match = next((re.fullmatch(r".+-task(\d+)", part) for part in parts if re.fullmatch(r".+-task\d+", part)), None)
        if stage_match is None:
            continue
        stage = int(stage_match.group(1))
        try:
            dataset_index = parts.index("each_dataset") + 1
            task_name = parts[dataset_index]
        except (ValueError, IndexError):
            if tasks is None:
                raise ValueError(f"cannot infer task name from {result}; pass tasks explicitly")
            task_name = next((name for name in tasks if name in parts), None)
            if task_name is None:
                raise ValueError(f"cannot infer task name from {result}")
        if task_name not in task_names:
            task_names.append(task_name)
        column = task_names.index(task_name)
        key = (stage - 1, column)
        value = _parse_result_text(result)
        if key in cells and cells[key] != value:
            raise ValueError(f"conflicting accuracies for matrix cell {key}: {cells[key]} vs {value}")
        cells[key] = value
    if not cells:
        raise FileNotFoundError(f"no CoIN Result.text or matrix JSON found under {root}")
    task_count = max(stage for stage, _ in cells) + 1
    if tasks is not None and len(tasks) != task_count:
        raise ValueError(f"expected {len(tasks)} tasks but results contain {task_count} stages")
    if not task_names:
        raise ValueError("no task names found")
    if len(task_names) != task_count:
        raise ValueError(f"found {len(task_names)} task names for {task_count} stages")
    matrix = []
    missing = []
    for row in range(task_count):
        values = []
        for column in range(row + 1):
            if (row, column) not in cells:
                missing.append([row, column])
            else:
                values.append(cells[row, column])
        matrix.append(values)
    if missing:
        raise ValueError(f"incomplete lower-triangular accuracy matrix; missing cells: {missing}")
    return _validate_matrix(matrix), task_names


def metrics_from_path(path: str | Path, tasks: Sequence[str] | None = None) -> dict[str, Any]:
    path = Path(path)
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        matrix = payload["matrix"] if isinstance(payload, dict) else payload
        task_names = list(payload.get("tasks", tasks or [])) if isinstance(payload, dict) else list(tasks or [])
        matrix = _validate_matrix(matrix)
    elif path.is_dir():
        if tasks is None:
            from ..registry import BENCHMARKS

            path_text = str(path).lower()
            benchmark = next((item for item in BENCHMARKS if item.id in path_text or item.v3_name.lower() in path_text), None)
            if benchmark is None:
                raise ValueError("cannot infer task order; pass --benchmark or use matrix.json with tasks")
            tasks = benchmark.tasks
        matrix, task_names = matrix_from_results(path, tasks)
    else:
        raise FileNotFoundError(f"metrics input does not exist: {path}")
    return {"source": str(path), "tasks": task_names, "matrix": matrix, "metrics": continual_metrics(matrix)}
