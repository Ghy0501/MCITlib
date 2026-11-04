# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Metric formulas validated against the paper's actual LLaVA matrices."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mcitlib.eval.metrics import continual_metrics, matrix_from_results, metrics_from_path

# The paper's Table 1 and detailed matrices disagree on these source values.
# Each pair is (rounded value recomputed from the matrix, printed Table 1 value).
PAPER_DISCREPANCIES = {
    ("ucit", "clmoe", "mft"): (71.45, 71.46),
    ("mllm_dcl", "olora", "bwt"): (-12.03, -12.53),
    ("mllm_dcl", "moelora", "bwt"): (-12.70, -12.71),
    ("mllm_dcl", "clmoe", "bwt"): (-13.96, -13.97),
    ("mllm_acl", "lora_ft", "mft"): (41.27, 41.28),
    ("mllm_acl", "olora", "mfn"): (37.16, 37.17),
    ("mllm_acl", "sefe", "mft"): (40.98, 40.99),
}


def test_all_paper_matrices_reproduce_table1_except_registered_source_conflicts(repo_root: Path) -> None:
    fixture = json.loads((repo_root / "tests/fixtures/paper_matrices/llava.json").read_text())
    checked = 0
    discrepancies = set()
    for benchmark, values in fixture["benchmarks"].items():
        assert len(values["methods"]) == 8
        for method, item in values["methods"].items():
            actual = continual_metrics(item["matrix"])
            for metric, expected in item["expected"].items():
                key = (benchmark, method, metric)
                rounded = round(actual[metric], 2)
                if key in PAPER_DISCREPANCIES:
                    matrix_value, table1_value = PAPER_DISCREPANCIES[key]
                    assert rounded == matrix_value
                    assert expected == table1_value
                    discrepancies.add(key)
                else:
                    assert rounded == expected, f"{benchmark}/{method}/{metric}"
            checked += 1
    assert checked == 24
    assert discrepancies == set(PAPER_DISCREPANCIES)


def test_lora_ucit_reference_values_and_single_task_bwt() -> None:
    matrix = [[92.0], [90.63, 91.03]]
    result = continual_metrics(matrix)
    assert result["mft"] == pytest.approx(91.515)
    assert result["mfn"] == pytest.approx(90.83)
    assert result["maa"] == pytest.approx(91.415)
    assert result["bwt"] == pytest.approx(-1.37)
    assert continual_metrics([[50]]) == {"mft": 50, "mfn": 50, "maa": 50, "bwt": 0}


def test_coin_result_directories_form_matrix_and_reject_missing_cells(tmp_path: Path) -> None:
    root = tmp_path / "results/UCIT"
    values = (("A", 1, "Accuracy: 80.00%\n"), ("B", 2, "Average: 90.00\n"), ("A", 2, "Accuracy: 70.00%\n"))
    for task, stage, text in values:
        directory = root / "each_dataset" / task / f"Method-task{stage}"
        directory.mkdir(parents=True)
        (directory / "Result.text").write_text(text)
    matrix, tasks = matrix_from_results(root, ["A", "B"])
    assert tasks == ["A", "B"]
    assert matrix == [[80.0], [70.0, 90.0]]
    assert continual_metrics(matrix)["bwt"] == -10
    (root / "each_dataset/A/Method-task2/Result.text").unlink()
    with pytest.raises(ValueError, match="missing cells"):
        matrix_from_results(root, ["A", "B"])


def test_metrics_json_input_and_invalid_shapes(tmp_path: Path) -> None:
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps({"tasks": ["A"], "matrix": [[75]]}))
    result = metrics_from_path(path)
    assert result["tasks"] == ["A"]
    assert result["metrics"]["mfn"] == 75
    with pytest.raises(ValueError, match="lower triangular"):
        continual_metrics([[1, 2]])
    with pytest.raises(ValueError, match="non-finite"):
        continual_metrics([[float("nan")]])


def test_metrics_cli_writes_json(tmp_path: Path, capsys) -> None:
    from mcitlib.cli import main

    matrix = tmp_path / "matrix.json"
    output = tmp_path / "out/metrics.json"
    matrix.write_text(json.dumps({"tasks": ["A"], "matrix": [[75]]}))
    assert main(["metrics", "--run-dir", str(matrix), "--output", str(output)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["metrics"]["mfn"] == 75
    assert json.loads(output.read_text()) == printed
