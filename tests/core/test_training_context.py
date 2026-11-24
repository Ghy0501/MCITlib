# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Training-task identity does not depend on models, datasets, or GPUs."""
from __future__ import annotations

from pathlib import Path

import pytest

from mcitlib.backbones.llava.train_context import training_task_id
from mcitlib.config.loader import read_yaml


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("UCIT_ImgNet", 1), ("UCIT_Arxiv", 2), ("UCIT_VizWiz", 3),
        ("UCIT_Icon", 4), ("UCIT_CLEVR", 5), ("UCIT_Flickr30k", 6),
        ("DCL_RS", 1), ("DCL_Med", 2), ("DCL_AD", 3), ("DCL_Sci", 4), ("DCL_Fin", 5),
        ("ACL_OCR", 1), ("ACL_Math", 2), ("ACL_VP", 3), ("ACL_APP", 4),
    ],
)
def test_clmoe_recorded_labels_resolve_to_one_based_task_ids(label: str, expected: int) -> None:
    assert training_task_id("clmoe", 0, label) == expected


def test_non_clmoe_retains_zero_based_cur_task_as_algorithm_input() -> None:
    assert training_task_id("olora", 0, "") == 1
    assert training_task_id("hide", 3, "unrelated string") == 4
    assert training_task_id("clmoe", 2, "") == 3
    assert training_task_id("clmoe", 0, "2") == 2


def test_unknown_and_ambiguous_clmoe_labels_fail(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="0 matches"):
        training_task_id("clmoe", 0, "UNKNOWN")

    methods = tmp_path / "methods"
    methods.mkdir()
    (methods / "clmoe.yaml").write_text(
        "overrides:\n  llava:\n    ucit:\n"
        "      task_1: {train_config: {task: repeated}}\n"
        "      task_2: {train_config: {task: repeated}}\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="2 matches"):
        training_task_id("clmoe", 0, "repeated", config_root=tmp_path)


def test_expected_labels_are_all_covered_by_registered_config() -> None:
    config = read_yaml(Path(__file__).resolve().parents[2] / "configs/methods/clmoe.yaml")
    entries = [
        (item["train_config"]["task"], int(key.removeprefix("task_")))
        for tasks in config["overrides"]["llava"].values()
        for key, item in tasks.items() if key.startswith("task_")
    ]
    assert len(entries) == 15
    assert all(training_task_id("clmoe", 0, name) == task_id for name, task_id in entries)
