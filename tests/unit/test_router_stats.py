# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""NB-2: router diagnostics are optional and context-scoped."""
from __future__ import annotations

from pathlib import Path

import torch

from mcitlib.utils.router_stats import RouterStats


def test_router_stats_default_off(tmp_path: Path) -> None:
    stats = RouterStats(task_id=2, output_dir=tmp_path)
    router = torch.tensor([[[0.1, 0.2, 0.7]]])
    stats.record(router)
    assert list(tmp_path.iterdir()) == []


def test_router_stats_enabled_preserves_top2_count_format(tmp_path: Path) -> None:
    stats = RouterStats(task_id=2, output_dir=tmp_path, enabled=True)
    router = torch.tensor([[[0.5, 0.5, 0.0]], [[0.1, 0.2, 0.7]]])
    stats.record(router)
    lines = (tmp_path / "value_counts_2.txt").read_text(encoding="utf-8").splitlines()
    assert lines == ["-1:2", "1:1", "2:1"]
