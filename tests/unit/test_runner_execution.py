# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Offline execution, failure-stop, and durable resume tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from mcitlib.config.schema import Command
from mcitlib.runner import execute_commands


def python_command(code: str, task_id: int = 1) -> Command:
    return Command(argv=[sys.executable, "-c", code], task_id=task_id)


def test_execute_in_order_and_resume_from_markers(tmp_path: Path) -> None:
    output = tmp_path / "order.txt"
    commands = [
        python_command(f"from pathlib import Path; Path({str(output)!r}).open('a').write('A')"),
        python_command(f"from pathlib import Path; Path({str(output)!r}).open('a').write('B')"),
    ]
    first = execute_commands(commands, tmp_path / "run", repo_root=tmp_path)
    assert [item["status"] for item in first] == ["completed", "completed"]
    assert output.read_text() == "AB"
    second = execute_commands(commands, tmp_path / "run", repo_root=tmp_path, resume=True)
    assert [item["status"] for item in second] == ["skipped", "skipped"]
    assert output.read_text() == "AB"
    markers = list((tmp_path / "run/.mcitlib/steps").glob("*.json"))
    assert len(markers) == 2
    assert all(json.loads(path.read_text())["status"] == "completed" for path in markers)


def test_failure_stops_sequence_and_resume_keeps_completed_steps(tmp_path: Path) -> None:
    output = tmp_path / "order.txt"
    first = python_command(f"from pathlib import Path; Path({str(output)!r}).write_text('first')")
    failing = python_command("raise RuntimeError('offline failure')")
    last = python_command(f"from pathlib import Path; Path({str(output)!r}).open('a').write('last')")
    with pytest.raises(RuntimeError, match="command 1 failed"):
        execute_commands([first, failing, last], tmp_path / "run", repo_root=tmp_path)
    assert output.read_text() == "first"
    statuses = sorted(path.name for path in (tmp_path / "run/.mcitlib/steps").glob("*.json"))
    assert len(statuses) == 2

    succeeding = python_command("print('fixed')")
    results = execute_commands([first, succeeding, last], tmp_path / "run", repo_root=tmp_path, resume=True)
    assert [item["status"] for item in results] == ["skipped", "completed", "completed"]
    assert output.read_text() == "firstlast"
