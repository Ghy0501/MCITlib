# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Offline checks for the evaluation entry's target and argv forwarding."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from mcitlib.backbones.llava.eval_runners import entry


def test_eval_entry_preserves_target_arguments(tmp_path: Path, monkeypatch, capsys) -> None:
    (tmp_path / "offline_eval_target.py").write_text(
        "import json, sys\nprint(json.dumps(sys.argv))\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(tmp_path)
    monkeypatch.setenv("MCITLIB_METHOD", "lora_ft")
    monkeypatch.setattr(sys, "argv", ["entry", "offline_eval_target", "--value", "7"])
    entry.main()
    assert json.loads(capsys.readouterr().out) == ["offline_eval_target", "--value", "7"]
