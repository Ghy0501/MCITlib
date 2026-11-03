# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""NB-4 boundary tests for checkpoint cleanup."""
from __future__ import annotations

from pathlib import Path

from mcitlib.backbones.llava.cleanup import remove_checkpoint_dirs


def test_cleanup_only_removes_direct_checkpoint_children(tmp_path: Path) -> None:
    direct = tmp_path / "checkpoint-10"
    nested = direct / "checkpoint-nested"
    keep = tmp_path / "keep"
    nested.mkdir(parents=True)
    keep.mkdir()
    (keep / "file").write_text("keep", encoding="utf-8")
    removed = remove_checkpoint_dirs(tmp_path)
    assert removed == [direct]
    assert not direct.exists()
    assert keep.exists()


def test_cleanup_skips_symlinks_and_missing_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "run"
    root.mkdir()
    (root / "checkpoint-link").symlink_to(outside, target_is_directory=True)
    assert remove_checkpoint_dirs(root) == []
    assert outside.exists()
    assert remove_checkpoint_dirs(tmp_path / "missing") == []
