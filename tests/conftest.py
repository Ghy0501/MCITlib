# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Shared paths for CPU-only MCITlib tests."""
from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def v3_root(repo_root: Path) -> Path:
    value = os.environ.get("MCITLIB_V3_ROOT")
    path = Path(value).expanduser().resolve() if value else repo_root.parent / "MCITlib_v3_ref"
    if not (path / "LLaVA").is_dir():
        pytest.skip(f"v3 reference worktree not found: {path}")
    return path
