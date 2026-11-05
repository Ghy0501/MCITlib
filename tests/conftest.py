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
        message = f"v3 reference worktree not found: {path}"
        if os.environ.get("MCITLIB_REQUIRE_V3") == "1":
            pytest.fail(message)
        pytest.skip(message)
    return path


def require_module(name: str):
    """Import ``name`` or skip; with MCITLIB_REQUIRE_FULL_ENV=1 (CI) a missing package is a failure."""
    import importlib

    if os.environ.get("MCITLIB_REQUIRE_FULL_ENV") == "1":
        return importlib.import_module(name)
    return pytest.importorskip(name)


def write_deepspeed_stub(directory: "Path") -> None:
    """Import-only deepspeed stand-in, written only when the real package is absent.

    Light CPU environments have no deepspeed; with the real one installed a stub would
    shadow it and break transformers/accelerate imports.
    """
    import importlib.util

    if importlib.util.find_spec("deepspeed") is None:
        (directory / "deepspeed.py").write_text("# import-only stub for CPU tests\n", encoding="utf-8")
