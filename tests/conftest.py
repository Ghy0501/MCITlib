# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Shared paths for CPU-only MCITlib tests."""
from __future__ import annotations

import os
from importlib.metadata import version
from pathlib import Path

import pytest


def requires_method_abi(method: str):
    """Select the declared Transformers ABI without deselecting the default CI job."""
    from mcitlib.doctor import _pinned
    from mcitlib.registry import get_method

    profile = get_method(method).env
    pin = _pinned(Path(__file__).resolve().parents[1] / "envs" / f"{profile}.txt")["transformers"]
    return pytest.mark.skipif(version("transformers") != pin, reason=f"{method} requires {profile} (Transformers {pin})")


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


def reference_env(implementation: str, pythonpath: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for an equivalence-test subprocess.

    ``v3`` subprocesses run the original v3 code and must import *v3's* ``llava``/``peft`` from
    ``pythonpath``, so ``import mcitlib`` must not put the vendored copies in front of them.
    ``v4`` subprocesses behave like a real run.
    """
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    env["PYTHONPATH"] = pythonpath
    if implementation == "v3":
        env["MCITLIB_NO_VENDORED_PATH"] = "1"
    else:
        env.pop("MCITLIB_NO_VENDORED_PATH", None)
    env.update(extra or {})
    return env
