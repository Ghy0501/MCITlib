# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Check checkout relocation and avoid packaging unrelated development trees."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def test_copied_checkout_and_wheel_do_not_pick_up_other_packages(repo_root: Path, tmp_path: Path) -> None:
    copy = tmp_path / "independent-project"
    copy.mkdir()
    for name in ("mcitlib", "configs", "third_party"):
        shutil.copytree(repo_root / name, copy / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("README.md", "pyproject.toml"):
        shutil.copy2(repo_root / name, copy / name)
    unrelated = copy / "mcitlib_extra"
    unrelated.mkdir()
    (unrelated / "__init__.py").write_text("# unrelated package\n", encoding="utf-8")
    workspace = copy / "mcitlib/.workspaces/snapshot/mcitlib"
    workspace.mkdir(parents=True)
    (workspace / "__init__.py").write_text("# development snapshot\n", encoding="utf-8")

    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH" and not key.startswith("MCITLIB_")}
    env.update(PYTHONPATH=str(copy), PYTHONDONTWRITEBYTECODE="1")
    where = subprocess.run([sys.executable, "-c", "import mcitlib; print(mcitlib.__file__)"], cwd=tmp_path,
                           env=env, check=True, capture_output=True, text=True)
    assert Path(where.stdout.strip()).resolve() == copy / "mcitlib/__init__.py"
    executable = str(Path(sys.executable).parent / "mcitlib")
    command = subprocess.run([executable, "train", "--method", "lora_ft", "--benchmark", "ucit", "--task", "1",
                              "--dry-run"], cwd=tmp_path, env=env, check=True, capture_output=True, text=True)
    assert str(copy / "configs/deepspeed/zero2.json") in command.stdout
    assert str(repo_root / "configs/deepspeed/zero2.json") not in command.stdout

    wheel_dir = tmp_path / "wheel"
    build_env = dict(env)
    build_env.pop("PYTHONPATH")
    subprocess.run([sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(wheel_dir)],
                   cwd=copy, env=build_env, check=True, capture_output=True, text=True)
    wheel, = wheel_dir.glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    assert "mcitlib/cli.py" in names
    assert not any(name.startswith("mcitlib_extra/") or ".workspaces" in name for name in names)
