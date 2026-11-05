# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""The single path-resolution entry point for MCITlib."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from .config.schema import PathSet


def repo_root() -> Path:
    """Checkout that contains this package (``pip install -e .`` is required)."""
    root = Path(__file__).resolve().parents[1]
    if not (root / "third_party").is_dir():
        raise RuntimeError(
            f"{root} has no third_party/ directory; install MCITlib from a checkout with `pip install -e .`"
        )
    return root


def vendored_pythonpath(root: str | Path | None = None) -> list[str]:
    """Import roots every MCITlib process needs, highest priority first.

    The vendored LLaVA and peft must shadow any ``llava``/``peft`` that an older
    v3 editable install left in the environment; ``doctor`` verifies the result.
    """
    base = Path(root).resolve() if root else repo_root()
    return [str(base), str(base / "third_party" / "LLaVA"), str(base / "third_party" / "peft")]


def merge_pythonpath(*groups: str | None, root: str | Path | None = None) -> str:
    """Vendored roots first, then every other entry once, in order."""
    entries = list(vendored_pythonpath(root))
    for group in groups:
        for item in (group or "").split(os.pathsep):
            if item and item not in entries:
                entries.append(item)
    return os.pathsep.join(entries)


def bootstrap_sys_path(root: str | Path | None = None) -> None:
    """Put the vendored roots in front of ``sys.path`` for the current process."""
    import sys

    for item in reversed(vendored_pythonpath(root)):
        if item in sys.path:
            sys.path.remove(item)
        sys.path.insert(0, item)


def process_env(command_env: dict[str, str], root: str | Path | None = None) -> dict[str, str]:
    """Environment for a child process: ours, the command's, and a safe PYTHONPATH."""
    env = os.environ.copy()
    env.update(command_env)
    env["PYTHONPATH"] = merge_pythonpath(command_env.get("PYTHONPATH"), os.environ.get("PYTHONPATH"), root=root)
    return env


class PathResolver:
    def __init__(self, paths: PathSet):
        self.paths = paths

    @classmethod
    def from_file(cls, path: str | Path | None = None, repo_root: str | Path | None = None) -> "PathResolver":
        root = Path(repo_root or Path.cwd()).expanduser().resolve()
        config_path = Path(path).expanduser() if path else root / "configs" / "paths.yaml"
        values: dict[str, Any] = {}
        if config_path.is_file():
            with config_path.open(encoding="utf-8") as handle:
                values = yaml.safe_load(handle) or {}
        elif path is not None:
            raise FileNotFoundError(f"path configuration not found: {config_path}")

        def resolve_root(name: str, default: str) -> Path:
            env_name = f"MCITLIB_{name.upper()}"
            raw = os.environ.get(env_name, values.get(name, default))
            candidate = Path(os.path.expanduser(str(raw)))
            return (candidate if candidate.is_absolute() else root / candidate).resolve()

        return cls(PathSet(
            model_root=resolve_root("model_root", "./models"),
            data_root=resolve_root("data_root", "./data"),
            ckpt_root=resolve_root("ckpt_root", "./ckpts"),
            repo_root=root,
        ))

    def resolve_templates(self, value: str) -> str:
        """Resolve path placeholders in argv/cwd without invoking a shell."""
        replacements = {
            "${model_root}": str(self.paths.model_root),
            "${data_root}": str(self.paths.data_root),
            "${ckpt_root}": str(self.paths.ckpt_root),
            "${repo_root}": str(self.paths.repo_root),
            "/your_model_path": str(self.paths.model_root),
            "/your_data_path": str(self.paths.data_root),
            "/your_ckpts_path": str(self.paths.ckpt_root),
        }
        for source, target in replacements.items():
            value = value.replace(source, target)
        return value

    def resolve(self, value: str | Path, kind: str = "data") -> Path:
        raw = os.path.expanduser(str(value))
        prefixes = {
            "/your_model_path": self.paths.model_root,
            "/your_data_path": self.paths.data_root,
            "/your_ckpts_path": self.paths.ckpt_root,
        }
        for prefix, root in prefixes.items():
            if raw == prefix or raw.startswith(prefix + "/"):
                return (root / raw[len(prefix):].lstrip("/")).resolve()
        candidate = Path(raw)
        if candidate.is_absolute():
            return candidate.resolve()
        roots = {"model": self.paths.model_root, "data": self.paths.data_root, "ckpt": self.paths.ckpt_root}
        return (roots.get(kind, self.paths.repo_root) / candidate).resolve()

    def run_dir(self, benchmark: str, method: str, task_id: int | None = None) -> Path:
        base = self.paths.ckpt_root / benchmark / "llava" / method
        return base if task_id is None else base / f"task{task_id}"

    def eval_dir(self, benchmark: str, method: str, after_task: int) -> Path:
        return self.run_dir(benchmark, method, after_task) / "eval" / f"after_task{after_task}"
