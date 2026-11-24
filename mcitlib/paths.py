# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""The single path-resolution entry point for MCITlib."""
from __future__ import annotations

import json
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


def _default_repo_root() -> Path:
    return repo_root()


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


def managed_path(path: str | Path, root: str | Path, *, within: str | Path | None = None,
                 label: str = "managed path") -> Path:
    """Resolve a project-owned path and reject traversal or symlink escapes.

    Relative paths are always relative to the project, never the caller's cwd.
    """
    project = Path(root).expanduser().resolve()
    candidate = Path(path).expanduser()
    candidate = (candidate if candidate.is_absolute() else project / candidate).resolve()
    boundary = Path(within).expanduser() if within is not None else project
    boundary = (boundary if boundary.is_absolute() else project / boundary).resolve()
    if (boundary != project and project not in boundary.parents) or (
        candidate != boundary and boundary not in candidate.parents
    ):
        raise ValueError(f"{label} must be inside {boundary} in project {project}: {candidate}")
    return candidate


class PathResolver:
    def __init__(self, paths: PathSet, mappings: dict[str, list[tuple[str, str]]] | None = None,
                 *, octopus_aux_root: Path | None = None):
        self.paths = paths
        self.mappings = mappings or {}
        self.octopus_aux_root = octopus_aux_root
        self._validate_mappings()

    def _validate_mappings(self) -> None:
        output_roots = (str(self.paths.ckpt_root), str(self.results_root), str(self.logs_root))
        for scope in self.mappings:
            if scope not in {"common", "ucit", "mllm_dcl", "mllm_acl", "general"}:
                raise ValueError(f"unknown path mapping scope: {scope}")
        for scope in {"ucit", "mllm_dcl", "mllm_acl", "general"}:
            seen: dict[str, str] = {}
            for source, target in self.mappings.get("common", []) + self.mappings.get(scope, []):
                if source in seen and seen[source] != target:
                    raise ValueError(f"conflicting path mappings for {source} in {scope}: {seen[source]} and {target}")
                seen[source] = target
                for value in (source, target):
                    if not os.path.isabs(value) or value == "/":
                        raise ValueError(f"path mapping must be an absolute, non-root path: {value}")
                    if any(_paths_overlap(value, root) or _paths_overlap(os.path.realpath(value), os.path.realpath(root))
                           for root in output_roots):
                        raise ValueError(f"path mapping overlaps checkpoint/results root or logs root: {value}")

    @classmethod
    def from_file(cls, path: str | Path | None = None, repo_root: str | Path | None = None) -> "PathResolver":
        root = Path(repo_root).expanduser().resolve() if repo_root is not None else _default_repo_root()
        config_path = Path(path).expanduser() if path else root / "configs" / "paths.yaml"
        if not config_path.is_absolute():
            config_path = root / config_path
        values: dict[str, Any] = {}
        if config_path.is_file():
            with config_path.open(encoding="utf-8") as handle:
                values = yaml.safe_load(handle) or {}
        elif path is not None:
            raise FileNotFoundError(f"path configuration not found: {config_path}")
        if not isinstance(values, dict):
            raise ValueError(f"path configuration must be a mapping: {config_path}")

        def resolve_root(name: str, default: str) -> Path:
            env_name = f"MCITLIB_{name.upper()}"
            raw = os.environ.get(env_name, values.get(name, default))
            candidate = Path(os.path.expanduser(str(raw)))
            return (candidate if candidate.is_absolute() else root / candidate).resolve()

        ckpt_root = resolve_root("ckpt_root", "./ckpts")
        logs_root = managed_path(resolve_root("logs_root", "./logs"), root, label="logs_root")
        results_root = resolve_root("results_root", "./results")
        if "results_root" not in values and "MCITLIB_RESULTS_ROOT" not in os.environ:
            results_root = managed_path(results_root, root, label="default results_root")
        paths = PathSet(
            model_root=resolve_root("model_root", "./models"),
            data_root=resolve_root("data_root", "./data"),
            ckpt_root=ckpt_root,
            repo_root=root,
            results_root=results_root,
            logs_root=logs_root,
        )
        resolver = cls(paths)
        raw_mappings = values.get("path_mappings", {})
        if not isinstance(raw_mappings, dict):
            raise ValueError("path_mappings must be a mapping of scopes to lists")
        mappings: dict[str, list[tuple[str, str]]] = {}
        for scope, entries in raw_mappings.items():
            if not isinstance(entries, list):
                raise ValueError(f"path_mappings.{scope} must be a list")
            mappings[scope] = []
            for entry in entries:
                if not isinstance(entry, dict) or set(entry) != {"from", "to"}:
                    raise ValueError(f"path_mappings.{scope} entries must contain exactly 'from' and 'to'")
                if not all(isinstance(entry[key], str) and entry[key] for key in ("from", "to")):
                    raise ValueError(f"path_mappings.{scope} paths must be non-empty strings")
                source, target = (os.path.normpath(os.path.expanduser(resolver.resolve_templates(entry[key])))
                                  for key in ("from", "to"))
                mappings[scope].append((source, target))
        auxiliary = values.get("octopus_aux_root")
        if auxiliary is not None:
            if not isinstance(auxiliary, str) or not auxiliary.strip():
                raise ValueError("octopus_aux_root must be a nonempty path")
            auxiliary = managed_path(resolver.resolve_templates(auxiliary), root, within=logs_root,
                                     label="octopus_aux_root")
        return cls(paths, mappings, octopus_aux_root=auxiliary)

    def resolve_templates(self, value: str, benchmark: str | None = None) -> str:
        """Resolve path placeholders in argv/cwd without invoking a shell."""
        replacements = {
            "${model_root}": str(self.paths.model_root),
            "${data_root}": str(self.paths.data_root),
            "${ckpt_root}": str(self.paths.ckpt_root),
            "${repo_root}": str(self.paths.repo_root),
            "${results_root}": str(self.results_root),
            "${logs_root}": str(self.logs_root),
            "/your_model_path": str(self.paths.model_root),
            "/your_data_path": str(self.paths.data_root),
            "/your_ckpts_path": str(self.paths.ckpt_root),
        }
        for source, target in replacements.items():
            value = value.replace(source, target)
        candidates = self.mappings.get("common", []) + self.mappings.get(benchmark or "", [])
        for source, target in sorted(candidates, key=lambda item: len(item[0]), reverse=True):
            if value == source or value.startswith(source + "/"):
                return target + value[len(source):]
        return value

    def resolve(self, value: str | Path, kind: str = "data", benchmark: str | None = None) -> Path:
        raw = os.path.expanduser(self.resolve_templates(str(value), benchmark))
        candidate = Path(raw)
        if candidate.is_absolute():
            return candidate.resolve()
        roots = {"model": self.paths.model_root, "data": self.paths.data_root, "ckpt": self.paths.ckpt_root}
        joined = str(roots.get(kind, self.paths.repo_root) / candidate)
        return Path(self.resolve_templates(joined, benchmark)).resolve()

    def external_roots(self, benchmark: str) -> tuple[str, ...]:
        """Resolved user inputs, including targets outside the default roots."""
        return tuple(dict.fromkeys([str(self.paths.model_root), str(self.paths.data_root)] + [
            target for _, target in self.mappings.get("common", []) + self.mappings.get(benchmark, [])
        ]))

    def context(self, benchmark: str, *, include_octopus_aux: bool = True) -> str:
        """Serialize the selected paths for extra entry points that read JSON configs."""
        context = {"paths": {name: str(value) for name, value in vars(self.paths).items()},
                   "mappings": self.mappings, "benchmark": benchmark}
        if include_octopus_aux and self.octopus_aux_root is not None:
            context["octopus_aux_root"] = str(self.octopus_aux_root)
        return json.dumps(context)

    @classmethod
    def from_context(cls, value: str) -> tuple["PathResolver", str]:
        context = json.loads(value)
        paths = PathSet(**{name: Path(value) if value != "None" else None
                          for name, value in context["paths"].items()})
        auxiliary = context.get("octopus_aux_root")
        if auxiliary is not None:
            auxiliary = managed_path(auxiliary, paths.repo_root, within=paths.logs_root or paths.repo_root / "logs",
                                     label="octopus_aux_root")
        return cls(paths, context["mappings"], octopus_aux_root=auxiliary), context["benchmark"]

    def method_results_dir(self, method: str, backbone: str = "llava") -> Path:
        """Evaluation outputs are separated by backbone and method."""
        if backbone not in {"llava", "internvl", "videollava", "videollama2"}:
            raise ValueError(f"unknown backbone: {backbone}")
        root = self.results_root if backbone == "llava" else self.results_root / backbone
        return root / method

    @property
    def results_root(self) -> Path:
        return self.paths.results_root or self.paths.repo_root / "results"

    @property
    def logs_root(self) -> Path:
        return managed_path(self.paths.logs_root or self.paths.repo_root / "logs", self.paths.repo_root,
                            label="logs_root")

    def managed_run_dir(self, path: str | Path) -> Path:
        return managed_path(path, self.paths.repo_root, within=self.logs_root, label="run_dir")

    def run_dir(self, benchmark: str, method: str, task_id: int | None = None,
                backbone: str = "llava") -> Path:
        if backbone not in {"llava", "internvl", "videollava", "videollama2"}:
            raise ValueError(f"unknown backbone: {backbone}")
        base = self.logs_root / benchmark / backbone / method
        return self.managed_run_dir(base if task_id is None else base / f"task{task_id}")

    def eval_dir(self, benchmark: str, method: str, after_task: int) -> Path:
        return self.run_dir(benchmark, method, after_task) / "eval" / f"after_task{after_task}"


def _paths_overlap(first: str, second: str) -> bool:
    return first == second or first.startswith(second + "/") or second.startswith(first + "/")
