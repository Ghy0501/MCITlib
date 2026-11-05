# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Build and execute ordered continual-learning command sequences."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config.loader import deep_merge, load_layered_config
from .config.schema import Command, DataSource, RunConfig, TaskSpec
from .eval.chunks import insert_merge_steps
from .gpus import apply_gpu_assignment
from .methods import create_method
from .paths import process_env, vendored_pythonpath
from .registry import get_backbone, get_benchmark, get_method


def _task(benchmark: Any, task_id: int, name: str, values: dict[str, Any], router: bool = False) -> TaskSpec:
    source = values["source"]
    return TaskSpec(
        benchmark=benchmark.id,
        task_id=task_id,
        name=name,
        model_config=source["model_config"],
        data_config=source["data_config"],
        train_config=source["train_config"],
        eval_config=source.get("eval_config"),
        router=router,
        data=DataSource.from_dict(values["data_config"]),
    )


def _command_from_record(item: dict[str, Any], task_id: int, kind: str, sequence: str) -> Command:
    return Command(
        argv=list(item["argv"]),
        cwd=item.get("cwd", "."),
        env=dict(item.get("env", {})),
        kind=kind,
        task_id=task_id,
        sequence=sequence,
        metadata={"order": item.get("order", -1), "source": "v3-command-dump"},
    )


def _recorded_commands(values: dict[str, Any], task: TaskSpec, include_extra: bool = True, sequence: str = "train") -> list[Command]:
    result: list[Command] = []
    kinds = [("eval_commands", "eval")]
    if include_extra:
        kinds.extend((("extra_commands", "extra"), ("other_commands", "other")))
    for key, kind in kinds:
        for item in values.get(key, []):
            result.append(_command_from_record(item, task.task_id, kind, sequence))
    return result


def _apply_script_aliases(commands: list[Command], method: Any) -> list[Command]:
    from .backbones.llava.eval_runners import resolve_eval_module

    aliases = dict(method.script_aliases)
    entry = "mcitlib.backbones.llava.eval_runners.entry"
    for command in commands:
        command.argv = [aliases.get(value, value) for value in command.argv]
        command.env.setdefault("MCITLIB_METHOD", method.id)
        if method.model_override:
            command.env.setdefault("MCITLIB_MODEL_OVERRIDE", method.model_override)
        if command.kind == "eval" and "-m" in command.argv:
            index = command.argv.index("-m")
            source_module = command.argv[index + 1]
            target = resolve_eval_module(source_module, method.id)
            command.argv = command.argv[:index + 1] + [entry, target] + command.argv[index + 2:]
            command.metadata["eval_target"] = target
            command.metadata["source_eval_module"] = source_module
            command.metadata["eval_module_alias"] = True
    return commands


RESULTS_PREFIXES = ("./results", "results")


def _isolate_results(commands: list[Command], method_id: str) -> None:
    """Give every method its own result folder.

    v3 ran each method in its own directory, so every ``./results/...`` was private.  From the
    repository root they would collide (LoRA-FT and Replay both write ``LoRA-task<k>``) and the
    later method would overwrite or mix with the earlier one's files.  Evaluation output goes
    to ``<results_root>/<method>/...`` instead; the MR-LoRA router extraction scripts get the same
    root through ``MCITLIB_RESULTS_ROOT``.
    """
    root = f"${{results_root}}/{method_id}"

    def rewrite(value: str) -> str:
        for prefix in RESULTS_PREFIXES:
            if value == prefix or value.startswith(prefix + "/"):
                return root + value[len(prefix):]
        return value

    for command in commands:
        command.argv = [rewrite(str(value)) for value in command.argv]
        if any("router_results" in str(value) for value in command.argv):
            command.env["MCITLIB_RESULTS_ROOT"] = root


def _ensure_pythonpath(commands: list[Command], config_root: str | Path) -> None:
    """Every child process imports the vendored llava/peft, not whatever is installed."""
    pythonpath = os.pathsep.join(vendored_pythonpath(Path(config_root).resolve().parent))
    for command in commands:
        command.env.setdefault("PYTHONPATH", pythonpath)


def build_sequence_commands(
    config_root: str | Path,
    backbone_id: str,
    method_id: str,
    benchmark_id: str,
    paths: Any = None,
    task_id: int | None = None,
    only_eval: bool = False,
    sequence: str = "train",
    model_path: str | None = None,
    model_base: str | None = None,
    num_chunks: int | None = None,
    gpus: str | None = None,
    master_port: int | None = None,
) -> list[Command]:
    """Build v3-ordered commands; NB-5 install commands are never emitted."""
    backbone = get_backbone(backbone_id)
    if not backbone.v4_supported:
        raise ValueError(f"backbone is not migrated yet: {backbone_id}")
    method = get_method(method_id)
    benchmark = get_benchmark(benchmark_id)
    if benchmark.id == "general":
        if not only_eval or sequence != "train":
            raise ValueError("general benchmarks are supported only by the eval command")
        from .backbones.llava.general import build_general_commands
        commands = build_general_commands(
            config_root, method.id, task_id, model_path, model_base, num_chunks
        )
        _isolate_results(commands, method.id)
        if paths is not None:
            for command in commands:
                command.argv = [paths.resolve_templates(str(value)) for value in command.argv]
                command.cwd = paths.resolve_templates(command.cwd)
                command.env = {key: paths.resolve_templates(str(value)) for key, value in command.env.items()}
        _ensure_pythonpath(commands, config_root)
        return commands
    if sequence not in {"train", "router"}:
        raise ValueError(f"unknown sequence: {sequence}")
    if sequence == "router" and method.id != "mrlora":
        raise ValueError(f"method {method.id} has no router sequence")
    if backbone_id != "llava":
        raise ValueError(f"unsupported backbone: {backbone_id}")
    from .backbones.llava.adapter import LlavaAdapter
    adapter = LlavaAdapter()
    commands: list[Command] = []
    if not only_eval and task_id in (None, 0, 1):
        base_values = load_layered_config(config_root, backbone_id, benchmark.id, method.id)
        benchmark_values = base_values.get("overrides", {}).get(backbone_id, {}).get(benchmark.id, {})
        commands.extend(_command_from_record(item, 0, item.get("kind", "extra"), sequence)
                        for item in benchmark_values.get("pre_commands", []))
        _apply_script_aliases(commands, method)
    for current_id, task_name in enumerate(benchmark.tasks, start=1):
        if task_id is not None and current_id != task_id:
            continue
        values = load_layered_config(config_root, backbone_id, benchmark.id, method.id, task_id=current_id)
        if sequence == "router":
            router_task = values["router"]["benchmarks"][benchmark.id]["tasks"][f"task_{current_id}"]
            values = deep_merge(values, router_task)
        values = {**values, "repo_root": str(Path(config_root).resolve().parent)}
        task = _task(benchmark, current_id, task_name, values, router=sequence == "router")
        if not only_eval:
            run_cfg = RunConfig(backbone_id, method.id, benchmark.id, task=task, values=values, paths=paths)
            plugin = create_method(method.id, values)
            train = adapter.build_command(run_cfg, plugin, task)
            train.metadata["order"] = values["train_command"].get("order", -1)
            train.sequence = sequence
            commands.append(train)
        if only_eval or not method.separate_train_and_infer:
            recorded = _recorded_commands(values, task, include_extra=not only_eval, sequence=sequence)
            commands.extend(_apply_script_aliases(recorded, method))
    _isolate_results(commands, method.id)
    if paths is not None:
        for command in commands:
            command.argv = [paths.resolve_templates(str(value)) for value in command.argv]
            command.cwd = paths.resolve_templates(command.cwd)
            command.env = {key: paths.resolve_templates(str(value)) for key, value in command.env.items()}
    apply_gpu_assignment(commands, gpus, master_port)
    _ensure_pythonpath(commands, config_root)
    commands.sort(key=lambda command: (command.task_id or 0, command.metadata.get("order", -1)))
    # v3 merged the per-GPU chunk files with shell `cat` before scoring; restore that step.
    return insert_merge_steps(commands)


def _step_identity(command: Command) -> str:
    payload = json.dumps(command.as_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def _write_marker(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _parallel_batches(commands: list[Command]) -> list[list[tuple[int, Command]]]:
    """Group consecutive chunked-inference commands of one dataset so they run side by side.

    v3 started the ``N`` chunks of a dataset in the background (``... &``), one per GPU, and
    ``wait``ed for all of them.  Everything else is strictly sequential.
    """
    batches: list[list[tuple[int, Command]]] = []
    for index, command in enumerate(commands):
        group = command.metadata.get("parallel_group")
        if group and batches and batches[-1][0][1].metadata.get("parallel_group") == group:
            batches[-1].append((index, command))
        else:
            batches.append([(index, command)])
    return batches


class _Step:
    """One running command: process handle, log file and the data for its marker."""

    def __init__(self, index: int, command: Command, repo_root: Path, logs: Path, identity: str):
        argv = list(command.argv)
        if not argv:
            raise ValueError(f"empty command at sequence index {index}")
        if argv[0] in {"python", "python3"}:
            argv[0] = sys.executable
        cwd = Path(command.cwd)
        cwd = cwd if cwd.is_absolute() else repo_root / cwd
        # Vendored llava/peft always come first, whatever the user's environment holds.
        env = process_env(command.env, repo_root)
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
        self.index, self.command, self.identity = index, command, identity
        self.log = logs / f"{index:05d}-{identity}.log"
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.started = datetime.now(timezone.utc).isoformat()
        self.handle = self.log.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )

    def pump(self, prefix: str = "") -> int:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.handle.write(line)
            self.handle.flush()
            sys.stdout.write(prefix + line)
        code = self.process.wait()
        self.handle.close()
        return code


def execute_commands(
    commands: list[Command],
    run_dir: str | Path,
    repo_root: str | Path,
    resume: bool = False,
) -> list[dict[str, Any]]:
    """Execute commands with durable logs and resume markers.

    Steps run one after another, except the chunks of one evaluation dataset, which run in
    parallel (one GPU each, as in v3) and are all waited for before the next step starts.  If any
    member fails the others are allowed to finish, every outcome is recorded, and the run stops.
    """
    from concurrent.futures import ThreadPoolExecutor

    run_dir = Path(run_dir).resolve()
    repo_root = Path(repo_root).resolve()
    markers = run_dir / ".mcitlib" / "steps"
    logs = run_dir / ".mcitlib" / "logs"
    results: list[dict[str, Any]] = []
    for batch in _parallel_batches(commands):
        pending: list[_Step] = []
        for index, command in batch:
            identity = _step_identity(command)
            marker = markers / f"{identity}.json"
            if resume and marker.is_file():
                if json.loads(marker.read_text(encoding="utf-8")).get("status") == "completed":
                    results.append({"index": index, "id": identity, "status": "skipped"})
                    continue
            pending.append(_Step(index, command, repo_root, logs, identity))
        if not pending:
            continue
        if len(pending) == 1:
            codes = [pending[0].pump()]
        else:
            with ThreadPoolExecutor(max_workers=len(pending)) as pool:
                futures = [pool.submit(step.pump, f"[chunk {n}] ") for n, step in enumerate(pending)]
                codes = [future.result() for future in futures]
        failed: list[_Step] = []
        for step, code in zip(pending, codes):
            status = "completed" if code == 0 else "failed"
            _write_marker(markers / f"{step.identity}.json", {
                "id": step.identity, "index": step.index, "status": status,
                "return_code": code, "command": step.command.as_dict(),
                "log": str(step.log), "started_at": step.started,
                "finished_at": datetime.now(timezone.utc).isoformat(),
            })
            results.append({"index": step.index, "id": step.identity, "status": status, "return_code": code})
            if code != 0:
                failed.append(step)
        if failed:
            first = failed[0]
            raise RuntimeError(
                f"command {first.index} failed with exit {codes[pending.index(first)]}; log: {first.log}"
                + (f" (+{len(failed) - 1} more in the same parallel group)" if len(failed) > 1 else "")
            )
    return results


def build_commands(run_cfg: RunConfig, tasks: list[TaskSpec], dry_run: bool = True) -> list[Command]:
    builder: Any = run_cfg.values.get("command_builder")
    if builder is None:
        raise NotImplementedError("use build_sequence_commands with the LLaVA adapter")
    return list(builder(run_cfg, tasks, dry_run=dry_run))


def run(run_cfg: RunConfig, tasks: list[TaskSpec], dry_run: bool = True,
        run_dir: str | Path | None = None, resume: bool = False) -> list[Command]:
    """Compatibility entry point for builders that return command objects."""
    commands = build_commands(run_cfg, tasks, dry_run=dry_run)
    if not dry_run:
        if run_dir is None:
            raise ValueError("run_dir is required for actual execution")
        execute_commands(commands, run_dir, run_cfg.values["repo_root"], resume=resume)
    return commands
