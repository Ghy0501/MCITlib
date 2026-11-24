# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Build and execute ordered continual-learning command sequences."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .config.loader import deep_merge, load_layered_config
from .config.schema import Command, DataSource, RunConfig, TaskSpec
from .eval.chunks import insert_merge_steps
from .gpus import (
    apply_execution_profile,
    apply_gpu_assignment,
    require_four_discovered_chunks,
)
from .gpus import (
    execution_profile as resolve_execution_profile,
)
from .methods import create_method
from .paths import managed_path, process_env, vendored_pythonpath
from .registry import compatibility, get_backbone, get_benchmark, get_method


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


def _isolate_results(commands: list[Command], method_id: str, backbone_id: str = "llava") -> None:
    """Give every method its own result folder.

    v3 ran each method in its own directory, so every ``./results/...`` was private.  From the
    repository root they would collide (LoRA-FT and Replay both write ``LoRA-task<k>``) and the
    later method would overwrite or mix with the earlier one's files.  Evaluation output goes
    to ``<results_root>/<method>/...`` instead; the MR-LoRA router extraction scripts get the same
    root through ``MCITLIB_RESULTS_ROOT``.
    """
    prefix = "" if backbone_id == "llava" else f"/{backbone_id}"
    root = f"${{results_root}}{prefix}/{method_id}"

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


def _value(command: Command, flag: str) -> str:
    if command.argv.count(flag) != 1 or command.argv.index(flag) + 1 >= len(command.argv):
        raise ValueError(f"validation repair requires exactly one {flag}: {command.argv}")
    return command.argv[command.argv.index(flag) + 1]


def _mrlora_ucit_datasets(config_root: str | Path, method: Any, backbone: str = "llava") -> dict[str, dict[str, Any]]:
    from .methods._mrlora.router_results.ucit_validation import EXPERTS

    return {domain: load_layered_config(config_root, backbone, "ucit", method.id, task_id=index)["data_config"]
            for index, domain in enumerate(EXPERTS, 1)}


def _mrlora_ucit_experts(config_root: str | Path, values: dict[str, Any], task: TaskSpec,
                         method: Any) -> list[Command]:
    """Generate every expert/domain candidate before retrospective router selection."""
    datasets = _mrlora_ucit_datasets(config_root, method, values.get("backbone", "llava"))
    result = []
    for domain_index, (domain, data) in enumerate(datasets.items()):
        directory = Path(values["eval_config"]["result_path"]) / domain / values["eval_config"]["stage"]
        order = 100 + domain_index * 5
        for chunk in range(2):
            argv = ["python", "-m", "llava.eval.CoIN.model_others",
                    "--model-path", values["eval_config"]["model_path"],
                    "--model-base", values["model_config"]["model_name"],
                    "--question-file", data["test_path"], "--image-folder", data["test_folder"],
                    "--answers-file", str(directory / f"2_{chunk}.jsonl"),
                    "--num-chunks", "2", "--chunk-idx", str(chunk), "--temperature", "0", "--conv-mode", "vicuna_v1"]
            result.append(Command(argv=argv, kind="eval", task_id=task.task_id, sequence="train",
                                  metadata={"order": order + chunk, "validation_fix": "mrlora_ucit_expert_inference"}))
        scorer = "eval_caption" if domain in {"VizWiz", "Flickr30k"} else "eval_deepseek_r1"
        annotation = data["anno_path"] if scorer == "eval_caption" else data["test_path"]
        result.append(Command(argv=["python", "-m", f"llava.eval.CoIN.{scorer}", "--annotation-file", annotation,
                                    "--result-file", str(directory / "merge.jsonl"), "--output-dir", str(directory)],
                              kind="eval", task_id=task.task_id, sequence="train",
                              metadata={"order": order + 3, "validation_fix": "mrlora_ucit_expert_scoring"}))
    return _apply_script_aliases(result, method)


def _mrlora_ucit_router(config_root: str | Path, values: dict[str, Any], task: TaskSpec,
                        method: Any) -> list[Command]:
    """Evaluate the actual router checkpoint on each domain seen through this stage."""
    datasets = _mrlora_ucit_datasets(config_root, method, values.get("backbone", "llava"))
    root = Path("./results/UCIT_router_raw/each_dataset")
    result = []
    for domain_index, (domain, data) in enumerate(datasets.items()):
        if domain_index >= task.task_id:
            break
        directory = root / domain / f"MR-LoRA-router-task{task.task_id}"
        for chunk in range(2):
            argv = ["python", "-m", "llava.eval.CoIN.model_ucit_router",
                    "--model-path", values["eval_config"]["model_path"],
                    "--model-base", values["model_config"]["model_name"],
                    "--question-file", data["test_path"], "--image-folder", data["test_folder"],
                    "--answers-file", str(directory / f"2_{chunk}.jsonl"),
                    "--num-chunks", "2", "--chunk-idx", str(chunk), "--temperature", "0", "--conv-mode", "vicuna_v1"]
            result.append(Command(argv=argv, kind="eval", task_id=task.task_id, sequence="router",
                                  metadata={"order": 100 + domain_index * 5 + chunk,
                                            "validation_fix": "mrlora_ucit_router_inference"}))
    return _apply_script_aliases(result, method)


def _mrlora_ucit_route_scores(commands: list[Command], method: Any, config_root: str | Path, paths: Any,
                              expected_eval_samples: int | None, expected_eval_hashes: dict[str, str] | None,
                              backbone: str = "llava"
                              ) -> list[Command]:
    """Join each router output with six same-domain experts, then use the original scorer."""
    from .methods._mrlora.router_results.ucit_validation import EXPERTS

    datasets = _mrlora_ucit_datasets(config_root, method, backbone)
    pending: dict[tuple[int | None, str], str] = {}
    result = []
    for command in commands:
        if command.metadata.get("eval_target") == "mcitlib.methods._mrlora.eval_runners.model_ucit_router":
            directory = str(Path(_value(command, "--answers-file")).parent)
            key = (command.task_id, directory)
            question = _value(command, "--question-file")
            if key in pending and pending[key] != question:
                raise ValueError(f"UCIT router chunks disagree on question file: {directory}")
            pending[key] = question
        result.append(command)
        if not command.metadata.get("merge_chunks") or "--discover" in command.argv:
            continue
        merged = Path(_value(command, "--output"))
        key = (command.task_id, str(merged.parent))
        if key not in pending:
            continue
        question = pending.pop(key)
        domain_dir = merged.parent.parent
        domain = domain_dir.name
        if (domain not in datasets or domain_dir.parent.name != "each_dataset" or
                domain_dir.parent.parent.name != "UCIT_router_raw" or
                merged.parent.name != f"MR-LoRA-router-task{command.task_id}"):
            raise ValueError(f"unexpected MR-LoRA UCIT router output: {merged}")
        method_dir = domain_dir.parent.parent.parent
        route_dir = method_dir / "UCIT_router" / "each_dataset" / domain / merged.parent.name
        routed = route_dir / "routed_predictions.jsonl"
        normalized = route_dir / "scorer_annotations.json"
        inputs = [str(method_dir / "UCIT" / "each_dataset" / domain / f"MR-LoRA-task{index}" / "merge.jsonl")
                  for index in range(1, len(EXPERTS) + 1)]
        argv = ["python", "-m", "mcitlib.methods._mrlora.router_results.ucit_validation", "--annotation-file", question,
                "--result-file", str(merged), "--output", str(routed), "--normalized-annotation", str(normalized),
                "--stats-file", str(route_dir / "routing_stats.json")]
        if expected_eval_samples is not None:
            argv.extend(("--expected-count", str(expected_eval_samples)))
        if expected_eval_hashes is not None:
            argv.extend(("--annotation-sha256", expected_eval_hashes[domain]))
        caption = domain in {"VizWiz", "Flickr30k"}
        reference = datasets[domain].get("anno_path") if caption else None
        if reference and paths is not None:
            reference = paths.resolve_templates(reference, "ucit")
        if reference:
            argv.extend(("--caption-reference", reference))
            if expected_eval_hashes is not None:
                argv.extend(("--caption-sha256", expected_eval_hashes[f"{domain}:caption"]))
        argv.extend(("--inputs", *inputs))
        join = Command(argv=argv, cwd=command.cwd, env=dict(command.env), kind="extra", task_id=command.task_id,
                       sequence="router", metadata={"order": command.metadata.get("order", 0) + 0.1,
                                                    "validation_fix": "mrlora_ucit_router_selection"})
        scorer = "eval_caption" if caption else "eval_deepseek_r1"
        scorer_annotation = reference if caption else str(normalized)
        if not scorer_annotation:
            raise ValueError(f"missing caption reference for {domain}")
        score = Command(argv=["python", "-m", f"llava.eval.CoIN.{scorer}", "--annotation-file", scorer_annotation,
                              "--result-file", str(routed), "--output-dir", str(route_dir)],
                        cwd=command.cwd, env=dict(command.env), kind="eval", task_id=command.task_id,
                        sequence="router", metadata={"order": command.metadata.get("order", 0) + 0.2,
                                                     "validation_fix": "mrlora_ucit_router_scoring"})
        result.append(join)
        result.extend(_apply_script_aliases([score], method))
    if pending:
        raise ValueError(f"MR-LoRA UCIT router predictions have no merge: {list(pending)}")
    return result


def _mrlora_dcl_expert_inference(config_root: str | Path, values: dict[str, Any], task: TaskSpec,
                                  recorded: list[Command], method: Any) -> list[Command]:
    """Restore the five expert predictions commented out in the migrated DCL scripts."""
    from .methods._mrlora.router_results.dcl_validation import SCORERS

    if any("--answers-file" in item.argv for item in recorded):
        raise ValueError("MR-LoRA DCL expert records already contain inference; review the validation repair")
    datasets = {}
    for index, domain in enumerate(("RS", "Med", "AD", "Sci", "Fin"), 1):
        source = load_layered_config(config_root, values.get("backbone", "llava"), "mllm_dcl", method.id, task_id=index)
        datasets[domain] = source["data_config"]
    inference = {"RS": "model_ai2d", "Med": "model_pvqa", "AD": "model_ai2d",
                 "Sci": "model_ai2d", "Fin": "model_fin"}
    added: list[Command] = []
    seen = set()
    for scorer in recorded:
        if scorer.kind != "eval" or "--result-file" not in scorer.argv:
            continue
        result_file = Path(_value(scorer, "--result-file"))
        domain = result_file.parent.parent.name
        if (domain not in datasets or domain in seen or result_file.name != "merge.jsonl" or
                scorer.metadata.get("source_eval_module") != f"llava.eval.CoIN.{SCORERS[domain]}" or
                _value(scorer, "--annotation-file") != datasets[domain]["test_path"]):
            raise ValueError(f"unexpected MR-LoRA DCL scorer: {scorer.argv}")
        seen.add(domain)
        for index in range(2):
            argv = ["python", "-m", f"llava.eval.CoIN.{inference[domain]}",
                    "--model-path", values["eval_config"]["model_path"],
                    "--model-base", values["model_config"]["model_name"],
                    "--question-file", datasets[domain]["test_path"],
                    "--image-folder", datasets[domain]["test_folder"],
                    "--answers-file", str(result_file.parent / f"2_{index}.jsonl"),
                    "--num-chunks", "2", "--chunk-idx", str(index),
                    "--temperature", "0", "--conv-mode", "vicuna_v1"]
            added.append(Command(argv=argv, kind="eval", task_id=task.task_id, sequence="train",
                                 metadata={"order": scorer.metadata["order"] - 0.9 + index * 0.3,
                                           "validation_fix": "mrlora_dcl_expert_inference"}))
    if seen != set(datasets):
        raise ValueError(f"MR-LoRA DCL expert scorer coverage is incomplete: {sorted(seen)}")
    return _apply_script_aliases(added, method)


def _keeplora_dcl_task2_med(config_root: str | Path, values: dict[str, Any], task: TaskSpec,
                            recorded: list[Command], method: Any) -> list[Command]:
    """Reuse the method's task-3 Med evaluation with the task-2 checkpoint and stage."""
    later = load_layered_config(config_root, values.get("backbone", "llava"), "mllm_dcl", method.id, task_id=3)
    old_stage, new_stage = later["eval_config"]["stage"], values["eval_config"]["stage"]
    selected = [item for item in later["eval_commands"] if
                any(f"/Med/{old_stage}" in str(token) for token in item["argv"])]
    if (len(selected) != 3 or any(f"/Med/{new_stage}" in str(token)
                                  for command in recorded for token in command.argv)):
        raise ValueError("unexpected KeepLoRA DCL Med evaluation; review the validation repair")
    added = []
    last = max((command.metadata.get("order", -1) for command in recorded), default=-1)
    for index, item in enumerate(selected, 1):
        command = _command_from_record(item, task.task_id, "eval", "train")
        command.argv = [values["eval_config"]["model_path"] if token == later["eval_config"]["model_path"] else
                        token.replace(f"/Med/{old_stage}/", f"/Med/{new_stage}/").replace(
                            f"/Med/{old_stage}", f"/Med/{new_stage}") for token in command.argv]
        command.metadata.update(order=last + index, validation_fix="keeplora_dcl_stage2_med")
        added.append(command)
    return _apply_script_aliases(added, method)


def _mrlora_dcl_route_scores(commands: list[Command], method: Any, expected_eval_samples: int | None,
                             expected_eval_hashes: dict[str, str] | None) -> list[Command]:
    """After each router merge, select expert answers and run the original domain scorer."""
    from .methods._mrlora.router_results.dcl_validation import EXPERTS, SCORERS

    pending: dict[tuple[int | None, str], str] = {}
    result = []
    for command in commands:
        if command.metadata.get("eval_target") == "mcitlib.methods._mrlora.eval_runners.model_dcl_router":
            directory = str(Path(_value(command, "--answers-file")).parent)
            key = (command.task_id, directory)
            question = _value(command, "--question-file")
            if key in pending and pending[key] != question:
                raise ValueError(f"router chunks disagree about their question file: {directory}")
            pending[key] = question
        result.append(command)
        if not command.metadata.get("merge_chunks") or "--discover" in command.argv:
            continue
        merged = Path(_value(command, "--output"))
        key = (command.task_id, str(merged.parent))
        if key not in pending:
            continue
        question = pending.pop(key)
        domain_dir = merged.parent.parent
        domain = domain_dir.name
        if (domain not in SCORERS or domain_dir.parent.name != "each_dataset" or
                domain_dir.parent.parent.name != "DCL" or
                merged.parent.name != f"MR-LoRA-router-task{command.task_id}"):
            raise ValueError(f"unexpected MR-LoRA DCL router output: {merged}")
        route_dir = domain_dir.parent.parent.parent / "DCL_router" / "each_dataset" / domain / merged.parent.name
        routed = route_dir / "routed_predictions.jsonl"
        inputs = [str(domain_dir / f"MR-LoRA-task{index}" / "merge.jsonl") for index in range(1, len(EXPERTS) + 1)]
        join_argv = ["python", "-m", "mcitlib.methods._mrlora.router_results.dcl_validation",
                     "--annotation-file", question, "--result-file", str(merged), "--output", str(routed),
                     "--stats-file", str(route_dir / "routing_stats.json")]
        if expected_eval_samples is not None:
            join_argv.extend(("--expected-count", str(expected_eval_samples)))
        if expected_eval_hashes is not None:
            join_argv.extend(("--annotation-sha256", expected_eval_hashes[domain]))
        join_argv.extend(("--inputs", *inputs))
        join = Command(argv=join_argv,
                       cwd=command.cwd, env=dict(command.env), kind="extra", task_id=command.task_id,
                       sequence="router", metadata={"order": command.metadata.get("order", 0) + 0.1,
                                                    "validation_fix": "mrlora_dcl_router_selection"})
        scorer = Command(argv=["python", "-m", f"llava.eval.CoIN.{SCORERS[domain]}",
                               "--annotation-file", question, "--result-file", str(routed),
                               "--output-dir", str(route_dir)], cwd=command.cwd, env=dict(command.env),
                         kind="eval", task_id=command.task_id, sequence="router",
                         metadata={"order": command.metadata.get("order", 0) + 0.2,
                                   "validation_fix": "mrlora_dcl_router_scoring"})
        result.append(join)
        result.extend(_apply_script_aliases([scorer], method))
    if pending:
        raise ValueError(f"MR-LoRA DCL router predictions have no merge: {list(pending)}")
    return result


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
    validation_fixes: bool = False,
    execution_profile: str | None = None,
    expected_eval_samples: int | None = None,
    expected_eval_hashes: dict[str, str] | None = None,
) -> list[Command]:
    """Build v3-ordered commands; NB-5 install commands are never emitted."""
    if expected_eval_samples is not None and (type(expected_eval_samples) is not int or expected_eval_samples < 1 or not validation_fixes):
        raise ValueError("expected_eval_samples requires validation_fixes and a positive count")
    if expected_eval_hashes is not None:
        selected_benchmark = get_benchmark(benchmark_id)
        expected_domains = set(selected_benchmark.tasks)
        if selected_benchmark.id == "ucit":
            expected_domains.update({"VizWiz:caption", "Flickr30k:caption"})
        if (expected_eval_samples is None or not isinstance(expected_eval_hashes, dict) or
                set(expected_eval_hashes) != expected_domains or
                any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                    for value in expected_eval_hashes.values())):
            raise ValueError("expected_eval_hashes requires a positive count and SHA-256 for every benchmark input")
    profile = resolve_execution_profile(config_root, execution_profile, gpus)
    gpu_ids = ",".join(profile["gpu_ids"]) if profile else gpus
    backbone = get_backbone(backbone_id)
    if not backbone.v4_supported:
        raise ValueError(f"backbone is not migrated yet: {backbone_id}")
    method = get_method(method_id)
    benchmark = get_benchmark(benchmark_id)
    from .registry import benchmark_supported

    supported, reason = compatibility(method.id, backbone.id)
    if not supported:
        raise ValueError(f"unsupported {backbone.id}/{method.id}: {reason}")
    if not benchmark_supported(method.id, benchmark.id):
        raise ValueError(f"{method.id} is not configured for {benchmark.id}")
    if backbone.id == "internvl":
        from .gpus import parse_gpus

        selected = parse_gpus(gpu_ids) or ["0", "1"]
        if not set(selected) <= {"0", "1", "2", "3"}:
            raise ValueError("InternVL execution is restricted to GPUs 0,1,2,3")
        gpu_ids = ",".join(selected)
        if benchmark.id not in {"ucit", "mllm_dcl"}:
            raise ValueError("InternVL currently supports UCIT and MLLM-DCL")
    if profile and method.id == "dgg":
        raise ValueError("DGG requires the single-GPU execution path; use its separate sample profile")
    if profile and benchmark.id not in {"ucit", "mllm_dcl"}:
        raise ValueError("four_gpu_validation supports only ucit and mllm_dcl")
    if benchmark.id == "general":
        if not only_eval or sequence != "train":
            raise ValueError("general benchmarks are supported only by the eval command")
        from .backbones.llava.general import build_general_commands
        commands = build_general_commands(config_root, method.id, task_id, model_path, model_base, num_chunks)
        _isolate_results(commands, method.id, backbone.id)
        if paths is not None:
            for command in commands:
                command.argv = [paths.resolve_templates(str(value), benchmark.id) for value in command.argv]
                command.cwd = paths.resolve_templates(command.cwd, benchmark.id)
                command.env = {key: paths.resolve_templates(str(value), benchmark.id) for key, value in command.env.items()}
        _ensure_pythonpath(commands, config_root)
        return commands
    if sequence not in {"train", "router"}:
        raise ValueError(f"unknown sequence: {sequence}")
    if sequence == "router" and method.id != "mrlora":
        raise ValueError(f"method {method.id} has no router sequence")
    if backbone_id not in {"llava", "internvl"}:
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
            if validation_fixes and backbone.id == "llava" and method.id == "modalprompt" and benchmark.id == "ucit" and current_id == 2:
                # The recorded task-2 command points at task 1, while its inference and
                # task 3 both expect Task2. Keep the historical configuration intact.
                output = train.argv.index("--output_dir") + 1
                if train.argv[output] != "${ckpt_root}/UCIT/LLaVA/ModalPrompt/Task1_llava":
                    raise ValueError("unexpected ModalPrompt UCIT task 2 output; review validation repair")
                train.argv[output] = "${ckpt_root}/UCIT/LLaVA/ModalPrompt/Task2_llava"
                train.metadata["validation_fix"] = "modalprompt_ucit_task2_output_dir"
            if validation_fixes and backbone.id == "llava" and method.id == "modalprompt" and current_id > 1:
                previous = values.get("train_config", {}).get("previous_model")
                if not isinstance(previous, str) or not previous or "--previous_task_model_path" in train.argv:
                    raise ValueError("ModalPrompt validation repair needs a unique previous_model")
                train.argv.extend(["--previous_task_model_path", previous])
                train.metadata["previous_task_fix"] = "modalprompt_load_previous_task"
            train.metadata["order"] = values["train_command"].get("order", -1)
            if backbone.id == "internvl":
                train.metadata["internvl_source_sha256"] = values["internvl_source_sha256"]
                if values.get("internvl_repairs"):
                    train.metadata["internvl_repairs"] = values["internvl_repairs"]
            train.sequence = sequence
            if method.id == "pclr":
                from copy import deepcopy

                def set_flag(command, flag, value):
                    if flag in command.argv:
                        command.argv[command.argv.index(flag) + 1] = str(value)
                    else:
                        command.argv.extend([flag, str(value)])

                set_flag(train, "--lora_enable", "False")
                set_flag(train, "--freeze_backbone", "True")
                set_flag(train, "--freeze_mm_mlp_adapter", "True")
                set_flag(train, "--pretrain_mm_mlp_adapter", values["model_config"]["mm_projector"])
                train.env["MCITLIB_PCLR_STAGE"] = "learn"
                train.env["MCITLIB_PCLR_TASK_ID"] = str(current_id)
                train.metadata["pclr_phase"] = "learning"
                # Consolidate from task 4 onward, including a benchmark's final task.
                settings = values["pclr"]
                new_rank = settings["new_rank_by_task"].get(current_id, settings["new_rank"])
                skip_interval = settings["skip_interval_by_task"].get(current_id, settings["skip_interval"])
                set_flag(train, "--lora_r", new_rank)
                train.env["MCITLIB_PCLR_OPTIONS"] = json.dumps({
                    "new_rank": new_rank, "top_rank": settings["top_rank"],
                    "alpha": settings["alpha"], "share_rank": settings["shared_rank"],
                    "skip_interval": skip_interval,
                }, sort_keys=True)
                if current_id >= settings["first_compression_after_task"]:
                    output = train.argv[train.argv.index("--output_dir") + 1]
                    learning_output = output + "/learning"
                    set_flag(train, "--output_dir", learning_output)
                    integration = deepcopy(train)
                    set_flag(integration, "--previous_task_model_path", learning_output)
                    set_flag(integration, "--output_dir", output)
                    set_flag(integration, "--learning_rate", settings["integration_learning_rate"])
                    set_flag(integration, "--num_train_epochs", settings["integration_epochs"])
                    integration.env["MCITLIB_PCLR_STAGE"] = "integrate"
                    integration.sequence = "pclr_integration"
                    integration.metadata.update(pclr_phase="compression_integration", order=train.metadata["order"] + 0.5,
                                                epochs=settings["integration_epochs"],
                                                learning_rate=settings["integration_learning_rate"],
                                                skip_interval=skip_interval, new_rank=new_rank)
                else:
                    integration = None
            if method.id == "dgg" and current_id > 1:
                # A fresh seed is used for each old dataset by the preparation module.
                source_values = [values] + [load_layered_config(config_root, backbone_id, benchmark.id,
                                                                 method.id, task_id=old_id)
                                            for old_id in range(1, current_id)]
                replay_argv = ["python", "-m", "mcitlib.methods._dgg.replay"]
                for source_value in source_values:
                    data = source_value["data_config"]
                    replay_argv.extend(["--data_path", data["train_path"], "--image_folder", data["train_folder"]])
                replay_path = values["train_config"]["dgg_replay_path"]
                replay_manifest = replay_path + ".manifest.json"
                replay_argv.extend(["--output", replay_path, "--manifest", replay_manifest,
                                    "--samples-per-task", "2000", "--seed", "1993"])
                train.argv.extend(["--dgg_replay_manifest", replay_manifest])
                commands.append(Command(argv=replay_argv, kind="extra", task_id=current_id, sequence=sequence,
                                        env={"MCITLIB_METHOD": "dgg"},
                                        metadata={"order": train.metadata["order"] - 1, "stage": "dgg_replay"}))
            if method.id == "octopus":
                from .methods._octopus.plan import train_stages
                octopus_config = values["train_config"]
                auxiliary = getattr(paths, "octopus_aux_root", None)
                if auxiliary is not None:
                    octopus_config = {**octopus_config, "octopus_aux_root": str(auxiliary)}
                commands.extend(train_stages(train, octopus_config, current_id))
            else:
                commands.append(train)
                if method.id == "pclr" and integration is not None:
                    commands.append(integration)
        if only_eval or not method.separate_train_and_infer:
            recorded = _recorded_commands(values, task, include_extra=not only_eval, sequence=sequence)
            _apply_script_aliases(recorded, method)
            if method.id == "same" and backbone.id == "llava":
                for command in recorded:
                    if command.kind == "eval":
                        command.env["MCITLIB_SAME_CLIP_TOWER"] = values["model_config"]["vision_tower"]
            if validation_fixes and method.id == "mrlora" and benchmark.id == "ucit":
                # Historical router records resolve to ordinary expert commands and overwrite
                # their outputs. Build distinct full cross-expert and real router predictions.
                recorded = (_mrlora_ucit_experts(config_root, values, task, method) if sequence == "train" else
                            _mrlora_ucit_router(config_root, values, task, method))
            if validation_fixes and benchmark.id == "mllm_dcl" and sequence == "train":
                if method.id == "mrlora":
                    recorded = [command for command in recorded if not (
                        command.kind == "other" and "mcitlib/methods/_mrlora/router_results/dcl.py" in command.argv
                    )]
                    recorded.extend(_mrlora_dcl_expert_inference(config_root, values, task, recorded, method))
                elif method.id == "keeplora" and current_id == 2:
                    recorded.extend(_keeplora_dcl_task2_med(config_root, values, task, recorded, method))
            commands.extend(recorded)
    _isolate_results(commands, method.id, backbone.id)
    if backbone.id == "internvl":
        for command in commands:
            command.env.update({
                "MCITLIB_BACKBONE": "internvl",
                "MCITLIB_INTERNVL_VISION_TOWER": "${model_root}/InternViT-6B-224px",
                "MCITLIB_INTERNVL_CLIP_TOWER": "${model_root}/clip-vit-large-patch14-336",
                "TRITON_CACHE_DIR": "${repo_root}/.cache/triton/internvl",
            })
    if paths is not None:
        for command in commands:
            command.argv = [paths.resolve_templates(str(value), benchmark.id) for value in command.argv]
            command.cwd = paths.resolve_templates(command.cwd, benchmark.id)
            command.env = {key: paths.resolve_templates(str(value), benchmark.id) for key, value in command.env.items()}
            if "MCITLIB_OCTOPUS_HISTORY" in command.env:
                history = json.loads(command.env["MCITLIB_OCTOPUS_HISTORY"])
                command.env["MCITLIB_OCTOPUS_HISTORY"] = json.dumps(
                    [paths.resolve_templates(value, benchmark.id) for value in history])
            if "octopus_inputs" in command.metadata:
                command.metadata["octopus_inputs"] = [paths.resolve_templates(value, benchmark.id) for value in command.metadata["octopus_inputs"]]
            if method.id == "keeplora" and command.kind == "extra" and any(
                name in command.argv for name in (
                    "mcitlib.methods._keeplora.extract_gradients", "mcitlib.methods._keeplora.extract_weights"
                )
            ):
                command.env["MCITLIB_PATH_CONTEXT"] = paths.context(benchmark.id, include_octopus_aux=False)
    if profile and backbone.id == "internvl":
        from .backbones.internvl.sample_profile import prepare_sample_training

        prepare_sample_training(commands, config_root, method.id, sequence)
    if profile:
        commands = apply_execution_profile(commands, profile)
    apply_gpu_assignment(commands, gpu_ids, master_port)
    for command in commands:
        if command.metadata.get("octopus_stage") == "gpwc":
            command.env["CUDA_VISIBLE_DEVICES"] = (gpu_ids or "0,1").split(",")[0]
    if profile and backbone.id == "internvl":
        from .backbones.internvl.sample_profile import finalize_sample_training

        finalize_sample_training(commands)
    _ensure_pythonpath(commands, config_root)
    commands.sort(key=lambda command: (command.task_id or 0, command.metadata.get("order", -1)))
    # v3 merged the per-GPU chunk files with shell `cat` before scoring; restore that step.
    commands = insert_merge_steps(commands)
    if validation_fixes and method.id == "mrlora" and benchmark.id == "mllm_dcl" and sequence == "router":
        commands = _mrlora_dcl_route_scores(commands, method, expected_eval_samples, expected_eval_hashes)
    if validation_fixes and method.id == "mrlora" and benchmark.id == "ucit" and sequence == "router":
        commands = _mrlora_ucit_route_scores(commands, method, config_root, paths, expected_eval_samples,
                                             expected_eval_hashes, backbone.id)
    if profile:
        require_four_discovered_chunks(commands)
        for command in commands:
            command.metadata["execution_profile"] = profile["id"]
    return commands


def _step_identity(command: Command) -> str:
    payload = json.dumps(command.as_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def _write_marker(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


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

    def __init__(self, index: int, command: Command, repo_root: Path, logs: Path, identity: str,
                 attempt: str):
        argv = list(command.argv)
        if not argv:
            raise ValueError(f"empty command at sequence index {index}")
        # The generated local include lists physical GPU IDs. DeepSpeed may
        # cache CUDA's device count before it removes an inherited visibility
        # mask, so a mask of "4,5" makes physical slot 4 look out of range.
        # Only discard inherited masks for this generated launcher form;
        # explicit command masks and nonlocal/no-include launchers retain theirs.
        physical_include = (len(argv) > 2 and argv[0] == "deepspeed" and argv[1] == "--include" and
                            argv.count("--include") == 1 and
                            re.fullmatch(r"localhost:[0-9]+(?:,[0-9]+)*", argv[2]) is not None)
        if physical_include:
            slots = argv[2].split(":", 1)[1].split(",")
            physical_include = len(slots) == len(set(slots))
        if argv[0] in {"python", "python3"}:
            argv[0] = sys.executable
        elif argv[0] == "deepspeed":
            # A venv can inherit DeepSpeed from a parent environment without
            # providing a console script of its own. Always use this worker's
            # interpreter instead of a script with another Python's shebang.
            argv = [sys.executable, "-m", "deepspeed.launcher.runner", *argv[1:]]
        cwd = Path(command.cwd)
        cwd = cwd if cwd.is_absolute() else repo_root / cwd
        # Vendored llava/peft always come first, whatever the user's environment holds.
        env = process_env(command.env, repo_root)
        if physical_include and "CUDA_VISIBLE_DEVICES" not in command.env:
            env.pop("CUDA_VISIBLE_DEVICES", None)
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
        self.index, self.command, self.identity, self.attempt = index, command, identity, attempt
        self.log = logs / f"{index:05d}-{identity}-{attempt}.log"
        self.log.parent.mkdir(parents=True, exist_ok=True)
        # Transformers otherwise derives its logging directory from the checkpoint output.
        if command.kind == "train" and "mcitlib.backbones.llava.train_entry" in argv:
            logging_dir = str(logs / f"{index:05d}-{identity}-{attempt}-training")
            if "--logging_dir" in argv:
                argv[argv.index("--logging_dir") + 1] = logging_dir
            else:
                argv.extend(["--logging_dir", logging_dir])
            env["WANDB_DIR"] = str(logs)
        self.executed_argv = list(argv)
        self.started = datetime.now(timezone.utc).isoformat()
        self.handle = self.log.open("x", encoding="utf-8")
        try:
            self.process = subprocess.Popen(
                argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
        except BaseException:
            self.handle.close()
            raise

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
    logs_root: str | Path | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    before_step: Callable[[int, Command], None] | None = None,
) -> list[dict[str, Any]]:
    """Execute commands with durable logs and resume markers.

    Steps run one after another, except the chunks of one evaluation dataset, which run in
    parallel (one GPU each, as in v3) and are all waited for before the next step starts.  If any
    member fails the others are allowed to finish, every outcome is recorded, and the run stops.
    """
    from concurrent.futures import ThreadPoolExecutor

    repo_root = Path(repo_root).resolve()
    logs_root = managed_path(logs_root or repo_root / "logs", repo_root, label="logs_root")
    run_dir = managed_path(run_dir, repo_root, within=logs_root, label="run_dir")
    markers = managed_path(run_dir / ".mcitlib" / "steps", repo_root, within=logs_root, label="markers")
    logs = managed_path(run_dir / ".mcitlib" / "logs", repo_root, within=logs_root, label="logs")
    results: list[dict[str, Any]] = []

    def emit(status: str, index: int, command: Command, identity: str, **details: Any) -> None:
        if on_event:
            on_event({"status": status, "index": index, "id": identity, "task": command.task_id,
                      "kind": command.kind, "sequence": command.sequence, "command": command.as_dict(), **details})

    for batch in _parallel_batches(commands):
        pending: list[_Step] = []
        for index, command in batch:
            identity = _step_identity(command)
            prior_markers = [markers / f"{identity}.json", *markers.glob(f"{identity}-*.json")]
            if resume and any(marker.is_file() and json.loads(marker.read_text(encoding="utf-8")).get("status")
                              == "completed" for marker in prior_markers):
                results.append({"index": index, "id": identity, "status": "skipped"})
                emit("skipped", index, command, identity)
                continue
            if before_step:
                before_step(index, command)
            step = _Step(index, command, repo_root, logs, identity, uuid.uuid4().hex)
            pending.append(step)
            emit("started", index, command, identity, log=str(step.log), executed_argv=step.executed_argv)
        if not pending:
            continue
        if len(pending) == 1:
            codes = [pending[0].pump()]
        else:
            with ThreadPoolExecutor(max_workers=len(pending)) as pool:
                futures = [pool.submit(step.pump, f"[chunk {n}] ") for n, step in enumerate(pending)]
                codes = [future.result() for future in futures]
        failed: list[tuple[_Step, str]] = []
        for step, code in zip(pending, codes):
            status = "completed" if code == 0 else "failed"
            validation_error: str | None = None
            try:
                emit(status, step.index, step.command, step.identity, log=str(step.log), return_code=code,
                     executed_argv=step.executed_argv)
            except Exception as error:
                # A successful subprocess can still lack its required outputs.
                # Record that failure before allowing any following step.
                status = "failed"
                validation_error = f"{type(error).__name__}: {error}"
            _write_marker(markers / f"{step.identity}-{step.attempt}.json", {
                "id": step.identity, "index": step.index, "status": status,
                "return_code": code, "command": step.command.as_dict(),
                "executed_argv": step.executed_argv,
                "log": str(step.log), "started_at": step.started,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "validation_error": validation_error,
            })
            results.append({"index": step.index, "id": step.identity, "status": status, "return_code": code})
            if status == "failed":
                failed.append((step, validation_error or f"exit {code}"))
        if failed:
            first, reason = failed[0]
            raise RuntimeError(
                f"command {first.index} failed with {reason}; log: {first.log}"
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
