# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Generate v4 YAML from v3 JSON and the recorded command fixtures.

No configuration is hand-transcribed. The input fixtures are produced by
``tools/v3_command_dump.py``; this tool only normalizes paths and reorganizes
values into the layered v4 schema.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

from mcitlib.registry import BENCHMARKS, METHODS, get_method

PATH_PREFIXES = (
    ("/opt/mcitlib-example/clip-vit-large-patch14-336", "${model_root}/clip-vit-large-patch14-336"),
    ("/opt/mcitlib-example/MCITlib/checkpoints", "${ckpt_root}"),
    ("/opt/mcitlib-example/MCITlib", "${repo_root}"),
    ("/opt/mcitlib-example/env/bin/python", "${python_root}"),
    ("/your_model_path", "${model_root}"),
    ("/your_data_path", "${data_root}"),
    ("/your_ckpts_path", "${ckpt_root}"),
    ("/your_path/MCITlib_v3", "${repo_root}"),
    ("__V3_ROOT__", "${repo_root}"),
)


def normalise(value: Any) -> Any:
    if isinstance(value, str):
        for source, target in PATH_PREFIXES:
            if value == source:
                return target
            if value.startswith(source + "/"):
                return target + value[len(source):]
        return value
    if isinstance(value, list):
        return [normalise(item) for item in value]
    if isinstance(value, dict):
        return {key: normalise(item) for key, item in value.items()}
    return value


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_options(tokens: list[str]) -> tuple[dict[str, Any], list[str]]:
    options: dict[str, Any] = {}
    positional: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            positional.append(token)
            index += 1
            continue
        key_value = token[2:]
        if "=" in key_value:
            key, value = key_value.split("=", 1)
            index += 1
        elif index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            key, value = key_value, tokens[index + 1]
            index += 2
        else:
            key, value = key_value, True
            index += 1
        if key in options:
            previous = options[key]
            options[key] = previous + [value] if isinstance(previous, list) else [previous, value]
        else:
            options[key] = value
    return options, positional


def command_details(argv: list[str]) -> dict[str, Any]:
    argv = normalise(argv)
    result: dict[str, Any] = {"argv": argv}
    if argv and argv[0] == "deepspeed":
        entry_index = next((i for i, value in enumerate(argv) if value.endswith(".py")), None)
        entry = argv[entry_index] if entry_index is not None else None
        result["launcher"] = parse_options(argv[1:entry_index])[0] if entry_index is not None else {}
        result["entry"] = entry
        if entry_index is not None:
            result["args"], result["positional"] = parse_options(argv[entry_index + 1:])
        else:
            result["args"], result["positional"] = {}, []
    else:
        result["args"], result["positional"] = parse_options(argv[1:])
    return result


def config_paths(v3_root: Path, method: Any, benchmark: Any, task: Any, router: bool = False) -> dict[str, str]:
    config_method = "RegLoRA" if method.id == "sefe" else method.v3_dir
    split = "train_router" if router else "train"
    eval_split = "eval_router" if router else "eval"
    return {
        "model_config": "model_configs/llava.json",
        "data_config": f"data_configs/{benchmark.v3_name}/{task}.json",
        "train_config": f"train_configs/{config_method}/LLaVA/{benchmark.v3_name}/{split}/task{TASK_ID_PLACEHOLDER}.json",
        "eval_config": f"train_configs/{config_method}/LLaVA/{benchmark.v3_name}/{eval_split}/task{TASK_ID_PLACEHOLDER}.json",
    }


TASK_ID_PLACEHOLDER = "{task_id}"


def source_values(v3_root: Path, method: Any, benchmark: Any, task_name: str, task_id: int, router: bool) -> dict[str, Any]:
    paths = config_paths(v3_root, method, benchmark, task_name, router)
    values: dict[str, Any] = {}
    for key, pattern in paths.items():
        relative = pattern.replace(TASK_ID_PLACEHOLDER, str(task_id))
        path = v3_root / "configs" / relative
        if not path.is_file():
            raise FileNotFoundError(f"required v3 config is missing: {path}")
        values[key] = normalise(read_json(path))
        values.setdefault("source", {})[key] = relative
    return values


def common_values(commands: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep only identical training options across a set of commands."""
    if not commands:
        return {}
    details = [command_details(c["argv"])["args"] for c in commands if c["kind"] == "train"]
    keys = set.intersection(*(set(item) for item in details)) if details else set()
    return {key: details[0][key] for key in sorted(keys) if all(item[key] == details[0][key] for item in details[1:])}


def command_record(command: dict[str, Any]) -> dict[str, Any]:
    result = command_details(command["argv"]) | {"kind": command["kind"], "order": command["order"]}
    env = {key: value for key, value in command.get("env", {}).items() if key == "CUDA_VISIBLE_DEVICES"}
    if env:
        result["env"] = env
    return result


def task_commands(payload: dict[str, Any], task_id: int) -> dict[str, Any]:
    commands = [c for c in payload["commands"] if c["task_id"] == task_id]
    result: dict[str, Any] = {}
    for kind, output_key in (
        ("train", "train_command"),
        ("eval", "eval_commands"),
        ("extra", "extra_commands"),
        ("other", "other_commands"),
    ):
        selected = [command_record(c) for c in commands if c["kind"] == kind]
        if kind == "train":
            if len(selected) != 1:
                raise ValueError(f"expected one training command for task {task_id}, got {len(selected)}")
            result[output_key] = selected[0]
        else:
            result[output_key] = selected
    installs = [c["argv"] for c in payload["commands"] if c["kind"] == "install"]
    result["removed_install_commands"] = sorted({tuple(x) for x in installs})
    result["removed_install_commands"] = [list(x) for x in result["removed_install_commands"]]
    return result


def make_task(v3_root: Path, payload: dict[str, Any], method: Any, benchmark: Any, task_name: str, task_id: int, router: bool = False) -> dict[str, Any]:
    values = source_values(v3_root, method, benchmark, task_name, task_id, router)
    placeholders = payload.get("config_placeholders", {})
    for config_key, relative in values["source"].items():
        if relative in placeholders:
            values[config_key].update(placeholders[relative])
    return {
        "task_id": task_id,
        "task_name": task_name,
        **values,
        **task_commands(payload, task_id),
    }


def relocate_smolora_assets(value: Any) -> Any:
    """Move only the three authorized instruction-embedding assets."""
    if isinstance(value, str):
        prefix = "${ckpt_root}/MCITlib_v3/LLaVA/SMoLoRA/ins_emb_multi_"
        if value.startswith(prefix):
            return "${repo_root}/assets/smolora/ins_emb_multi_" + value[len(prefix):]
        return value
    if isinstance(value, list):
        return [relocate_smolora_assets(item) for item in value]
    if isinstance(value, dict):
        return {key: relocate_smolora_assets(item) for key, item in value.items()}
    return value


def smolora_asset_metadata(v3_root: Path, benchmarks: list[Any]) -> dict[str, Any]:
    records = []
    for benchmark in benchmarks:
        name = f"ins_emb_multi_{benchmark.id.removeprefix('mllm_')}.pkl"
        source = v3_root / "LLaVA/SMoLoRA" / name
        records.append({
            "benchmark": benchmark.id,
            "path": f"${{repo_root}}/assets/smolora/{name}",
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        })
    return {"instruction_embeddings": records}


def migrate_method(v3_root: Path, commands_root: Path, method: Any, benchmarks: list[Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": method.id,
        "v3_dir": method.v3_dir,
        "environment": method.env,
        "train_entry": method.train_entry,
        "deepspeed_config": method.deepspeed_config,
        "separate_train_and_infer": method.separate_train_and_infer,
        "source_metadata": {},
        "overrides": {"llava": {}},
    }
    all_train_commands = []
    for benchmark in benchmarks:
        path = commands_root / method.id / f"{benchmark.id}.json"
        payload = read_json(path)
        all_train_commands.extend(payload["commands"])
        result["source_metadata"][benchmark.id] = {
            "sequence_repairs": payload.get("sequence_repairs", []),
            "path_placeholders": payload.get("path_placeholders", {}),
            "config_aliases": payload.get("config_aliases", {}),
        }
        benchmark_values = {
            f"task_{task_id}": make_task(v3_root, payload, method, benchmark, task_name, task_id)
            for task_id, task_name in enumerate(benchmark.tasks, start=1)
        }
        pre_commands = [command_record(c) | {"task_id": 0} for c in payload["commands"] if c["task_id"] == 0]
        if pre_commands:
            benchmark_values = {"pre_commands": pre_commands, **benchmark_values}
        result["overrides"]["llava"][benchmark.id] = benchmark_values
    result["defaults"] = {"train_args": common_values(all_train_commands)}
    if method.id == "mrlora":
        result["router"] = {"sequence": "router", "benchmarks": {}}
        for benchmark in benchmarks:
            payload = read_json(commands_root / method.id / f"{benchmark.id}_router.json")
            result["router"]["benchmarks"][benchmark.id] = {
                "config_aliases": payload.get("config_aliases", {}),
                "script_aliases": payload.get("script_aliases", {}),
                "config_placeholders": payload.get("config_placeholders", {}),
                "tasks": {
                    f"task_{task_id}": make_task(v3_root, payload, method, benchmark, task_name, task_id, router=True)
                    for task_id, task_name in enumerate(benchmark.tasks, start=1)
                },
            }
    if method.id == "smolora":
        result = relocate_smolora_assets(result)
        result["assets"] = smolora_asset_metadata(v3_root, benchmarks)
    return result


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-root", type=Path, default=os.environ.get("MCITLIB_V3_ROOT", project.parent / "MCITlib_v3_ref"))
    parser.add_argument("--commands-root", type=Path, default=project / "tests/fixtures/v3_commands/llava")
    parser.add_argument("--out", type=Path, default=project / "configs")
    parser.add_argument("--methods", help="comma-separated method IDs; default all")
    args = parser.parse_args()
    v3_root = args.v3_root.expanduser().resolve()
    methods = [get_method(x.strip()) for x in args.methods.split(",")] if args.methods else list(METHODS)
    benchmarks = list(BENCHMARKS)

    (args.out / "backbones").mkdir(parents=True, exist_ok=True)
    (args.out / "benchmarks").mkdir(parents=True, exist_ok=True)
    (args.out / "methods").mkdir(parents=True, exist_ok=True)
    backbone = {
        "id": "llava",
        "environment": "llava",
        "model_config": "model_configs/llava.json",
        "train_launcher": {"master_port": "9001"},
        "entries": {"standard": "llava/train/train_mem.py", "moe": "llava/train/train_mem_MOE.py"},
        "deepspeed_configs": {"zero2": "configs/deepspeed/zero2.json", "zero3_offload": "configs/deepspeed/zero3_offload.json"},
    }
    (args.out / "backbones" / "llava.yaml").write_text(yaml.safe_dump(backbone, sort_keys=False, allow_unicode=True), encoding="utf-8")
    for benchmark in benchmarks:
        value = {
            "id": benchmark.id,
            "v3_name": benchmark.v3_name,
            "train_script": benchmark.train_script,
            "eval_dir": benchmark.eval_dir,
            "tasks": [{"task_id": i, "name": name} for i, name in enumerate(benchmark.tasks, start=1)],
        }
        (args.out / "benchmarks" / f"{benchmark.id}.yaml").write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8")
    for method in methods:
        value = migrate_method(v3_root, args.commands_root, method, benchmarks)
        destination = args.out / "methods" / f"{method.id}.yaml"
        destination.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"wrote {destination}")
    print(f"migrated {len(methods)} methods × {len(benchmarks)} benchmarks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
