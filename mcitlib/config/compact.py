# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Expand the compact, declarative method configuration into the legacy shape.

The on-disk command templates contain ordered argv, and each command supplies
only its deviations. No shell parsing or sorting is involved in expansion.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .loader import deep_merge, read_yaml

FORMAT = "mcitlib-method-v1"
REFERENCE = re.compile(r"@\{([a-zA-Z0-9_.]+)\}")


def _lookup(context: dict[str, Any], name: str) -> Any:
    value: Any = context
    for key in name.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"unknown configuration reference: {name}")
        value = value[key]
    if not isinstance(value, (str, int, float, bool)):
        raise ValueError(f"configuration reference is not a scalar: {name}")
    return value


def _interpolate(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return REFERENCE.sub(lambda m: str(_lookup(context, m.group(1))), value)
    if isinstance(value, list):
        return [_interpolate(item, context) for item in value]
    if isinstance(value, dict):
        return {key: _interpolate(item, context) for key, item in value.items()}
    return value


def _option_positions(argv: list[str]) -> dict[str, int]:
    """Identify option values, distinguishing repeated flags as --flag#2, etc."""
    positions: dict[str, int] = {}
    counts: dict[str, int] = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if token.startswith("--") and "=" not in token and index + 1 < len(argv) and not argv[index + 1].startswith("--"):
            counts[token] = counts.get(token, 0) + 1
            key = token if counts[token] == 1 else f"{token}#{counts[token]}"
            positions[key] = index + 1
            index += 1
        index += 1
    return positions


def _parse_options(tokens: list[str]) -> tuple[dict[str, Any], list[str]]:
    """Reconstruct the historical inspection fields from the canonical argv."""
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


def _record(record: dict[str, Any], templates: dict[str, Any], fragments: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    name = record.get("template")
    if not isinstance(name, str) or name not in templates:
        raise ValueError(f"unknown command template: {name}")
    template = templates[name]
    if not isinstance(template, dict) or not isinstance(template.get("argv"), list):
        raise ValueError(f"command template {name} argv must be a list")
    argv = []
    for item in template["argv"]:
        if isinstance(item, str):
            argv.append(item)
        elif isinstance(item, dict) and set(item) == {"use"}:
            fragment_name = item["use"]
            if not isinstance(fragment_name, str) or fragment_name not in fragments:
                raise ValueError(f"unknown command fragment {fragment_name!r} in template {name}")
            fragment = fragments[fragment_name]
            if (not isinstance(fragment, list) or not fragment or
                    any(not isinstance(part, str) or not part for part in fragment)):
                raise ValueError(f"command fragment {fragment_name} in template {name} must be a non-empty string list")
            argv.extend(fragment)
        else:
            raise ValueError(f"invalid command fragment in template {name}: {item!r}")
    positions = _option_positions(argv)
    overrides = record.get("set", {})
    if not isinstance(overrides, dict):
        raise ValueError(f"command overrides for {name} must be a mapping")
    for key, value in overrides.items():
        if key not in positions:
            raise ValueError(f"unknown option {key!r} in command template {name}")
        argv[positions[key]] = value
    argv = _interpolate(argv, context)
    if not argv or any(not isinstance(item, str) or not item for item in argv):
        raise ValueError(f"command template {name} must expand to non-empty string argv")
    result: dict[str, Any] = {"argv": argv}
    if argv[0] == "deepspeed":
        entry_index = next((i for i, value in enumerate(argv) if value.endswith(".py")), None)
        result["launcher"] = _parse_options(argv[1:entry_index])[0] if entry_index is not None else {}
        result["entry"] = argv[entry_index] if entry_index is not None else None
        result["args"], result["positional"] = _parse_options(argv[entry_index + 1:]) if entry_index is not None else ({}, [])
    else:
        result["args"], result["positional"] = _parse_options(argv[1:])
    result.update(_interpolate({key: value for key, value in record.items() if key not in {"template", "set"}}, context))
    return result


def expand_method(root: Path, compact: dict[str, Any]) -> dict[str, Any]:
    """Return the historical method mapping; malformed references fail eagerly."""
    common = read_yaml(root / "common/tasks.yaml")
    command_library = read_yaml(root / "common/commands.yaml")
    templates = command_library["templates"]
    fragments = command_library.get("fragments", {})
    provenance = read_yaml(root / "common/provenance.yaml")
    result = {k: v for k, v in compact.items() if k not in {"format", "source_metadata", "overrides", "router"}}
    result["source_metadata"] = {
        benchmark: deep_merge(provenance, compact.get("source_metadata", {}).get(benchmark, {}))
        for benchmark in compact["overrides"]["llava"]
    }
    result["overrides"] = {"llava": {}}

    def expand_task(benchmark: str, task_key: str, spec: dict[str, Any], defaults: dict[str, Any], router: bool) -> dict[str, Any]:
        dataset = common["benchmarks"][benchmark]["tasks"][task_key]
        context: dict[str, Any] = {
            "method": result,
            "benchmark": common["benchmarks"][benchmark],
            "datasets": common["benchmarks"][benchmark]["tasks"],
            "task": {"task_id": dataset["task_id"], "task_name": dataset["task_name"]},
            "split": {"train": "train_router" if router else "train", "eval": "eval_router" if router else "eval"},
        }
        base = deep_merge(common["task"], dataset)
        base = deep_merge(base, defaults)
        ordinary = {k: v for k, v in spec.items() if k not in {"train_command", "eval_commands", "extra_commands", "other_commands"}}
        task = _interpolate(deep_merge(base, ordinary), context)
        context["task"] = task
        for key in ("train_command", "eval_commands", "extra_commands", "other_commands"):
            if key in spec:
                records = spec[key]
                task[key] = _record(records, templates, fragments, context) if key == "train_command" else [
                    _record(record, templates, fragments, context) for record in records
                ]
        return task

    def expand_benchmark(benchmark: str, values: dict[str, Any], router: bool) -> dict[str, Any]:
        defaults = values.get("task_defaults", {})
        expanded: dict[str, Any] = {}
        for key, spec in values.items():
            if key == "task_defaults":
                continue
            if key == "pre_commands":
                context = {"method": result, "benchmark": common["benchmarks"][benchmark],
                           "datasets": common["benchmarks"][benchmark]["tasks"]}
                expanded[key] = [_record(record, templates, fragments, context) for record in spec]
            elif key.startswith("task_"):
                expanded[key] = expand_task(benchmark, key, spec, defaults, router)
            else:
                expanded[key] = spec
        return expanded

    for benchmark, values in compact["overrides"]["llava"].items():
        result["overrides"]["llava"][benchmark] = expand_benchmark(benchmark, values, False)
    if "router" in compact:
        router = compact["router"]
        result["router"] = {key: value for key, value in router.items() if key != "benchmarks"}
        result["router"]["benchmarks"] = {}
        for benchmark, values in router["benchmarks"].items():
            item = {key: value for key, value in values.items() if key != "tasks"}
            item["tasks"] = expand_benchmark(benchmark, values["tasks"], True)
            result["router"]["benchmarks"][benchmark] = item
    return result
