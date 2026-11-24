# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Convert recorded method YAML into shared ordered templates and task deltas.

This is a migration aid, not a build step: the compact YAML is the maintained
source of truth. SOURCE must be a separate, unmodified checkout of the older
configuration and OUTPUT is a directory that will receive common/ and methods/.
"""
from __future__ import annotations

import argparse
import collections
import copy
import re
from pathlib import Path
from typing import Any

import yaml

from mcitlib.config.compact import _option_positions


class FlowDict(dict):
    pass


class FlowList(list):
    pass


class Dumper(yaml.SafeDumper):
    pass


Dumper.add_representer(FlowDict, lambda dumper, value: dumper.represent_mapping("tag:yaml.org,2002:map", value, flow_style=True))
Dumper.add_representer(FlowList, lambda dumper, value: dumper.represent_sequence("tag:yaml.org,2002:seq", value, flow_style=True))


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(value, Dumper=Dumper, sort_keys=False, allow_unicode=True, width=140), encoding="utf-8")


def intersection(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {}
    result: dict[str, Any] = {}
    for key, value in items[0].items():
        if not all(key in other for other in items):
            continue
        if all(isinstance(other[key], dict) for other in items):
            shared = intersection([other[key] for other in items])
            if shared or all(other[key] == {} for other in items):
                result[key] = shared
        elif all(other[key] == value for other in items):
            result[key] = copy.deepcopy(value)
    return result


def delta(base: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    if set(base) - set(actual):
        raise ValueError(f"defaults contain keys absent from a task: {set(base) - set(actual)}")
    result: dict[str, Any] = {}
    for key, value in actual.items():
        if key not in base:
            result[key] = value
        elif isinstance(base[key], dict) and isinstance(value, dict):
            nested = delta(base[key], value)
            if nested:
                result[key] = nested
        elif base[key] != value or type(base[key]) is not type(value):
            result[key] = value
    return result


def flatten(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    if isinstance(value, dict):
        result = []
        for key, child in value.items():
            result.extend(flatten(child, f"{prefix}.{key}" if prefix else key))
        return result
    if isinstance(value, (str, int, float, bool)) and len(str(value)) >= 6:
        return [(str(value), prefix)]
    return []


def contextual(value: str, context: dict[str, Any]) -> str:
    """Replace complete known values with readable references, longest first."""
    if "@{" in value:
        raise ValueError(f"recorded value contains reserved reference syntax: {value}")
    candidates = flatten(context)
    by_value: dict[str, str] = {}
    for text, name in candidates:
        by_value.setdefault(text, name)
    choices = sorted(by_value, key=lambda x: (-len(x), x))
    if not choices:
        return value
    pattern = re.compile("|".join(re.escape(item) for item in choices))
    return pattern.sub(lambda match: "@{" + by_value[match.group()] + "}", value)


def signature(argv: list[str]) -> tuple[str | None, ...]:
    positions = set(_option_positions(argv).values())
    return tuple(None if index in positions else token for index, token in enumerate(argv))


def mode(items: list[str]) -> str:
    counts = collections.Counter(items)
    return max(counts, key=lambda x: (counts[x], -items.index(x)))


def factor_fragments(templates: dict[str, Any]) -> dict[str, FlowList]:
    """Share repeated consecutive option groups without changing their order."""
    streams: dict[str, list[Any]] = {}
    for name, template in templates.items():
        argv = template["argv"]
        positions = set(_option_positions(argv).values())
        atoms: list[Any] = []
        i = 0
        while i < len(argv):
            if i + 1 in positions:
                atoms.append((argv[i], argv[i + 1]))
                i += 2
            else:
                atoms.append((argv[i],))
                i += 1
        streams[name] = atoms
    fragments: dict[str, FlowList] = {}
    names: collections.Counter[str] = collections.Counter()
    while True:
        counts: collections.Counter[tuple[Any, ...]] = collections.Counter()
        for atoms in streams.values():
            for start in range(len(atoms)):
                for end in range(start + 1, min(start + 20, len(atoms)) + 1):
                    part = atoms[start:end]
                    if any(not isinstance(atom, tuple) or not atom[0].startswith("--") for atom in part):
                        break
                    if len(part) >= 3:
                        counts[tuple(part)] += 1
        choices = [(sum(len(atom) for atom in part) * (count - 1) - 2, part)
                   for part, count in counts.items() if count > 1]
        if not choices:
            break
        gain, best = max(choices, key=lambda item: (item[0], len(item[1])))
        if gain < 14:
            break
        first, last = best[0][0][2:], best[-1][0][2:]
        stem = re.sub(r"[^a-z0-9]+", "_", f"options_{first}_to_{last}".lower()).strip("_")
        names[stem] += 1
        name = stem if names[stem] == 1 else f"{stem}_{names[stem]}"
        fragments[name] = FlowList([token for atom in best for token in atom])
        for template, atoms in streams.items():
            result = []
            index = 0
            while index < len(atoms):
                if tuple(atoms[index:index + len(best)]) == best:
                    result.append({"use": name})
                    index += len(best)
                else:
                    result.append(atoms[index])
                    index += 1
            streams[template] = result
    for name, atoms in streams.items():
        templates[name]["argv"] = FlowList([token for atom in atoms for token in (atom if isinstance(atom, tuple) else (FlowDict(atom),))])
    return fragments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="legacy configs directory")
    parser.add_argument("--output", type=Path, required=True, help="destination configs directory")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if source == output:
        parser.error("source and output must be different directories")
    methods = {p.stem: yaml.safe_load(p.read_text(encoding="utf-8")) for p in sorted((source / "methods").glob("*.yaml"))}
    if not methods or any("overrides" not in values or "format" in values for values in methods.values()):
        parser.error("source must contain the original expanded method YAML")
    first = next(iter(methods.values()))
    provenance = intersection([v for data in methods.values() for v in data["source_metadata"].values()])
    # The old provenance included absolute paths from the recording machine.
    # They are not used for execution; fixtures retain the original evidence.
    provenance.pop("path_placeholders", None)
    provenance["sequence_repairs"] = []
    provenance["config_aliases"] = {}
    common: dict[str, Any] = {
        "task": {"model_config": first["overrides"]["llava"]["ucit"]["task_1"]["model_config"],
                 "source": {
                     "model_config": "model_configs/llava.json",
                     "data_config": "data_configs/@{benchmark.v3_name}/@{task.task_name}.json",
                     "train_config": "train_configs/@{method.v3_dir}/LLaVA/@{benchmark.v3_name}/@{split.train}/task@{task.task_id}.json",
                     "eval_config": "train_configs/@{method.v3_dir}/LLaVA/@{benchmark.v3_name}/@{split.eval}/task@{task.task_id}.json",
                 }, "removed_install_commands": [["pip", "install", "-e", "."]]},
        "benchmarks": {},
    }
    benchmarks = {"ucit": "UCIT", "mllm_dcl": "MLLM-DCL", "mllm_acl": "MLLM-ACL"}
    for benchmark, name in benchmarks.items():
        tasks: dict[str, Any] = {}
        for task_key in first["overrides"]["llava"][benchmark]:
            if not task_key.startswith("task_"):
                continue
            samples = [item["overrides"]["llava"][benchmark][task_key] for item in methods.values()]
            samples.append(methods["mrlora"]["router"]["benchmarks"][benchmark]["tasks"][task_key])
            tasks[task_key] = intersection([{key: task[key] for key in ("task_id", "task_name", "data_config")} for task in samples])
        common["benchmarks"][benchmark] = {"v3_name": name, "tasks": tasks}
    pending: list[tuple[dict[str, Any], dict[str, Any], str]] = []

    def pack_record(record: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        transformed = [contextual(value, context) for value in record["argv"]]
        # These are semantic links, including short numerical values that a
        # substring matcher cannot safely infer. Inference model paths belong
        # to eval_config even where their initial values equal train outputs.
        links = {
            "--lora_r": "task.train_config.rank",
            "--num_train_epochs": "task.train_config.epoch",
            "--per_device_train_batch_size": "task.train_config.batch_size",
            "--gradient_accumulation_steps": "task.train_config.grad_acc",
            "--learning_rate": "task.train_config.lr",
            "--model-path": "task.eval_config.model_path" if record.get("kind") == "eval" else "task.train_config.output_dir",
            "--model-base": "task.model_config.model_name",
            "--data_path": "task.data_config.train_path",
            "--image_folder": "task.data_config.train_folder",
            "--output_dir": "task.train_config.output_dir",
            "--previous_task_model_path": "task.train_config.previous_model",
        }
        from mcitlib.config.compact import _lookup
        for flag, position in _option_positions(record["argv"]).items():
            ref = links.get(flag)
            if ref:
                try:
                    candidate = _lookup(context, ref)
                except ValueError:
                    continue
                if str(candidate) == record["argv"][position]:
                    transformed[position] = f"@{{{ref}}}"
        positions = _option_positions(record["argv"])
        if record.get("kind") == "eval" and "--question-file" in positions and "--image-folder" in positions:
            question = record["argv"][positions["--question-file"]]
            folder = record["argv"][positions["--image-folder"]]
            current = context.get("task", {}).get("data_config", {})
            if current.get("test_path") == question and current.get("test_folder") == folder:
                transformed[positions["--image-folder"]] = "@{task.data_config.test_folder}"
            else:
                for task_key, item in context.get("datasets", {}).items():
                    data = item.get("data_config", {})
                    if data.get("test_path") == question and data.get("test_folder") == folder:
                        transformed[positions["--image-folder"]] = f"@{{datasets.{task_key}.data_config.test_folder}}"
                        break
        output_record = {"_argv": transformed}
        output_record.update({key: copy.deepcopy(value) for key, value in record.items() if key not in {"argv", "args", "launcher", "entry", "positional"}})
        pending.append((output_record, record, record.get("kind", "other")))
        return output_record

    def pack_benchmark(method: dict[str, Any], benchmark: str, tasks: dict[str, Any], router: bool) -> dict[str, Any]:
        task_items = {key: value for key, value in tasks.items() if key.startswith("task_")}
        defaults = intersection([{key: task[key] for key in ("train_config", "eval_config")} for task in task_items.values()])
        result: dict[str, Any] = {}
        if defaults:
            result["task_defaults"] = defaults
        for key, task in tasks.items():
            if key == "pre_commands":
                ctx = {"method": {k: method[k] for k in ("id", "v3_dir")}, "benchmark": {"v3_name": benchmarks[benchmark]},
                       "datasets": common["benchmarks"][benchmark]["tasks"]}
                result[key] = [pack_record(record, ctx) for record in task]
                continue
            if not key.startswith("task_"):
                result[key] = task
                continue
            dataset = common["benchmarks"][benchmark]["tasks"][key]
            base = copy.deepcopy(common["task"])
            from mcitlib.config.compact import _interpolate
            from mcitlib.config.loader import deep_merge
            base = deep_merge(deep_merge(base, dataset), defaults)
            ctx = {"method": {k: method[k] for k in ("id", "v3_dir")},
                   "benchmark": {"v3_name": benchmarks[benchmark]},
                   "task": {"task_name": task["task_name"], "task_id": task["task_id"]},
                   "split": {"train": "train_router" if router else "train", "eval": "eval_router" if router else "eval"}}
            base = _interpolate(base, ctx)
            ordinary = {k: v for k, v in task.items() if k not in {"train_command", "eval_commands", "extra_commands", "other_commands"}}
            spec = delta(base, ordinary)
            ctx = {"task": ordinary, "datasets": common["benchmarks"][benchmark]["tasks"],
                   "method": {k: method[k] for k in ("id", "v3_dir")}, "benchmark": {"v3_name": benchmarks[benchmark]}}
            for field in ("train_command", "eval_commands", "extra_commands", "other_commands"):
                if field in task:
                    spec[field] = pack_record(task[field], ctx) if field == "train_command" else [pack_record(record, ctx) for record in task[field]]
            result[key] = spec
        return result

    packed: dict[str, Any] = {}
    for method_name, method in methods.items():
        out = {"format": "mcitlib-method-v1"}
        out.update({key: value for key, value in method.items() if key not in {"source_metadata", "defaults", "overrides", "router"}})
        out["source_metadata"] = {benchmark: delta(provenance, {k: v for k, v in value.items() if k != "path_placeholders"})
                                  for benchmark, value in method["source_metadata"].items()}
        out["source_metadata"] = {k: v for k, v in out["source_metadata"].items() if v}
        out["overrides"] = {"llava": {benchmark: pack_benchmark(method, benchmark, tasks, False)
                                       for benchmark, tasks in method["overrides"]["llava"].items()}}
        if "router" in method:
            out["router"] = {key: value for key, value in method["router"].items() if key != "benchmarks"}
            out["router"]["benchmarks"] = {}
            for benchmark, values in method["router"]["benchmarks"].items():
                item = {key: value for key, value in values.items() if key != "tasks"}
                item["tasks"] = pack_benchmark(method, benchmark, values["tasks"], True)
                out["router"]["benchmarks"][benchmark] = item
        packed[method_name] = out

    groups: dict[tuple[str | None, ...], list[tuple[dict[str, Any], str]]] = collections.defaultdict(list)
    for record, _original, kind in pending:
        groups[signature(record["_argv"])].append((record, kind))
    names: collections.Counter[str] = collections.Counter()
    templates: dict[str, Any] = {}
    for sig, records in groups.items():
        role = records[0][1]
        argv = [mode([record["_argv"][i] for record, _ in records]) if part is None else part for i, part in enumerate(sig)]
        entry = next((part.rsplit("/", 1)[-1].removesuffix(".py").rsplit(".", 1)[-1] for part in sig if isinstance(part, str) and (part.endswith(".py") or part.startswith("llava."))), "command")
        stem = re.sub(r"[^a-z0-9]+", "_", f"{role}_{entry}".lower()).strip("_")
        names[stem] += 1
        name = f"{stem}_{names[stem]}"
        templates[name] = {"argv": FlowList(argv)}
        positions = _option_positions(argv)
        for record, _ in records:
            old = record.pop("_argv")
            sets = {flag: old[position] for flag, position in positions.items() if old[position] != argv[position]}
            reordered = FlowDict({"template": name})
            if sets:
                reordered["set"] = FlowDict(sets)
            reordered.update(record)
            record.clear()
            record.update(reordered)
    # Keep each command record compact while leaving ordered option names readable.
    def flow_records(obj: Any) -> Any:
        if isinstance(obj, dict):
            if "template" in obj:
                return FlowDict({key: flow_records(value) for key, value in obj.items()})
            return {key: flow_records(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [flow_records(value) for value in obj]
        return obj
    fragments = factor_fragments(templates)
    dump(output / "common/tasks.yaml", common)
    dump(output / "common/commands.yaml", {"fragments": fragments, "templates": templates})
    dump(output / "common/provenance.yaml", provenance)
    for method_name, values in packed.items():
        dump(output / "methods" / f"{method_name}.yaml", flow_records(values))
    print(f"wrote {len(packed)} methods, {len(fragments)} shared option groups, {len(templates)} ordered templates and {len(pending)} command records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
