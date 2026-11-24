#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Prepare Replay QA mixtures and MR-LoRA router samples with traceable sources."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from mcitlib.paths import PathResolver, managed_path  # noqa: E402
from tools.sample_validation_data import (  # noqa: E402
    _rename_noreplace,
    json_bytes,
    question_and_answer,
    read_json,
    sha256,
)

ALIASES = {"CLEVR-Math": "CLEVR"}
QUESTION_START = "Here is the user's question:"
QUESTION_END = "You only need to select the suitable model and do not answer the question."


def normalize_question(value: str) -> str:
    return " ".join(value.replace("<image>", " ").split())


def router_question(row: dict[str, Any]) -> str:
    human, _ = question_and_answer(row)
    if human.count(QUESTION_START) != 1 or human.count(QUESTION_END) != 1:
        raise ValueError("unrecognized router prompt; cannot recover original question")
    return normalize_question(human.split(QUESTION_START, 1)[1].split(QUESTION_END, 1)[0])


def image_path(root: Path, image: str) -> Path:
    value = PurePosixPath(image)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError(f"image must be a safe relative path: {image}")
    path = (root / image).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing image: {path}")
    return path


def router_key(row: dict[str, Any], benchmark: str) -> tuple[str, str, str, str]:
    value = row.get("question_id")
    if not isinstance(value, str) or "/" not in value:
        raise ValueError("router question_id must contain its task prefix")
    prefix, original_id = value.split("/", 1)
    image = row.get("image")
    if not isinstance(image, str) or not image.startswith(prefix + "/"):
        raise ValueError(f"router image and task prefix disagree: {value}, {image}")
    source_image = image[len(prefix) + 1:] if benchmark == "mllm_dcl" else image
    return prefix, original_id, source_image, router_question(row)


def _read_bytes_json(path: Path) -> tuple[Any, bytes]:
    raw = path.read_bytes()
    return json.loads(raw), raw


def build_bundle(project: Path, source_paths: Path, recipe_file: Path, sample_manifest_file: Path,
                 router_configs: Path, output: Path, staging: Path, *, pool_size: int = 20,
                 sample_count: int = 3000) -> dict[str, Any]:
    """Build into staging; leave all sample inputs and the recipe/manifest unchanged."""
    resolver = PathResolver.from_file(source_paths, repo_root=project)
    recipe, recipe_hash = read_json(recipe_file)
    sample_manifest, sample_manifest_hash = read_json(sample_manifest_file)
    tasks_by_scope = {scope: [task for task in recipe["tasks"] if task["benchmark"] == scope]
                      for scope in ("ucit", "mllm_dcl")}
    manifest_by_task = {(task["benchmark"], task["name"]): task for task in sample_manifest["tasks"]}
    all_tasks = {(task["benchmark"], task["name"]): task for task in recipe["tasks"]}
    if len(all_tasks) != sum(map(len, tasks_by_scope.values())) or set(all_tasks) != set(manifest_by_task):
        raise ValueError("recipe and sampling manifest tasks do not agree")
    bundle = {"version": "validation-aux-v1", "recipe_sha256": recipe_hash,
              "sample_manifest_sha256": sample_manifest_hash, "source_paths": str(source_paths),
              "router_config_root": str(router_configs), "pool_size_per_task": pool_size,
              "sample_count_per_task": sample_count, "tasks": []}
    overlay: dict[str, Any] = {"path_mappings": {scope: [] for scope in tasks_by_scope}}
    router_data: dict[tuple[str, int], tuple[list[dict], bytes, Path, Path, str]] = {}
    targets: dict[tuple[str, str], dict[tuple[str, str, str], None]] = defaultdict(dict)
    task_aliases: dict[tuple[str, str], str] = {}
    for scope, tasks in tasks_by_scope.items():
        benchmark_dir = "UCIT" if scope == "ucit" else "MLLM-DCL"
        prefixes = {ALIASES.get(task["name"], task["name"]): task["name"] for task in tasks}
        if len(prefixes) != len(tasks):
            raise ValueError("ambiguous task-to-router prefix")
        previous: set[tuple[str, str, str, str]] = set()
        for stage, task in enumerate(tasks, 1):
            name = task["name"]
            alias = ALIASES.get(name, name)
            task_aliases[(scope, name)] = alias
            config_file = router_configs / "data_configs_router" / benchmark_dir / f"{name}.json"
            config, config_hash = read_json(config_file)
            router_file = Path(config["train_path"]).resolve()
            rows, raw = _read_bytes_json(router_file)
            if not isinstance(rows, list):
                raise ValueError(f"router JSON must be an array: {router_file}")
            expected_root = resolver.resolve("${data_root}/UCIT" if scope == "ucit" else "${data_root}/MLLM-CL",
                                             benchmark=scope)
            if Path(config["train_folder"]).resolve() != expected_root:
                raise ValueError(f"router image root differs from configured benchmark root: {config_file}")
            keyed = [router_key(row, scope) for row in rows]
            expected = set(prefixes_for_tasks(tasks[:stage]))
            expected_counts = Counter({name: pool_size for name in expected})
            # Original UCIT task 1 is empty; task 2 supplies the first task's 20 rows below.
            counts = Counter(key[0] for key in keyed)
            if not (scope == "ucit" and stage == 1 and not rows) and counts != expected_counts:
                raise ValueError(f"unexpected router task pool in {router_file}: {counts}, expected {expected_counts}")
            current = set(keyed)
            if len(current) != len(keyed) or not previous.issubset(current):
                raise ValueError(f"router pool contains duplicates or dropped previous members: {router_file}")
            previous = current
            router_data[(scope, stage)] = (rows, raw, router_file, config_file, config_hash)
            for prefix, original_id, image, question in keyed:
                if prefix not in prefixes:
                    raise ValueError(f"unknown task prefix {prefix} in {router_file}")
                targets[(scope, prefixes[prefix])][(original_id, image, question)] = None
        if tasks_by_scope[scope] and scope == "ucit":
            if len(tasks) < 2:
                raise ValueError("UCIT router task 1 requires the task 2 source pool")
            first_alias = ALIASES.get(tasks[0]["name"], tasks[0]["name"])
            rows, _, router_file, _, _ = router_data[(scope, 2)]
            if sum(router_key(row, scope)[0] == first_alias for row in rows) != pool_size:
                raise ValueError(f"task 2 router pool lacks the first UCIT task: {router_file}")

    # Index only requested examples, keeping the first source index for captions.
    originals: dict[tuple[str, str], dict[tuple[str, str, str], dict[str, Any]]] = {}
    sources: dict[tuple[str, str], dict[str, Any]] = {}
    for key, task in all_tasks.items():
        scope, name = key
        split = manifest_by_task[key]["splits"]["train"]
        source_file = resolver.resolve(split["source"], benchmark=scope)
        records, source_hash = read_json(source_file)
        if source_hash != split["source_sha256"]:
            raise ValueError(f"source training data changed since sample generation: {source_file}")
        wanted = targets[key]
        wanted_pairs = {(wanted_id, wanted_image) for wanted_id, wanted_image, _ in wanted}
        found: dict[tuple[str, str, str], dict[str, Any]] = {}
        for index, row in enumerate(records):
            image = row.get("image")
            if not isinstance(image, str):
                continue
            ids = {str(row[field]) for field in ("question_id", "id") if row.get(field) is not None}
            if not ids:
                continue
            # Avoid decoding question fields for source rows outside the selected pool.
            if not any((ident, image) in wanted_pairs for ident in ids):
                continue
            human, _ = question_and_answer(row)
            question = normalize_question(human)
            for ident in ids:
                identity = (ident, image, question)
                if identity in wanted:
                    if identity not in found:
                        found[identity] = {"index": index, "row": copy.deepcopy(row), "matching_rows": 0}
                    found[identity]["matching_rows"] += 1
        missing = set(wanted) - set(found)
        if missing:
            raise ValueError(f"cannot restore {len(missing)} original QA rows in {source_file}; first: {next(iter(missing))}")
        originals[key] = found
        sources[key] = {"source": str(source_file), "source_sha256": source_hash,
                        "image_root": str(resolver.resolve(task["image_root"], benchmark=scope))}
        del records

    relative_output = output.relative_to(project).as_posix()
    for scope, tasks in tasks_by_scope.items():
        replay_root = resolver.resolve("${data_root}/UCIT" if scope == "ucit" else "${data_root}/MLLM-CL",
                                       benchmark=scope)
        for stage, task in enumerate(tasks, 1):
            name = task["name"]
            task_dir = staging / scope / name
            task_dir.mkdir(parents=True)
            router_rows, router_raw, router_file, config_file, config_hash = router_data[(scope, stage)]
            router_indices = list(range(len(router_rows)))
            if scope == "ucit" and stage == 1:
                router_rows, _, router_file, _, _ = router_data[(scope, 2)]
                alias = task_aliases[(scope, name)]
                router_indices = [index for index, row in enumerate(router_rows) if router_key(row, scope)[0] == alias]
                router_raw = json_bytes([router_rows[index] for index in router_indices])
            (task_dir / "router.json").write_bytes(router_raw)
            sample_path = project / "data/validation_3k_1k" / scope / name / "train.json"
            sampled, sample_hash = read_json(sample_path)
            split = manifest_by_task[(scope, name)]["splits"]["train"]
            if sample_hash != split["output_sha256"] or len(sampled) != sample_count:
                raise ValueError(f"sample training data changed or has wrong length: {sample_path}")
            mixture: list[dict[str, Any]] = []
            old_metadata = []
            digest_source, digest_output = hashlib.sha256(), hashlib.sha256()
            def append_row(record: dict[str, Any], source_task: str) -> None:
                row = copy.deepcopy(record)
                original_image = row["image"]
                source_root = Path(sources[(scope, source_task)]["image_root"])
                source_image = image_path(source_root, original_image)
                if scope == "mllm_dcl":
                    row["image"] = f"{source_task}/{original_image}"
                output_image = image_path(replay_root, row["image"])
                if source_image != output_image:
                    raise ValueError(f"image path changed: {source_image} != {output_image}")
                digest_source.update((str(source_image) + "\n").encode())
                digest_output.update((str(output_image) + "\n").encode())
                mixture.append(row)
            for row in sampled:
                append_row(row, name)
            prior = {task_aliases[(scope, item["name"])]: item["name"] for item in tasks[:stage - 1]}
            for router_index, row in enumerate(router_rows):
                prefix, original_id, original_image, question = router_key(row, scope)
                if prefix not in prior:
                    continue
                old_task = prior[prefix]
                found = originals[(scope, old_task)][(original_id, original_image, question)]
                append_row(found["row"], old_task)
                old_metadata.append({"task": old_task, "router_row_index_zero_based": router_index,
                                     "router_question_id": row["question_id"], "source_index_zero_based": found["index"],
                                     "candidate_count": found["matching_rows"],
                                     "source_image": original_image,
                                     "physical_image": str(image_path(replay_root, mixture[-1]["image"]))})
            if len(old_metadata) != pool_size * (stage - 1):
                raise ValueError(f"wrong number of previous-task examples for {scope}/{name}")
            replay_raw = json_bytes(mixture)
            (task_dir / "replay.json").write_bytes(replay_raw)
            logical = task["logical_dir"]
            for kind in ("replay", "router"):
                overlay["path_mappings"][scope].append({
                    "from": f"{logical}/{kind}.json",
                    "to": f"${{repo_root}}/{relative_output}/{scope}/{name}/{kind}.json"})
            bundle["tasks"].append({"benchmark": scope, "name": name, "stage": stage,
                                    "sample_train": str(sample_path), "sample_train_sha256": sample_hash,
                                    "source_train": sources[(scope, name)],
                                    "router_config": str(config_file), "router_config_sha256": config_hash,
                                    "router_source": str(router_file), "router_source_sha256": sha256(router_file.read_bytes()),
                                    "router_source_indices_zero_based": router_indices,
                                    "router_count": len(router_indices), "router_output_sha256": sha256(router_raw),
                                    "replay_count": len(mixture), "replay_current_count": len(sampled),
                                    "replay_previous_count": len(old_metadata), "previous_rows": old_metadata,
                                    "replay_output_sha256": sha256(replay_raw),
                                    "physical_image_paths_checked": len(mixture),
                                    "source_physical_paths_sha256": digest_source.hexdigest(),
                                    "replay_physical_paths_sha256": digest_output.hexdigest()})
    (staging / "paths.yaml").write_text(yaml.safe_dump(overlay, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (staging / "manifest.json").write_bytes(json_bytes(bundle))
    return bundle


def prefixes_for_tasks(tasks: list[dict[str, Any]]) -> list[str]:
    return [ALIASES.get(task["name"], task["name"]) for task in tasks]


def run(project: Path, source_paths: Path, recipe_file: Path, sample_manifest_file: Path,
        router_configs: Path, output: Path, *, verify: bool = False) -> dict[str, Any]:
    project = project.resolve()
    def absolute(path: Path) -> Path:
        return (path if path.is_absolute() else project / path).resolve()
    source_paths, recipe_file, sample_manifest_file, router_configs = map(
        absolute, (source_paths, recipe_file, sample_manifest_file, router_configs))
    output = managed_path(output, project, label="auxiliary output")
    if output == project:
        raise ValueError("auxiliary output cannot be the project root")
    if not verify and (output.exists() or output.is_symlink()):
        raise FileExistsError(f"refusing to overwrite auxiliary output: {output}")
    if verify and not output.is_dir():
        raise FileNotFoundError(f"auxiliary output does not exist: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".validation-aux-", dir=output.parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        bundle = build_bundle(project, source_paths, recipe_file, sample_manifest_file, router_configs, output, staging)
        if verify:
            expected = {p.relative_to(staging) for p in staging.rglob("*") if p.is_file()}
            actual = {p.relative_to(output) for p in output.rglob("*") if p.is_file()}
            if expected != actual or any((staging / path).read_bytes() != (output / path).read_bytes() for path in expected):
                raise ValueError("auxiliary bundle differs from its sources or manifest")
        else:
            _rename_noreplace(staging, output)
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-paths", type=Path, default=Path("configs/paths.yaml"))
    parser.add_argument("--recipe", type=Path, default=Path("configs/validation_3k_1k/recipe.json"))
    parser.add_argument("--sample-manifest", type=Path, default=Path("configs/validation_3k_1k/manifest.json"))
    parser.add_argument("--router-config-root", type=Path,
                        default=PROJECT.parents[1] / "MCITlib/MR-LoRA/configs")
    parser.add_argument("--output", type=Path, default=Path("data/validation_aux"))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    try:
        bundle = run(PROJECT, args.source_paths, args.recipe, args.sample_manifest, args.router_config_root,
                     args.output, verify=args.verify)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"validation auxiliary: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "verified" if args.verify else "created", "output": str(args.output),
                      "tasks": len(bundle["tasks"]), "pool_size": bundle["pool_size_per_task"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
