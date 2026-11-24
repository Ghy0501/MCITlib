#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Make the fixed UCIT/DCL validation subset without changing source annotations."""
from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from mcitlib.paths import PathResolver  # noqa: E402

TRAIN_ALIASES = ("train.json", "train_4w.json", "train_brief_4w.json")
TEST_ALIASES = ("test.json", "test_3000.json")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def read_json(path: Path) -> tuple[Any, str]:
    data = path.read_bytes()
    return json.loads(data), sha256(data)


def question_and_answer(row: dict[str, Any]) -> tuple[str, str]:
    if "conversations" in row:
        conversations = row["conversations"]
        human = next((item.get("value") for item in conversations if item.get("from") == "human"), None)
        answer = next((item.get("value") for item in conversations if item.get("from") == "gpt"), None)
    else:
        human, answer = row.get("text"), row.get("answer")
    if not isinstance(human, str) or not isinstance(answer, str) or not isinstance(row.get("image"), str):
        raise ValueError("each record must have a string image, question, and answer")
    return human, answer


def pair_key(row: dict[str, Any]) -> tuple[str, str]:
    question, _ = question_and_answer(row)
    return row["image"], " ".join(question.replace("<image>", " ").split())


def stratum(row: dict[str, Any], kind: str) -> str:
    question, answer = question_and_answer(row)
    image = PurePosixPath(row["image"])
    if kind == "imagenet_class":
        name = image.parent.name
        if not re.fullmatch(r"n\d{8}", name):
            raise ValueError(f"unrecognized ImageNet-R class in {image}")
        return name
    if kind == "iconqa_kind":
        groups = set(image.parts) & {"choose_txt", "fill_in_blank"}
        if len(groups) != 1:
            raise ValueError(f"unrecognized IconQA kind in {image}")
        return next(iter(groups))
    if kind == "sci_source":
        name = image.name.split("_", 1)[0]
        if name not in {"MapQA", "AI2D", "TQA", "SciVerse"}:
            raise ValueError(f"unrecognized Sci source in {image}")
        return name
    if kind == "caption_length":
        words = len(answer.split())
        return "1-8_words" if words <= 8 else "9-16_words" if words <= 16 else "17+_words"
    if kind == "answer_kind":
        answer = answer.strip().lower().rstrip(".! ")
        if answer in {"yes", "no"}:
            return "yes_no"
        options = bool(re.search(r"(?:\b[a-e][.)]\s*.*\b[b-e][.)]|option.s letter|multiple.choice)", question, re.I | re.S))
        if options or (len(answer) == 1 and answer in "abcde"):
            return "multiple_choice"
        if re.fullmatch(r"[+-]?(?:\d[\d,]*)(?:\.\d+)?%?", answer):
            return "numeric"
        return "open"
    raise ValueError(f"unknown stratifier: {kind}")


def proportional_quotas(counts: dict[str, int], size: int) -> dict[str, int]:
    """Proportional integer allocation, with one per group when possible.

    Exact integer deficits determine additions/removals; ties use group name.
    """
    if size < 0 or any(n <= 0 for n in counts.values()) or size > sum(counts.values()):
        raise ValueError(f"cannot sample {size} rows from {sum(counts.values())}")
    if not counts:
        if size:
            raise ValueError("cannot sample from an empty source")
        return {}
    total = sum(counts.values())
    minimum = int(size >= len(counts))
    quotas = {name: max(minimum, n * size // total) for name, n in counts.items()}
    while sum(quotas.values()) < size:
        eligible = [name for name in counts if quotas[name] < counts[name]]
        name = min(eligible, key=lambda x: (-(counts[x] * size - quotas[x] * total), x))
        quotas[name] += 1
    while sum(quotas.values()) > size:
        eligible = [name for name in counts if quotas[name] > minimum]
        name = min(eligible, key=lambda x: (-(quotas[x] * total - counts[x] * size), x))
        quotas[name] -= 1
    return dict(sorted(quotas.items()))


def derived_seed(seed: int, benchmark: str, task: str, split: str, version: str) -> str:
    return sha256(f"{version}|{seed}|{benchmark}|{task}|{split}".encode("utf-8"))


def select_rows(rows: list[dict[str, Any]], size: int, kind: str, seed: str,
                excluded: set[tuple[str, str]] | None = None) -> tuple[list[int], dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    source_counts: Counter[str] = Counter()
    excluded_count = 0
    for index, row in enumerate(rows):
        group = stratum(row, kind)
        source_counts[group] += 1
        if excluded is not None and pair_key(row) in excluded:
            excluded_count += 1
        else:
            groups[group].append(index)
    counts = {group: len(indices) for group, indices in groups.items()}
    quotas = proportional_quotas(counts, size)
    seed_bytes = bytes.fromhex(seed)
    selected = []
    for group in sorted(groups):
        ordered = sorted(groups[group], key=lambda i: (hashlib.sha256(seed_bytes + i.to_bytes(8, "big")).digest(), i))
        selected.extend(ordered[:quotas[group]])
    selected.sort()
    return selected, {
        "source_strata": dict(sorted(source_counts.items())),
        "eligible_strata": dict(sorted(counts.items())),
        "sample_strata": quotas,
        "excluded_matching_train_pairs": excluded_count,
    }


def make_caption_reference(source: dict[str, Any], test_rows: list[dict[str, Any]],
                           indices: list[int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Align by filename, not by question_id or assumed source COCO image_id."""
    images = source["images"]
    annotations = source["annotations"]
    by_file: dict[str, tuple[int, dict[str, Any]]] = {}
    seen_ids: set[Any] = set()
    for index, item in enumerate(images):
        name, image_id = item["file_name"], item["id"]
        if name in by_file or image_id in seen_ids:
            raise ValueError(f"duplicate COCO file_name or image id: {name!r}, {image_id!r}")
        by_file[name] = (index, item)
        seen_ids.add(image_id)
    by_id: dict[Any, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, item in enumerate(annotations):
        by_id[item["image_id"]].append((index, item))
    result = copy.deepcopy(source)
    result["images"], result["annotations"] = [], []
    mapping = []
    used_files = set()
    for new_id, source_index in enumerate(indices, start=1):
        row = test_rows[source_index]
        name = PurePosixPath(row["image"]).name
        if name in used_files or name not in by_file:
            raise ValueError(f"COCO file_name has no unique corresponding test image: {name}")
        used_files.add(name)
        source_image_index, image = by_file[name]
        source_id = image["id"]
        references = by_id[source_id]
        if len(references) != 5:
            raise ValueError(f"expected 5 captions for {name}, got {len(references)}")
        new_image = copy.deepcopy(image)
        new_image["id"] = new_id
        result["images"].append(new_image)
        new_annotation_ids = []
        for _, annotation in references:
            new_annotation = copy.deepcopy(annotation)
            new_annotation["image_id"] = new_id
            # Source files reuse annotation.id within an image; pycocotools requires unique IDs.
            new_annotation["id"] = len(result["annotations"]) + 1
            new_annotation_ids.append(new_annotation["id"])
            result["annotations"].append(new_annotation)
        mapping.append({
            "test_source_index": source_index,
            "question_id": row.get("question_id"),
            "file_name": name,
            "source_coco_image_index": source_image_index,
            "source_coco_image_id": source_id,
            "sample_coco_image_id": new_id,
            "source_annotation_indices": [i for i, _ in references],
            "sample_annotation_ids": new_annotation_ids,
        })
    return result, mapping


def check_images(rows: list[dict[str, Any]], image_root: Path, decode_count: int) -> dict[str, int]:
    if decode_count < 0:
        raise ValueError("decode sample count cannot be negative")
    from PIL import Image

    for index, row in enumerate(rows):
        value = row["image"]
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"image path must be relative to its original image folder: {value}")
        image_file = image_root.joinpath(*path.parts)
        if not image_file.is_file():
            raise FileNotFoundError(f"missing image for selected row {index}: {image_file}")
        if index < decode_count:
            with Image.open(image_file) as image:
                image.load()
    return {"existence_checked": len(rows), "decoded_first_rows": min(len(rows), decode_count)}


def _logical_output(project: Path, path: Path) -> str:
    return "${repo_root}/" + path.resolve().relative_to(project.resolve()).as_posix()


def make_validation_profile(base: dict[str, Any], resolver: PathResolver, tasks: list[dict[str, Any]],
                            project: Path, output_dir: Path) -> dict[str, Any]:
    """Retain UCIT/DCL image and model mappings, adding exact-file overrides."""
    profile = copy.deepcopy(base)
    root = _logical_output(project, output_dir)
    mappings = base.get("path_mappings", {})
    profile["path_mappings"] = {}
    # root fields do not expand ${repo_root}; resolve relative roots as PathResolver does.
    profile["ckpt_root"] = "./ckpts/validation_3k_1k"
    profile["results_root"] = "./results/validation_3k_1k"
    profile["logs_root"] = "./logs/validation_3k_1k"
    bare = PathResolver(resolver.paths)
    for scope in ("ucit", "mllm_dcl"):
        retained = copy.deepcopy(mappings.get("common", []) + mappings.get(scope, []))
        additions = []
        for task in (item for item in tasks if item["benchmark"] == scope):
            logical_dir = task["logical_dir"]
            sample_dir = f"{root}/{scope}/{task['name']}"
            for alias in TRAIN_ALIASES:
                additions.append({"from": f"{logical_dir}/{alias}", "to": f"{sample_dir}/train.json"})
            for alias in TEST_ALIASES:
                additions.append({"from": f"{logical_dir}/{alias}", "to": f"{sample_dir}/test.json"})
            if task.get("caption_reference"):
                ref = task["caption_reference"]
                additions.append({"from": f"{logical_dir}/{ref}", "to": f"{sample_dir}/{ref}"})
            for auxiliary in ("replay.json", "router.json"):
                additions.append({"from": f"{logical_dir}/{auxiliary}",
                                  "to": f"{root}/unavailable/{scope}/{task['name']}/{auxiliary}"})
        new_sources = {os.path.normpath(bare.resolve_templates(entry["from"])) for entry in additions}
        retained = [entry for entry in retained
                    if os.path.normpath(bare.resolve_templates(entry["from"])) not in new_sources]
        profile["path_mappings"][scope] = retained + additions
    for scope in ("mllm_acl", "general"):
        profile["path_mappings"][scope] = [{"from": "${data_root}", "to": f"{root}/unavailable/{scope}"}]
    return profile


def build_bundle(recipe: dict[str, Any], recipe_hash: str, resolver: PathResolver,
                 project: Path, final_output: Path, stage_data: Path) -> dict[str, Any]:
    tasks = recipe["tasks"]
    version, base_seed = recipe["version"], recipe["seed"]
    manifest: dict[str, Any] = {
        "version": version,
        "recipe_sha256": recipe_hash,
        "seed": base_seed,
        "sampling": "SHA-256 rank of derived split seed and zero-based source row index; "
                    "integer proportional quotas with one per nonempty stratum when possible; "
                    "selected rows sorted by source index",
        "overlap_rule": "exclude test candidates with (image, normalized question) matching sampled train; "
                        "normalization removes <image> and collapses whitespace; IDs are ignored",
        "caption_rule": "match file_name; retain five captions; sequential sample image_id and unique "
                        "annotation.id; source annotation rows are indexed because source IDs may repeat",
        "unavailable_inputs": ["replay.json", "router.json", "mllm_acl", "general"],
        "tasks": [],
    }
    for task in tasks:
        scope, name = task["benchmark"], task["name"]
        out_dir = stage_data / scope / name
        out_dir.mkdir(parents=True)
        task_manifest: dict[str, Any] = {"benchmark": scope, "name": name,
                                         "stratifier": task["stratifier"], "splits": {}}
        chosen_train_keys: set[tuple[str, str]] = set()
        selected_test = None
        for split in ("train", "test"):
            logical = f"{task['logical_dir']}/{split}.json"
            source_path = resolver.resolve(logical, benchmark=scope)
            rows, source_hash = read_json(source_path)
            if not isinstance(rows, list):
                raise ValueError(f"source must be a JSON array: {logical}")
            aliases = task.get(f"{split}_aliases", [])
            if split == "test" and scope == "ucit":
                aliases = list(aliases) + recipe.get("ucit_test_aliases", [])
            alias_manifest = []
            for alias in aliases:
                alias_logical = f"{task['logical_dir']}/{alias}"
                _, alias_hash = read_json(resolver.resolve(alias_logical, benchmark=scope))
                if alias_hash != source_hash:
                    raise ValueError(f"source alias differs from {logical}: {alias_logical}")
                alias_manifest.append({"source": alias_logical, "sha256": alias_hash})
            seed = derived_seed(base_seed, scope, name, split, version)
            indices, stats = select_rows(rows, recipe[f"{split}_count"], task["stratifier"], seed,
                                         chosen_train_keys if split == "test" else None)
            selected = [rows[i] for i in indices]
            if split == "train":
                chosen_train_keys = {pair_key(row) for row in selected}
            else:
                if chosen_train_keys & {pair_key(row) for row in selected}:
                    raise ValueError(f"selected train/test overlap in {scope}/{name}")
                selected_test = rows, indices
            image_root = resolver.resolve(task["image_root"], benchmark=scope)
            image_checks = check_images(selected, image_root, recipe["decode_samples_per_split"])
            content = json_bytes(selected)
            (out_dir / f"{split}.json").write_bytes(content)
            task_manifest["splits"][split] = {
                "source": logical,
                "source_sha256": source_hash,
                "source_aliases": alias_manifest,
                "source_count": len(rows),
                "sample_count": len(selected),
                "derived_seed_sha256": seed,
                "indices_zero_based": indices,
                "output": f"{_logical_output(project, final_output)}/{scope}/{name}/{split}.json",
                "output_sha256": sha256(content),
                "image_root": task["image_root"],
                "image_checks": image_checks,
                **stats,
            }
        if task.get("caption_reference"):
            ref_name = task["caption_reference"]
            ref_logical = f"{task['logical_dir']}/{ref_name}"
            reference, reference_hash = read_json(resolver.resolve(ref_logical, benchmark=scope))
            assert selected_test is not None
            output, mapping = make_caption_reference(reference, *selected_test)
            content = json_bytes(output)
            (out_dir / ref_name).write_bytes(content)
            task_manifest["caption_reference"] = {
                "source": ref_logical, "source_sha256": reference_hash,
                "output": f"{_logical_output(project, final_output)}/{scope}/{name}/{ref_name}",
                "output_sha256": sha256(content), "images": len(output["images"]),
                "annotations": len(output["annotations"]), "mapping": mapping,
            }
        manifest["tasks"].append(task_manifest)
        print(f"checked {scope}/{name}", flush=True)
    return manifest


def _safe_output(project: Path, value: Path) -> Path:
    project = project.resolve()
    target = (value if value.is_absolute() else project / value).resolve()
    if target == project or project not in target.parents:
        raise ValueError(f"output must be inside project: {target}")
    return target


def _rename_noreplace(source: Path, target: Path) -> None:
    """Atomically publish a directory without ever replacing an existing path (Linux)."""
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
        renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        renameat2.restype = ctypes.c_int
        result = renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1)
    except AttributeError:
        # glibc before 2.28 has no wrapper although supported Linux kernels expose the syscall.
        syscall_numbers = {"x86_64": 316, "aarch64": 276}
        number = syscall_numbers.get(platform.machine()) if sys.platform == "linux" else None
        if number is None:
            raise OSError("atomic no-replace directory publication requires renameat2")
        syscall = libc.syscall
        syscall.restype = ctypes.c_long
        result = syscall(number, -100, os.fsencode(source), -100, os.fsencode(target), 1)
    if result != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(target))


def run(recipe_file: Path, source_paths: Path, output_dir: Path, manifest_path: Path,
        profile_path: Path, *, project: Path = PROJECT_ROOT, verify: bool = False) -> None:
    project = project.resolve()
    recipe_file = (recipe_file if recipe_file.is_absolute() else project / recipe_file).resolve()
    source_paths = (source_paths if source_paths.is_absolute() else project / source_paths).resolve()
    output_dir, manifest_path, profile_path = (_safe_output(project, path)
                                               for path in (output_dir, manifest_path, profile_path))
    targets = (output_dir, manifest_path, profile_path)
    for index, target in enumerate(targets):
        for other in targets[index + 1:]:
            if target == other or target in other.parents or other in target.parents:
                raise ValueError("output, manifest and profile paths must not overlap")
        if not verify and os.path.lexists(target):
            raise FileExistsError(f"refusing to overwrite: {target}")
    recipe, recipe_hash = read_json(recipe_file)
    resolver = PathResolver.from_file(source_paths, repo_root=project)
    base = yaml.safe_load(source_paths.read_text(encoding="utf-8")) or {}
    profile = make_validation_profile(base, resolver, recipe["tasks"], project, output_dir)
    # Validate mappings before doing any work, including conflicts with output roots.
    with tempfile.TemporaryDirectory(prefix=".validation-3k1k-", dir=project) as temp:
        stage = Path(temp)
        staged_profile = stage / "paths.validation.yaml"
        staged_profile.write_text(yaml.safe_dump(profile, allow_unicode=True, sort_keys=False), encoding="utf-8")
        PathResolver.from_file(staged_profile, repo_root=project)
        data_stage = stage / "data"
        manifest = build_bundle(recipe, recipe_hash, resolver, project, output_dir, data_stage)
        staged_manifest = stage / "manifest.json"
        staged_manifest.write_bytes(json_bytes(manifest))
        if verify:
            existing, _ = read_json(manifest_path)
            if existing != manifest:
                raise ValueError("manifest differs from regenerated source, selection, or checks")
            existing_profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
            if existing_profile != profile:
                raise ValueError("validation paths configuration differs from the expected profile")
            for staged_file in data_stage.rglob("*.json"):
                installed = output_dir / staged_file.relative_to(data_stage)
                if staged_file.read_bytes() != installed.read_bytes():
                    raise ValueError(f"sample differs from regenerated output: {installed}")
            print("verified source and output hashes, indices, strata, cross-split pairs, images and profile")
            return
        for target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
        data_installed = manifest_installed = profile_installed = False
        try:
            _rename_noreplace(data_stage, output_dir)
            data_installed = True
            os.link(staged_manifest, manifest_path)
            manifest_installed = True
            os.link(staged_profile, profile_path)
            profile_installed = True
        except BaseException:
            if profile_installed:
                profile_path.unlink()
            if manifest_installed:
                manifest_path.unlink()
            if data_installed:
                shutil.rmtree(output_dir)
            raise
    print(f"created {output_dir}, {manifest_path}, and {profile_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=PROJECT_ROOT / "configs/validation_3k_1k/recipe.json")
    parser.add_argument("--source-paths", type=Path, default=PROJECT_ROOT / "configs/paths.yaml")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/validation_3k_1k")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "configs/validation_3k_1k/manifest.json")
    parser.add_argument("--validation-paths", type=Path, default=PROJECT_ROOT / "configs/paths.validation.yaml")
    parser.add_argument("--verify", action="store_true", help="regenerate in temporary storage and compare")
    args = parser.parse_args(argv)
    try:
        run(args.recipe, args.source_paths, args.output_dir, args.manifest, args.validation_paths, verify=args.verify)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"validation subset: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
