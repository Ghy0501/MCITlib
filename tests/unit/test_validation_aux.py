# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Replay/router preparation preserves QA answers, source provenance and image roots."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from tools.generate_validation_aux import build_bundle
from tools.sample_validation_data import json_bytes, sha256


def dump(path: Path, value) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json_bytes(value)
    path.write_bytes(content)
    return sha256(content)


def test_auxiliary_uses_exact_qa_and_physical_images(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    source_root = project / "source"
    configs = project / "router-configs"
    paths = project / "paths.yaml"
    paths.write_text(yaml.safe_dump({"data_root": str(source_root), "model_root": str(project / "models")}))
    recipe_tasks, manifest_tasks = [], []
    router_files = {}
    for scope, group, names in (("ucit", "UCIT", ("ImageNet-R", "CLEVR-Math")),
                                ("mllm_dcl", "MLLM-CL", ("RS", "Med"))):
        accumulated = []
        for stage, name in enumerate(names, 1):
            prefix = "CLEVR" if name == "CLEVR-Math" else name
            logical_dir = f"${{data_root}}/{group}/{name}"
            image_root = f"${{data_root}}/{group}" if scope == "ucit" else logical_dir
            recipe_tasks.append({"benchmark": scope, "name": name, "logical_dir": logical_dir,
                                 "image_root": image_root})
            records = []
            for index in range(2):
                original_image = f"{prefix}/{index}.jpg" if scope == "ucit" else f"{index}.jpg"
                physical = source_root / group / original_image if scope == "ucit" else source_root / group / name / original_image
                physical.parent.mkdir(parents=True, exist_ok=True)
                physical.write_bytes(b"image")
                records.append({"question_id": str(index), "image": original_image, "conversations": [
                    {"from": "human", "value": f"<image>\nquestion {index}?"},
                    {"from": "gpt", "value": f"original answer {index}"}]})
                accumulated.append({"question_id": f"{prefix}/{index}",
                                    "image": original_image if scope == "ucit" else f"{prefix}/{original_image}",
                                    "conversations": [
                                        {"from": "human", "value": "<image>\nrouter prompt\nHere is the user's question: "
                                         f"question {index}?\nYou only need to select the suitable model and do not answer the question."},
                                        {"from": "gpt", "value": "A"}]})
            # A second caption with the same id, image and question must not replace the first answer.
            if name == "ImageNet-R":
                duplicate = json.loads(json.dumps(records[0]))
                duplicate["conversations"][1]["value"] = "other caption"
                full_records = records + [duplicate]
            else:
                full_records = records
            train = source_root / group / name / "train.json"
            train_hash = dump(train, full_records)
            sampled = project / "data/validation_3k_1k" / scope / name / "train.json"
            sample_hash = dump(sampled, records)
            manifest_tasks.append({"benchmark": scope, "name": name, "splits": {"train": {
                "source": f"{logical_dir}/train.json", "source_sha256": train_hash, "output_sha256": sample_hash}}})
            router = project / "router" / scope / f"task{stage}.json"
            dump(router, [] if scope == "ucit" and stage == 1 else list(accumulated))
            router_files[(scope, stage)] = router
            folder = "UCIT" if scope == "ucit" else "MLLM-DCL"
            dump(configs / "data_configs_router" / folder / f"{name}.json", {
                "train_path": str(router), "train_folder": str(source_root / group)})
    recipe = project / "recipe.json"
    sample_manifest = project / "sample-manifest.json"
    dump(recipe, {"tasks": recipe_tasks})
    dump(sample_manifest, {"tasks": manifest_tasks})
    staging = project / "staging"
    staging.mkdir()
    output = project / "data/validation_aux"
    manifest = build_bundle(project, paths, recipe, sample_manifest, configs, output, staging,
                            pool_size=2, sample_count=2)
    indexed = {(row["benchmark"], row["name"]): row for row in manifest["tasks"]}
    ucit = indexed[("ucit", "CLEVR-Math")]
    assert ucit["router_output_sha256"] == sha256(router_files[("ucit", 2)].read_bytes())
    assert indexed[("ucit", "ImageNet-R")]["router_count"] == 2
    assert ucit["replay_current_count"] == 2 and ucit["replay_previous_count"] == 2
    assert ucit["previous_rows"][0]["source_index_zero_based"] == 0
    assert ucit["previous_rows"][0]["candidate_count"] == 2
    replay = json.loads((staging / "ucit/CLEVR-Math/replay.json").read_text())
    assert replay[2]["conversations"][1]["value"] == "original answer 0"
    assert replay[2]["image"] == "ImageNet-R/0.jpg"
    dcl = json.loads((staging / "mllm_dcl/Med/replay.json").read_text())
    assert [row["image"] for row in dcl] == ["Med/0.jpg", "Med/1.jpg", "RS/0.jpg", "RS/1.jpg"]
    assert dcl[2]["conversations"][1]["value"] == "original answer 0"
    assert all(row["source_physical_paths_sha256"] == row["replay_physical_paths_sha256"]
               for row in manifest["tasks"])
    overlay = yaml.safe_load((staging / "paths.yaml").read_text())
    assert len(overlay["path_mappings"]["ucit"]) == 4
    assert json.loads(router_files[("ucit", 1)].read_text()) == []
    # A router question that cannot be found in the source data must fail closed.
    changed = json.loads(router_files[("ucit", 2)].read_text())
    changed[0]["conversations"][0]["value"] = changed[0]["conversations"][0]["value"].replace("question 0?", "wrong?")
    dump(router_files[("ucit", 2)], changed)
    other = project / "bad-stage"
    other.mkdir()
    with pytest.raises(ValueError, match="cannot restore"):
        build_bundle(project, paths, recipe, sample_manifest, configs, output, other, pool_size=2, sample_count=2)
