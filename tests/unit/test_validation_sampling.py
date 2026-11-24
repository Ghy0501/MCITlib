# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Behavior checks for the isolated validation-subset generator."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from mcitlib.methods._keeplora.config_paths import resolve_fields
from mcitlib.paths import PathResolver
from tools.sample_validation_data import (
    derived_seed,
    make_caption_reference,
    pair_key,
    proportional_quotas,
    run,
    select_rows,
    stratum,
)


def training(image: str, question: str, answer: str, qid: str = "same") -> dict:
    return {"question_id": qid, "image": image, "conversations": [
        {"from": "human", "value": f"<image>\n{question}"},
        {"from": "gpt", "value": answer},
    ]}


def evaluation_row(image: str, question: str, answer: str, qid: str = "same") -> dict:
    return {"question_id": qid, "image": image, "text": question, "answer": answer}


def test_proportional_allocation_preserves_small_groups_and_rejects_shortage():
    assert proportional_quotas({"large": 90, "mid": 9, "small": 1}, 10) == {
        "large": 8, "mid": 1, "small": 1,
    }
    assert proportional_quotas({"a": 3, "b": 2}, 5) == {"a": 3, "b": 2}
    assert sum(proportional_quotas({"a": 20, "b": 2, "c": 1}, 2).values()) == 2
    with pytest.raises(ValueError, match="cannot sample"):
        proportional_quotas({"a": 2}, 3)


def test_determinism_and_question_pair_exclusion_does_not_use_ids():
    train_rows = [training("images/shared.jpg", "What  is it?", "yes", "train-id")]
    candidates = [evaluation_row("images/shared.jpg", " What is  it?\n", "no", "different-id"),
                  evaluation_row("images/shared.jpg", "What color?", "red", "train-id"),
                  evaluation_row("images/other.jpg", "What is it?", "yes", "train-id"),
                  evaluation_row("images/third.jpg", "A third question?", "4")]
    unchanged = copy.deepcopy(candidates)
    excluded = {pair_key(row) for row in train_rows}
    seed = derived_seed(20261106, "ucit", "task", "test", "v1")
    indices, stats = select_rows(candidates, 3, "answer_kind", seed, excluded)
    assert indices == [1, 2, 3]
    assert stats["excluded_matching_train_pairs"] == 1
    assert select_rows(candidates, 3, "answer_kind", seed, excluded) == (indices, stats)
    assert candidates == unchanged
    assert candidates[indices[0]]["image"] == "images/shared.jpg"
    with pytest.raises(ValueError, match="cannot sample"):
        select_rows(candidates, 4, "answer_kind", seed, excluded)

    many = [evaluation_row(f"images/{i}.jpg", "What?", "yes" if i < 70 else "42")
            for i in range(100)]
    indices, stats = select_rows(many, 20, "answer_kind", seed)
    assert stats["sample_strata"] == {"numeric": 6, "yes_no": 14}
    assert indices == select_rows(many, 20, "answer_kind", seed)[0]
    assert indices != select_rows(many, 20, "answer_kind", derived_seed(1, "ucit", "task", "test", "v1"))[0]
    assert indices != list(range(20))


def test_named_strata_follow_source_image_directories():
    assert stratum(training("ImageNet-R/train/n01443537/x.jpg", "What?", "goldfish"), "imagenet_class") == "n01443537"
    assert stratum(training("IconQA/train/choose_txt/123/image.png", "What?", "A"), "iconqa_kind") == "choose_txt"
    assert stratum(evaluation_row("images/SciVerse_123.png", "What?", "A"), "sci_source") == "SciVerse"
    with pytest.raises(ValueError, match="unrecognized Sci"):
        stratum(evaluation_row("images/Unknown_123.png", "What?", "A"), "sci_source")


def test_coco_reference_matches_filenames_and_keeps_all_five_annotations():
    tests = [evaluation_row("imgs/first.jpg", "caption", "one", "2"),
             evaluation_row("imgs/second.jpg", "caption", "two", "1"),
             evaluation_row("imgs/third.jpg", "caption", "three", "1")]
    images = [{"id": 17, "file_name": "third.jpg", "width": 2},
              {"id": 3, "file_name": "first.jpg", "width": 4},
              {"id": 91, "file_name": "second.jpg", "width": 8}]
    annotations = [{"id": old_id, "image_id": old_id, "caption": f"{old_id}-{n}", "extra": n}
                   for old_id in (17, 3, 91) for n in range(5)]
    source = {"images": images, "annotations": annotations, "info": {"owner": "test"}}
    original = copy.deepcopy(source)
    result, mapping = make_caption_reference(source, tests, [0, 2])
    assert source == original
    assert [i["id"] for i in result["images"]] == [1, 2]
    assert [i["file_name"] for i in result["images"]] == ["first.jpg", "third.jpg"]
    assert [a["caption"] for a in result["annotations"]] == [f"3-{n}" for n in range(5)] + [f"17-{n}" for n in range(5)]
    assert [a["id"] for a in result["annotations"]] == list(range(1, 11))
    assert [a["image_id"] for a in result["annotations"]] == [1] * 5 + [2] * 5
    assert [m["source_coco_image_id"] for m in mapping] == [3, 17]
    assert mapping[0]["source_annotation_indices"] == [5, 6, 7, 8, 9]
    assert [m["question_id"] for m in mapping] == ["2", "1"]
    bad = copy.deepcopy(source)
    bad["images"][0]["file_name"] = "first.jpg"
    with pytest.raises(ValueError, match="duplicate"):
        make_caption_reference(bad, tests, [0, 2])


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_generation_verification_paths_and_no_overwrite(tmp_path: Path, monkeypatch):
    project = tmp_path / "project"
    annotation = project / "source/annotations"
    images = project / "source/images"
    images.mkdir(parents=True)
    for i in range(6):
        Image.new("RGB", (2, 2), (i, 0, 0)).save(images / f"{i}.png")
    train = [training(f"{i}.png", "describe it?", f"caption {i}") for i in range(3)]
    test = [evaluation_row("0.png", "describe   it?", "caption 0", "different"),
            evaluation_row("3.png", "describe it?", "caption 3"),
            evaluation_row("4.png", "describe it?", "caption 4"),
            evaluation_row("5.png", "describe it?", "caption 5")]
    source = {"images": [{"id": i * 11 + 7, "file_name": f"{name}.png"} for i, name in enumerate((0, 3, 4, 5))],
              "annotations": [{"id": i, "image_id": i * 11 + 7, "caption": f"{i} caption {n}"}
                              for i in range(4) for n in range(5)],
              "info": {"description": "example"}}
    dump_json(annotation / "train.json", train)
    dump_json(annotation / "test.json", test)
    dump_json(annotation / "val_coco_type_3000.json", source)
    recipe = {"version": "test-v1", "seed": 20261106, "train_count": 3, "test_count": 2,
              "decode_samples_per_split": 1,
              "tasks": [{"benchmark": "ucit", "name": "Flickr30k",
                         "logical_dir": "${data_root}/UCIT/Flickr30k-cap",
                         "image_root": "${data_root}/UCIT", "stratifier": "caption_length",
                         "caption_reference": "val_coco_type_3000.json"}]}
    recipe_file = project / "configs/recipe.json"
    dump_json(recipe_file, recipe)
    paths = project / "configs/paths.yaml"
    paths.write_text(yaml.safe_dump({
        "model_root": str(project / "models"), "data_root": str(project / "logical"),
        "ckpt_root": str(tmp_path / "old_external_ckpts"), "path_mappings": {"ucit": [
            {"from": "${data_root}/UCIT", "to": str(images)},
            {"from": "${data_root}/UCIT/Flickr30k-cap", "to": str(annotation)},
            {"from": "${data_root}/UCIT/Flickr30k-cap/router.json", "to": str(annotation / "full_router.json")},
        ], "mllm_acl": [{"from": "${data_root}/MLLM-CL", "to": str(project / "full_acl")}]}
    }), encoding="utf-8")
    output, manifest, profile = project / "data/validation_3k_1k", project / "configs/manifest.json", project / "configs/paths.validation.yaml"
    args = (recipe_file, paths, output, manifest, profile)
    run(*args, project=project)
    report = json.loads(manifest.read_text(encoding="utf-8"))
    task = report["tasks"][0]
    assert task["splits"]["test"]["excluded_matching_train_pairs"] == 1
    indices = task["splits"]["test"]["indices_zero_based"]
    sampled = json.loads((output / "ucit/Flickr30k/test.json").read_text(encoding="utf-8"))
    assert sampled == [test[i] for i in indices]
    assert all(row["question_id"] == "same" for row in sampled)
    sample_ref = json.loads((output / "ucit/Flickr30k/val_coco_type_3000.json").read_text(encoding="utf-8"))
    assert [item["file_name"] for item in sample_ref["images"]] == [row["image"] for row in sampled]
    assert len(sample_ref["annotations"]) == 10
    resolver = PathResolver.from_file(profile, repo_root=project)
    fields = resolve_fields({"train_path": "/your_data_path/UCIT/Flickr30k-cap/train_brief_4w.json",
                             "train_folder": "/your_data_path/UCIT"},
                            ("train_path", "train_folder"), resolver.context("ucit"))
    assert fields["train_path"] == str(output / "ucit/Flickr30k/train.json")
    assert fields["train_folder"] == str(images)
    assert resolver.resolve("${data_root}/UCIT/Flickr30k-cap/test_3000.json", benchmark="ucit") == output / "ucit/Flickr30k/test.json"
    for extra in ("router.json", "replay.json"):
        path = resolver.resolve(f"${{data_root}}/UCIT/Flickr30k-cap/{extra}", benchmark="ucit")
        assert path == output / "unavailable/ucit/Flickr30k" / extra
        assert not path.exists()
    assert not resolver.resolve("${data_root}/MLLM-CL/OCR/train.json", benchmark="mllm_acl").exists()
    assert resolver.paths.ckpt_root == project / "ckpts/validation_3k_1k"
    assert yaml.safe_load(profile.read_text(encoding="utf-8"))["ckpt_root"] == "./ckpts/validation_3k_1k"
    monkeypatch.chdir(tmp_path)
    relative_args = (Path("configs/recipe.json"), Path("configs/paths.yaml"),
                     Path("data/validation_3k_1k"), Path("configs/manifest.json"),
                     Path("configs/paths.validation.yaml"))
    run(*relative_args, project=project, verify=True)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run(*args, project=project)
    (output / "ucit/Flickr30k/test.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="sample differs"):
        run(*args, project=project, verify=True)
