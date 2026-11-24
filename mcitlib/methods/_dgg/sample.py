# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Apply the reviewed 3k LLaVA sample budget to DGG without changing replay."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ...registry import get_benchmark
from ...unified_training import load_training_profile


def _value(argv, flag):
    if argv.count(flag) != 1 or argv.index(flag) + 1 >= len(argv):
        raise ValueError(f"DGG sample requires exactly one {flag}")
    return argv[argv.index(flag) + 1]


def _set(argv, flag, value):
    argv[argv.index(flag) + 1] = str(value)


def apply_dgg_sample_profile(commands, benchmark: str, repo_root: Path, manifest_path: Path | None = None):
    """Verify 3k sample bytes and apply lr=2e-4, batch=64, one epoch.

    No files are written here. Replay input hashes are passed to its later
    preparation process and checked again before it writes the replay file.
    """
    profile = load_training_profile(repo_root / "configs/training_profiles/dgg_sample_unified.yaml", repo_root=repo_root)
    if (profile.id != "dgg_sample_unified" or profile.primary_train_samples_per_task != 3000
            or profile.global_batch_size != 64 or profile.max_per_device_train_batch_size != 8
            or profile.primary_default_epochs != 1 or float(profile.base_learning_rate) != 0.0002):
        raise ValueError("DGG sample profile does not match the reviewed 3k/64/1-epoch budget")
    manifest_path = Path(manifest_path or repo_root / "configs/validation_3k_1k/manifest.json")
    raw_manifest = manifest_path.read_bytes()
    manifest = json.loads(raw_manifest)
    rows = {item["name"]: item["splits"]["train"] for item in manifest["tasks"] if item["benchmark"] == benchmark}
    task_names = get_benchmark(benchmark).tasks
    if set(rows) != set(task_names):
        raise ValueError("DGG sample manifest task coverage differs from the selected benchmark")
    train_commands = [c for c in commands if c.kind == "train" and c.env.get("MCITLIB_METHOD") == "dgg"]
    if not train_commands:
        raise ValueError("DGG sample profile requires at least one DGG training command")
    for train in train_commands:
        task_id = train.task_id
        if not isinstance(task_id, int) or not 1 <= task_id <= len(task_names):
            raise ValueError("invalid DGG sample task ID")
        if _value(train.argv, "--include").split(":", 1)[1].count(",") != 0:
            raise ValueError("DGG sample training requires a single GPU")
        if task_id == 1:
            data_paths = [_value(train.argv, "--data_path")]
            names = [task_names[0]]
            replay = None
        else:
            matches = [c for c in commands if c.task_id == task_id and "mcitlib.methods._dgg.replay" in c.argv]
            if len(matches) != 1:
                raise ValueError("DGG sample replay preparation is missing or ambiguous")
            replay = matches[0]
            if "--expected-sha256" in replay.argv:
                raise ValueError("DGG sample profile is already applied")
            data_paths = [replay.argv[i + 1] for i, flag in enumerate(replay.argv[:-1]) if flag == "--data_path"]
            names = [task_names[task_id - 1], *task_names[:task_id - 1]]
        if len(data_paths) != len(names):
            raise ValueError("DGG sample inputs differ from the task history")
        expected = []
        for name, path in zip(names, data_paths):
            split = rows[name]
            if split.get("sample_count") != 3000:
                raise ValueError(f"DGG sample count differs for {name}")
            raw = Path(path).read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            records = json.loads(raw)
            if digest != split.get("output_sha256") or not isinstance(records, list) or len(records) != 3000:
                raise ValueError(f"DGG sample input differs from the manifest: {name}")
            expected.append(digest)
        if replay is None:
            train.argv.extend(["--dgg_expected_data_sha256", expected[0]])
        else:
            for digest in expected:
                replay.argv.extend(["--expected-sha256", digest])
        _set(train.argv, "--learning_rate", profile.base_learning_rate)
        _set(train.argv, "--num_train_epochs", 1)
        _set(train.argv, "--per_device_train_batch_size", 8)
        _set(train.argv, "--gradient_accumulation_steps", 8)
        train.metadata["training_profile"] = {"id": profile.id, "sha256": profile.sha256,
                                                "sample_manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
                                                "primary_samples": 3000, "global_batch_size": 64}
    return commands
