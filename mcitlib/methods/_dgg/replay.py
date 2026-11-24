# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Prepare original-QA DGG replay, independently seeded for each old task."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .core import image_paths_absolute, replay_rows


def prepare(data_paths: list[str], image_roots: list[str], output: str, samples_per_task: int = 2000,
            seed: int = 1993, manifest: str | None = None, expected_sha256: list[str] | None = None) -> dict:
    """The first path is the current task; later paths are chronological history."""
    if not data_paths or len(data_paths) != len(image_roots):
        raise ValueError("each current/historical dataset needs an image root")
    if expected_sha256 is not None and len(expected_sha256) != len(data_paths):
        raise ValueError("expected SHA-256 count must match data inputs")
    inputs, hashes = [], []
    for index, (filename, root) in enumerate(zip(data_paths, image_roots)):
        raw = Path(filename).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256[index]:
            raise ValueError(f"training data differs from the sample manifest: {filename}")
        hashes.append(digest)
        rows = json.loads(raw)
        if not isinstance(rows, list):
            raise ValueError(f"training input must be a JSON array: {filename}")
        inputs.append(image_paths_absolute(rows, root))
    merged = replay_rows(inputs[0], inputs[1:], samples_per_task=samples_per_task, seed=seed)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(merged, file, ensure_ascii=False)
    temporary.replace(destination)
    report = {"version": 1, "current": len(inputs[0]),
              "replay_by_task": [min(samples_per_task, len(rows)) for rows in inputs[1:]],
              "samples_per_task": samples_per_task, "seed": seed, "output": str(destination),
              "input_sha256": hashes, "output_sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
    if manifest is not None:
        manifest_path = Path(manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
        temporary_manifest.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        temporary_manifest.replace(manifest_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", action="append", required=True)
    parser.add_argument("--image_folder", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples-per-task", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1993)
    parser.add_argument("--manifest")
    parser.add_argument("--expected-sha256", action="append")
    args = parser.parse_args()
    print(json.dumps(prepare(args.data_path, args.image_folder, args.output, args.samples_per_task, args.seed,
                             args.manifest, args.expected_sha256)))


if __name__ == "__main__":
    main()
