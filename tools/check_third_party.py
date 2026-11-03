# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Verify vendored upstream trees and their provenance manifests.

With no patches, ``MANIFEST.json`` is a byte-for-byte allowlist of the files
extracted from the upstream commit. It lets CI run without network access. If a
component gains a patch, run this tool with ``--upstream-root`` containing the
corresponding upstream clones; patch-aware verification is intentionally strict
rather than silently accepting a changed upstream file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

METADATA_FILES = {"UPSTREAM.md", "MANIFEST.json"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_files(component: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(component.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(component)
        if relative.parts[0] == "patches" or relative.name in METADATA_FILES:
            continue
        result[str(relative)] = digest(path)
    return result


def write_manifest(component: Path) -> None:
    upstream = component / "UPSTREAM.md"
    if not upstream.is_file():
        raise FileNotFoundError(f"missing {upstream}")
    commit = "unknown"
    for line in upstream.read_text(encoding="utf-8").splitlines():
        if line.startswith("- Commit:") or line.startswith("- Tag/commit:"):
            commit = line.split("`")[1] if "`" in line else line.split(":", 1)[1].strip()
            break
    manifest = {
        "schema_version": 1,
        "component": component.name,
        "upstream_commit": commit,
        "files": source_files(component),
    }
    (component / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {component / 'MANIFEST.json'} ({len(manifest['files'])} files)")


def check_component(component: Path, upstream_root: Path | None) -> list[str]:
    errors: list[str] = []
    if not (component / "UPSTREAM.md").is_file():
        return [f"missing UPSTREAM.md: {component}"]
    manifest_path = component / "MANIFEST.json"
    if not manifest_path.is_file():
        return [f"missing MANIFEST.json: {component}; run check_third_party.py --write-manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["files"]
    actual = source_files(component)
    for name in sorted(set(expected) - set(actual)):
        errors.append(f"missing file: {component.name}/{name}")
    for name in sorted(set(actual) - set(expected)):
        errors.append(f"unexpected file: {component.name}/{name}")
    for name in sorted(set(expected) & set(actual)):
        if expected[name] != actual[name]:
            errors.append(f"changed file: {component.name}/{name}")
    patches = component / "patches"
    patch_files = sorted(patches.glob("*.patch")) if patches.is_dir() else []
    if patch_files:
        if upstream_root is None:
            errors.append(f"{component.name}: patches require --upstream-root for verification")
        else:
            clone = upstream_root / component.name
            if not clone.is_dir():
                errors.append(f"{component.name}: upstream clone not found: {clone}")
            else:
                commit = manifest["upstream_commit"]
                verified = subprocess.run(
                    ["git", "-C", str(clone), "cat-file", "-e", f"{commit}^{{commit}}"],
                    capture_output=True,
                    check=False,
                )
                if verified.returncode:
                    errors.append(f"{component.name}: upstream commit unavailable: {commit}")
                for patch in patch_files:
                    applied = subprocess.run(
                        ["git", "-C", str(clone), "apply", "--check", str(patch)],
                        capture_output=True,
                        check=False,
                    )
                    if applied.returncode:
                        errors.append(f"{component.name}: patch does not apply: {patch.name}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."), help="MCITlib repository root")
    parser.add_argument("--upstream-root", type=Path, help="directory containing upstream clones for patch checks")
    parser.add_argument("--write-manifest", action="store_true", help="regenerate manifests from the current pristine trees")
    args = parser.parse_args()
    root = args.root.resolve() / "third_party"
    if not root.exists():
        print("third_party/: not vendored yet")
        return 0
    components = sorted(path for path in root.iterdir() if path.is_dir())
    if args.write_manifest:
        for component in components:
            write_manifest(component)
        return 0
    errors = [error for component in components for error in check_component(component, args.upstream_root)]
    for error in errors:
        print(error)
    print(f"third-party components: {len(components)}; errors: {len(errors)}")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
