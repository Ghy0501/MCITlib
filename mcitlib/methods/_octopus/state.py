# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 MCITlib authors
"""Portable chronological history manifests for Octopus adapters."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _validate_order(output: Path, history: list[Path]) -> None:
    pattern = re.compile(r"Task(\d+)_llava_lora$")
    match = pattern.fullmatch(output.name)
    if match is None or int(match.group(1)) != len(history) + 1:
        raise ValueError('Octopus current task is inconsistent with history length')
    for index, path in enumerate(history, 1):
        match = pattern.fullmatch(path.name)
        if match is None or int(match.group(1)) != index or path.parent.resolve() != output.parent.resolve():
            raise ValueError('Octopus history must be contiguous and chronological')


def write_manifest(output: Path, history: list[str], gpwc: dict | None = None) -> None:
    output = Path(output)
    _validate_order(output, [Path(item) for item in history])
    entries = []
    for path in map(Path, history):
        weight = path / 'adapter_model.bin'
        if not weight.is_file():
            raise FileNotFoundError(f'history adapter missing: {weight}')
        entries.append({'path': os.path.relpath(path, output), 'sha256': digest(weight)})
    adapter = output / 'adapter_model.bin'
    if not adapter.is_file():
        raise FileNotFoundError(f'current adapter missing: {adapter}')
    payload = {'version': 1, 'merge_order': 'chronological', 'history': entries,
               'adapter_sha256': digest(adapter)}
    if gpwc is not None:
        gpwc = dict(gpwc)
        gpwc['audit'] = os.path.relpath(gpwc['audit'], output)
        payload['gpwc'] = gpwc
    (output / 'octopus_manifest.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')


def load_manifest(output: Path) -> list[Path]:
    output = Path(output).resolve()
    payload = json.loads((output / 'octopus_manifest.json').read_text(encoding='utf-8'))
    if payload.get('version') != 1 or payload.get('merge_order') != 'chronological':
        raise ValueError('unsupported Octopus history manifest')
    if digest(output / 'adapter_model.bin') != payload.get('adapter_sha256'):
        raise ValueError('Octopus current adapter checksum mismatch')
    paths = []
    for item in payload['history']:
        path = (output / item['path']).resolve()
        if path.parent != output.parent or digest(path / 'adapter_model.bin') != item['sha256']:
            raise ValueError(f'Octopus history checksum or directory mismatch: {path}')
        paths.append(path)
    if len(set(paths)) != len(paths) or output in paths:
        raise ValueError('invalid duplicate or cyclic Octopus history')
    _validate_order(output, paths)
    return paths
