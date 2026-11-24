# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 MCITlib authors
"""Prepare repeated Octopus samples and task-one stage-two checkpoint."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .core import sample_indices
from .state import digest, load_manifest, write_manifest


def sample(source: Path, output: Path, count: int) -> None:
    rows = json.loads(source.read_text(encoding='utf-8'))
    if not isinstance(rows, list) or not rows:
        raise ValueError('Octopus requires a non-empty JSON array')
    selected = [rows[index] for index in sample_indices(len(rows), count)]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(selected, handle, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action', required=True)
    copy = sub.add_parser('copy-first')
    copy.add_argument('--input', type=Path, required=True)
    copy.add_argument('--output_dir', type=Path, required=True)
    prep = sub.add_parser('sample')
    prep.add_argument('--data_path', type=Path, required=True)
    prep.add_argument('--count', type=int, required=True)
    prep.add_argument('--output', type=Path, required=True)
    clean = sub.add_parser('cleanup')
    clean.add_argument('--input', type=Path, required=True)
    clean.add_argument('--checkpoint', type=Path, required=True)
    clean.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'sample':
        sample(args.data_path, args.output, args.count)
    elif args.action == 'copy-first':
        if not (args.input / 'adapter_model.bin').is_file():
            raise FileNotFoundError('stage-1 task-1 adapter is missing')
        shutil.copytree(args.input, args.output_dir)
        write_manifest(args.output_dir, [])
    else:
        if not args.input.name.endswith('_gpwc.pt'):
            raise ValueError('cleanup only accepts generated Octopus GPWC files')
        import torch
        load_manifest(args.checkpoint)
        manifest = json.loads((args.checkpoint / 'octopus_manifest.json').read_text())
        actual = digest(args.input)
        if manifest.get('gpwc', {}).get('sha256') != actual:
            raise ValueError('stage-two manifest does not confirm this GPWC artifact')
        audit = (args.checkpoint / manifest['gpwc']['audit']).resolve()
        if audit != args.input.with_suffix(args.input.suffix + '.audit.json').resolve():
            raise ValueError('stage-two manifest refers to a different GPWC audit')
        if digest(audit) != manifest['gpwc']['audit_sha256']:
            raise ValueError('stage-two GPWC audit checksum mismatch')
        weights = torch.load(args.checkpoint / 'adapter_model.bin', map_location='cpu')
        if not weights or not any('lora_A' in name for name in weights) or not any('lora_B' in name for name in weights):
            raise ValueError('saved Octopus adapter could not be reloaded')
        receipt = {'version': 1, 'gpwc_sha256': actual, 'gpwc_bytes': args.input.stat().st_size,
                   'adapter_sha256': manifest['adapter_sha256'], 'history': manifest['history'],
                   'audit': manifest['gpwc']['audit'], 'audit_sha256': manifest['gpwc']['audit_sha256'],
                   'adapter_reload': 'passed'}
        with args.receipt.open('x', encoding='utf-8') as handle:
            json.dump(receipt, handle, indent=2)
        args.input.unlink()


if __name__ == '__main__':
    main()
