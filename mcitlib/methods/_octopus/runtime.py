# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 MCITlib authors
"""Load frozen history and apply Octopus stage-two regularization."""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from .core import regularization
from .history import attach_history, pairs_for_regularization


def history_from_env():
    value = json.loads(os.environ.get('MCITLIB_OCTOPUS_HISTORY', '[]'))
    if not isinstance(value, list) or not value or any(not isinstance(item, str) for item in value):
        raise ValueError('Octopus stage two requires ordered history checkpoints')
    return value


def setup_stage_two(model, accumulation):
    history = history_from_env()
    adapters, scales = [], []
    from .state import digest, load_manifest
    for index, path in enumerate(history):
        checkpoint = Path(path)
        predecessors = load_manifest(checkpoint)
        if predecessors != [Path(item).resolve() for item in history[:index]]:
            raise ValueError('Octopus historical checkpoint chronology is inconsistent')
        adapters.append(torch.load(checkpoint / 'adapter_model.bin', map_location='cpu'))
        cfg = json.loads((checkpoint / 'adapter_config.json').read_text())
        scales.append(float(cfg['lora_alpha']) / int(cfg['r']))
    attach_history(model, adapters, scales)
    gpwc_path = Path(os.environ['MCITLIB_OCTOPUS_GPWC'])
    audit_path = gpwc_path.with_suffix(gpwc_path.suffix + '.audit.json')
    audit = json.loads(audit_path.read_text())
    if digest(gpwc_path) != audit.get('gpwc_sha256'):
        raise ValueError('Octopus GPWC checksum mismatch')
    gpwc = torch.load(gpwc_path, map_location='cpu')
    if gpwc.get('version') != 1 or not isinstance(gpwc.get('gradients'), dict):
        raise ValueError('invalid Octopus GPWC artifact')
    expected = [str(Path(item).resolve()) for item in history]
    if gpwc.get('history') != expected:
        raise ValueError('Octopus GPWC history differs from current history')
    return OctopusLoss(model, gpwc['gradients'], int(accumulation),
                       float(os.environ['MCITLIB_OCTOPUS_LAMBDA1']),
                       float(os.environ['MCITLIB_OCTOPUS_LAMBDA2']),
                       {'sha256': audit['gpwc_sha256'], 'audit': str(audit_path),
                        'audit_sha256': digest(audit_path)})


class OctopusLoss:
    def __init__(self, model, gradients, accumulation, lambda_1, lambda_2, provenance):
        if accumulation < 1:
            raise ValueError('gradient accumulation must be positive')
        self.pairs = pairs_for_regularization(model, gradients)
        self.accumulation = accumulation
        self.lambda_1, self.lambda_2 = lambda_1, lambda_2
        self.provenance = provenance

    def extra_loss(self, model, inputs, outputs, base_loss):
        loss, _, _ = regularization(self.pairs, self.lambda_1, self.lambda_2)
        # Accelerate/DeepSpeed divide the full backward loss by GAS. Multiplying
        # this term restores the release's task_loss/GAS + regularizer rule.
        return loss * self.accumulation
