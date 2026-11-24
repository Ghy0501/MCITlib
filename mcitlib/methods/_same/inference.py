# SPDX-License-Identifier: MIT
"""Install SAME prior restoration at the normal LLaVA inference builder."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .prior import FEATURE_DIM, SIDECAR, PriorState, load_clip_runtime


def install_builder():
    import llava.model.builder as builder

    original = builder.load_pretrained_model
    if getattr(original, '_same_prior_builder', False):
        return

    def load_pretrained_model(model_path, model_base, model_name, *args, **kwargs):
        checkpoint = Path(model_path)
        if not (checkpoint / SIDECAR).is_file():
            raise FileNotFoundError(
                f'SAME checkpoint lacks {SIDECAR}: {checkpoint}. Its image/text anchors cannot '
                'be inferred from a legacy adapter; use the frozen legacy source for old results.')
        with (checkpoint / 'adapter_config.json').open(encoding='utf8') as handle:
            config = json.load(handle)
        state = PriorState(int(config['expert_num']), FEATURE_DIM)
        state.load(checkpoint)
        state.require_observed_through(int(config['cur_task']))
        tokenizer, model, processor, length = original(model_path, model_base, model_name, *args, **kwargs)
        clip_path = os.environ.get('MCITLIB_SAME_CLIP_TOWER') or getattr(model.config, 'mm_vision_tower', None)
        param = next(model.parameters())
        runtime = load_clip_runtime(state, tokenizer, clip_path, device=param.device,
                                    dtype=param.dtype, task_id=int(config['cur_task']))
        runtime.install_generate_hook(model)
        model.eval()
        return tokenizer, model, processor, length

    load_pretrained_model._same_prior_builder = True
    builder.load_pretrained_model = load_pretrained_model
