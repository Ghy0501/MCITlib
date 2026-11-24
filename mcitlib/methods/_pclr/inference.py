# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 MCITlib authors
"""Load a PCLR boundary over the same LLaVA base and generate through its pools."""
from pathlib import Path

from .core import RankPool
from .llava import RoutedLlava, keep_norms_fp32


def install_builder():
    import llava.model.builder as builder

    original = builder.load_pretrained_model

    def load_pretrained_model(model_path, model_base, model_name, *args, **kwargs):
        if not model_base:
            raise ValueError('PCLR inference needs --model-base')
        tokenizer, base, processor, length = original(model_base, None, 'llava', *args, **kwargs)
        keep_norms_fp32(base)
        pool, phase = RankPool.load_boundary(base, Path(model_path) / 'pclr.pt')
        if phase not in {'learned', 'integrated'}:
            raise ValueError(f'unsupported PCLR checkpoint phase {phase}')
        model = RoutedLlava(pool, stage='eval')
        model.eval()
        return tokenizer, model, processor, length

    builder.load_pretrained_model = load_pretrained_model
