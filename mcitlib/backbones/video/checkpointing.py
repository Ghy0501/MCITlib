# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Keep the v4 non-reentrant checkpoint policy across the two Transformers ABIs."""
from contextlib import contextmanager

import torch.utils.checkpoint as torch_checkpoint
from transformers import modeling_utils

from ..llava.olora_checkpoint import non_reentrant_checkpoint_scope


@contextmanager
def video_non_reentrant_scope(args, label):
    with non_reentrant_checkpoint_scope(args, label=label) as observation:
        if observation is None or not hasattr(modeling_utils, "checkpoint"):
            yield observation
            return
        # Transformers 4.40 binds a module-local alias to checkpoint and
        # captures it when gradient_checkpointing_enable assigns its partial.
        before = modeling_utils.checkpoint
        modeling_utils.checkpoint = torch_checkpoint.checkpoint
        try:
            yield observation
        finally:
            modeling_utils.checkpoint = before
