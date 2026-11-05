# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Import-only stand-in for flash_attn.bert_padding."""


def _unavailable(*args, **kwargs):
    raise RuntimeError("fake flash_attn: import check only")


unpad_input = pad_input = index_first_axis = _unavailable
