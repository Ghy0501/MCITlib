# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Import-only stand-in for flash_attn.flash_attn_interface."""


def flash_attn_varlen_qkvpacked_func(*args, **kwargs):
    raise RuntimeError("fake flash_attn: import check only")


flash_attn_unpadded_qkvpacked_func = flash_attn_varlen_qkvpacked_func
