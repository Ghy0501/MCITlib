# SPDX-License-Identifier: Apache-2.0 AND MIT
# Copyright (c) 2025 Jinpeng Chen
# Copyright (c) 2025-2026 MCITlib authors
"""Any-resolution helpers required by SEFE's newer LLaVA model snapshot."""
from __future__ import annotations

import ast


def select_best_resolution(original_size, possible_resolutions):
    """Select the resolution exactly as in SEFE's v3 LLaVA snapshot."""
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")
    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width = int(original_width * scale)
        downscaled_height = int(original_height * scale)
        effective_resolution = min(
            downscaled_width * downscaled_height,
            original_width * original_height,
        )
        wasted_resolution = (width * height) - effective_resolution
        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution
            and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)
    return best_fit


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """Return the selected image patch grid, preserving v3 conversions."""
    possible_resolutions = grid_pinpoints if type(grid_pinpoints) is list else ast.literal_eval(grid_pinpoints)
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size
