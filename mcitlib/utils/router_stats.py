# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""Optional router diagnostics; task identity comes from the run context."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RouterStats:
    """Write v3-compatible top-2 counts only when explicitly enabled."""

    task_id: int
    output_dir: Path
    enabled: bool = False

    def record(self, router: Any) -> None:
        if not self.enabled:
            return
        import torch

        _, indices = torch.topk(router, 2, dim=-1)
        invalid = router_topk_values_equal(router, indices)
        indices[invalid] = -1
        counts = Counter(indices.detach().cpu().flatten().tolist())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / f"value_counts_{self.task_id}.txt").open("a", encoding="utf-8") as handle:
            for value, count in sorted(counts.items()):
                handle.write(f"{value}:{count}\n")


def router_topk_values_equal(router: Any, indices: Any) -> Any:
    """Select the two logged values without changing the router tensor."""
    import torch

    values = torch.gather(router, -1, indices)
    return values[..., 0] == values[..., 1]
