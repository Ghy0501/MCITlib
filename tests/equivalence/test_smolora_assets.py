# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""SMoLoRA explicit asset loading, integrity, and v3 content checks."""
from __future__ import annotations

import pickle
from pathlib import Path

import pytest
import yaml

from mcitlib.methods.smolora import load_instruction_embeddings


def metadata(repo_root: Path) -> dict[str, dict[str, str]]:
    values = yaml.safe_load((repo_root / "configs/methods/smolora.yaml").read_text(encoding="utf-8"))
    return {item["benchmark"]: item for item in values["assets"]["instruction_embeddings"]}


@pytest.mark.parametrize("benchmark", ["ucit", "mllm_dcl", "mllm_acl"])
def test_instruction_assets_match_v3(repo_root: Path, v3_root: Path, benchmark: str) -> None:
    record = metadata(repo_root)[benchmark]
    asset = repo_root / "assets/smolora" / Path(record["path"]).name
    loaded = load_instruction_embeddings(asset, record["sha256"])
    with (v3_root / "LLaVA/SMoLoRA" / asset.name).open("rb") as handle:
        expected = pickle.load(handle).tolist()
    assert loaded == expected


def test_instruction_asset_rejects_hash_mismatch(tmp_path: Path) -> None:
    asset = tmp_path / "ins_emb.pkl"
    asset.write_bytes(b"not a valid embedding asset")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_instruction_embeddings(asset, "0" * 64)
