# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""B4: migrated evaluation sources preserve each method's v3 implementation."""
from __future__ import annotations

from pathlib import Path

from mcitlib.registry import BENCHMARKS, METHODS
from mcitlib.runner import build_sequence_commands

COiN_IMPORT = "from llava.eval.CoIN.coin_utils import get_model_name_from_path"
MCIT_IMPORT = "from mcitlib.backbones.llava.eval_runners.coin.coin_utils import get_model_name_from_path"


def _without_added_headers(text: str) -> str:
    lines = text.splitlines()
    while lines and (not lines[0] or lines[0].startswith((
        "# SPDX-License-Identifier:", "# Copyright (c) 2025-2026 MCITlib authors",
    ))):
        lines.pop(0)
    return "\n".join(lines).rstrip()


def _source_text(path: Path) -> str:
    return _without_added_headers(path.read_text().replace(COiN_IMPORT, MCIT_IMPORT))


def test_eval_files_match_their_method_specific_v3_sources(repo_root: Path, v3_root: Path) -> None:
    checked: set[tuple[str, str]] = set()
    groups = [(method, benchmark, "train") for method in METHODS for benchmark in BENCHMARKS]
    mrlora = next(method for method in METHODS if method.id == "mrlora")
    groups.extend((mrlora, benchmark, "router") for benchmark in BENCHMARKS)
    for method, benchmark, sequence in groups:
        commands = build_sequence_commands(
            repo_root / "configs", "llava", method.id, benchmark.id,
            only_eval=True, sequence=sequence,
        )
        for command in commands:
            source_module = command.metadata["source_eval_module"]
            key = (method.id, source_module)
            if key in checked:
                continue
            checked.add(key)
            source = v3_root / "LLaVA" / method.v3_dir / Path(*source_module.split("."))
            source = source.with_suffix(".py")
            if not source.is_file():
                # KeepLoRA relies on the shared conversion implementation.
                source = v3_root / "LLaVA/LoRA-FT" / Path(*source_module.split("."))
                source = source.with_suffix(".py")
            target_module = command.metadata["eval_target"]
            target = repo_root / Path(*target_module.split("."))
            target = target.with_suffix(".py")
            assert source.is_file(), source
            assert target.is_file(), target
            assert _source_text(target) == _source_text(source), f"{method.id}: {source_module}"
    assert len(checked) == 178


def test_router_scripts_only_change_the_configurable_results_root(repo_root: Path, v3_root: Path) -> None:
    for benchmark, source_name in (("ucit", "UCIT"), ("mllm_dcl", "DCL"), ("mllm_acl", "ACL")):
        target = repo_root / f"mcitlib/methods/_mrlora/router_results/{benchmark.split('_')[-1]}.py"
        source = v3_root / f"LLaVA/MR-LoRA/scripts/{source_name}_extract_router_result.py"
        expected = source.read_text()
        v3_names = {"UCIT": "UCIT", "DCL": "DCL", "ACL": "MLLM-ACL"}
        old = f'"/your_path/MCITlib_v3/LLaVA/MR-LoRA/results/{v3_names[source_name]}/each_dataset"'
        new = f'os.path.join(os.environ.get("MCITLIB_RESULTS_ROOT", os.path.join(os.getcwd(), "results")), "{v3_names[source_name]}", "each_dataset")'
        if source_name == "DCL":
            old = old.replace('"/your_path', 'BASE_PATH = "/your_path')
            new = "BASE_PATH = " + new
        else:
            old = "    base_path = " + old
            new = "    base_path = " + new
        assert old in expected
        expected = expected.replace(old, new)
        assert _without_added_headers(target.read_text()) == _without_added_headers(expected)
