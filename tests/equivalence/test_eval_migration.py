# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""B4: migrated evaluation sources preserve each method's v3 implementation."""
from __future__ import annotations

import re
from pathlib import Path

from mcitlib.registry import BENCHMARKS, METHODS
from mcitlib.runner import build_sequence_commands

COiN_IMPORT = "from llava.eval.CoIN.coin_utils import get_model_name_from_path"
MCIT_IMPORT = "from mcitlib.backbones.llava.eval_runners.coin.coin_utils import get_model_name_from_path"
# v3 imported the per-method peft fork; v4 has one shared SMoLoRA tuner.
SMOLORA_V3_IMPORT = "from CoIN.peft.tuners.smolora import SMoLoraLinear"
SMOLORA_V4_IMPORT = "from mcitlib.tuners.mcit_separable_expert_lora import MCITSeparableExpertLoraLinear as SMoLoraLinear"


def _without_added_headers(text: str) -> str:
    lines = text.splitlines()
    while lines and (not lines[0] or lines[0].startswith((
        "# SPDX-License-Identifier:", "# Copyright (c) 2025-2026 MCITlib authors",
    ))):
        lines.pop(0)
    return "\n".join(lines).rstrip()


# v3 files that put their own directory on sys.path; removed in v4 (no hard-coded paths).
V3_PATH_HACK = re.compile(r"^sys\.path\.append\('/your_path/[^']*'\)\n", re.MULTILINE)


def _source_text(path: Path) -> str:
    text = path.read_text().replace(COiN_IMPORT, MCIT_IMPORT).replace(SMOLORA_V3_IMPORT, SMOLORA_V4_IMPORT)
    return _without_added_headers(V3_PATH_HACK.sub("", text))


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
            if command.metadata.get("merge_chunks"):
                continue
            source_module = command.metadata["source_eval_module"]
            key = (method.id, source_module)
            if key in checked:
                continue
            checked.add(key)
            source = v3_root / "LLaVA" / method.v3_dir / Path(*source_module.split("."))
            source = source.with_suffix(".py")
            if not source.is_file() and source_module.startswith("llava.eval.ModalPrompt."):
                # AR-6: the module named by v3's ModalPrompt scripts does not exist in v3; its own
                # tree keeps the same scorer one level up (llava/eval/<leaf>.py).
                leaf = source_module.rsplit(".", 1)[1]
                source = v3_root / "LLaVA" / method.v3_dir / "llava/eval" / f"{leaf}.py"
                assert source.is_file(), source
                shared_copy = v3_root / "LLaVA/LoRA-FT/llava/eval/CoIN" / f"{leaf}.py"
                assert source.read_bytes() == shared_copy.read_bytes(), f"{leaf}: ModalPrompt scorer differs from CoIN's"
            elif not source.is_file():
                # KeepLoRA relies on the shared conversion implementation.
                source = v3_root / "LLaVA/LoRA-FT" / Path(*source_module.split("."))
                source = source.with_suffix(".py")
            target_module = command.metadata["eval_target"]
            target = repo_root / Path(*target_module.split("."))
            target = target.with_suffix(".py")
            assert source.is_file(), source
            assert target.is_file(), target
            assert _source_text(target) == _source_text(source), f"{method.id}: {source_module}"
    assert len(checked) == 178 + 9, len(checked)  # + ModalPrompt UCIT/DCL evaluation modules


def test_router_scripts_only_change_the_configurable_results_root(repo_root: Path, v3_root: Path) -> None:
    for benchmark, source_name in (("ucit", "UCIT"), ("mllm_dcl", "DCL"), ("mllm_acl", "ACL")):
        target = repo_root / f"mcitlib/methods/_mrlora/router_results/{benchmark.split('_')[-1]}.py"
        source = v3_root / f"LLaVA/MR-LoRA/scripts/{source_name}_extract_router_result.py"
        expected = source.read_text()
        v3_names = {"UCIT": "UCIT", "DCL": "DCL", "ACL": "MLLM-ACL"}
        # AR-8: v3's ACL extraction reads results/MLLM-ACL while the ACL evaluation writes results/ACL.
        v4_names = {"UCIT": "UCIT", "DCL": "DCL", "ACL": "ACL"}
        old = f'"/your_path/MCITlib_v3/LLaVA/MR-LoRA/results/{v3_names[source_name]}/each_dataset"'
        new = f'os.path.join(os.environ.get("MCITLIB_RESULTS_ROOT", os.path.join(os.getcwd(), "results")), "{v4_names[source_name]}", "each_dataset")'
        if source_name == "DCL":
            old = old.replace('"/your_path', 'BASE_PATH = "/your_path')
            new = "BASE_PATH = " + new
        else:
            old = "    base_path = " + old
            new = "    base_path = " + new
        assert old in expected
        expected = expected.replace(old, new)
        assert _without_added_headers(target.read_text()) == _without_added_headers(expected)
