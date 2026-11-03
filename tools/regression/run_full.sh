#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
# T5 covers LoRA-FT/UCIT, HiDe/DCL and SEFE/ACL (llava-t437).
if [[ " $* " != *" --combo "* ]]; then
  set -- --combo lora_ft:ucit --combo hide:mllm_dcl --combo sefe:mllm_acl "$@"
fi
"${PYTHON:-python3}" tools/regression/compare.py run --full "$@"
