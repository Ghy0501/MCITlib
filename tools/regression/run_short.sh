#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
# Default is a command preview. On a GPU node, append --execute.
"${PYTHON:-python3}" tools/regression/compare.py run --tasks "${TASKS:-1,2}" --max-steps "${MAX_STEPS:-20}" "$@"
