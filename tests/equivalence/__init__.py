# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""CPU/GPU equivalence tests."""

# These are the migrated methods with recorded v3 command and source fixtures.
# New algorithms have their own tests and do not have a v3 implementation.
V3_METHOD_IDS = (
    "lora_ft", "replay", "moelora", "mrlora", "olora", "clmoe", "disco", "hide",
    "smolora", "modalprompt", "sefe", "keeplora",
)
