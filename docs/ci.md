# CI and test environments

The lightweight CPU job uses Python 3.10, PyYAML, pytest, and the declared development tools. It validates configurations, representative command plans, registry/CLI behavior, queue isolation, reports, and metrics without loading models or requiring Torch, CUDA, private datasets, or weights.

```bash
python -m pip install -e '.[dev]'
python -m pytest -q tests/core tests/unit/test_config_templates.py \
  tests/unit/test_registry_and_cli.py tests/unit/test_runner_execution.py \
  tests/unit/test_metrics.py tests/unit/test_four_gpu_validation.py
```

Model integration jobs install the matching requirements from `envs/llava.txt`, `envs/llava-t437.txt`, or `envs/llava-modalprompt-ds126.txt`. They check model construction, training helpers, state handling, and entry points on CPU. Tests requiring additional inputs are explicitly gated. The flash-attention import stand-in exercises imports only; it does not validate CUDA kernels, distributed collectives, or GPU training.

Lint, SPDX, and third-party checks run separately. `tools/check_third_party.py` verifies vendored files against their manifests without network access.

Use a fresh environment to check the lightweight dependency boundary. Before GPU work, run the complete doctor for that method and hardware.
