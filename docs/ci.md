# CI and test environments

For a lightweight local check, use Linux, Python 3.10, PyYAML, pytest, and the declared development tools. This checks configurations, representative command plans, registry/CLI behavior, queue isolation, reports, and metrics without loading models or requiring Torch, CUDA, private datasets, or weights. The queue tests need Linux process identity and pidfd support.

```bash
python -m pip install -e '.[dev]'
python -m pytest -q tests/core tests/unit/test_config_templates.py \
  tests/unit/test_registry_and_cli.py tests/unit/test_runner_execution.py \
  tests/unit/test_metrics.py tests/unit/test_four_gpu_validation.py
```

The [hosted workflow](../.github/workflows/ci.yml) runs two image integration jobs on fresh Ubuntu runners: `envs/llava.txt` (Transformers 4.31) and `envs/llava-t437.txt` (4.37). Both use CPU PyTorch and a pinned v3 worktree for reference comparisons. Method-specific model tests run under the method's declared Transformers version; the test suite does not certify alternate versions. The FlashAttention stand-in is for CPU checks only, not CUDA kernel validation. There is no separate hosted GPU, video, or ModalPrompt/DeepSpeed 0.12.6 job.

Lint, SPDX, and third-party checks run separately. `tools/check_third_party.py` verifies vendored files against their manifests without network access.

See [first image score](first-score-images.md) and [first video score](first-score-video.md) for actual GPU training and checkpoint reload checks. Before a full benchmark, run the complete doctor for that method and hardware.
