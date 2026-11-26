# First run from a public checkout

Use a **source checkout**. The training configurations and vendored runtimes are part of the checkout; a wheel alone is not a complete training distribution. You do not need access to the authors' servers, filesystem or accounts.

## 1. Verify the checkout (CPU only)

```bash
git clone --depth 1 --branch v4-dev https://github.com/Ghy0501/MCITlib.git
cd MCITlib
git rev-parse HEAD  # keep this with the result
python3.10 -m venv .venv-control
source .venv-control/bin/activate
python -m pip install -e .
mcitlib list
mcitlib config validate --json
```

This stage needs Python with `venv` and pip (on Debian/Ubuntu the OS package is `python3.10-venv`), but no GPU, model download or private data. It verifies the control commands, **not** that training will run. For training use Linux x86_64 with NVIDIA GPUs, matching driver/toolkit, and a separate environment for the chosen worker. Do not install all four GPU stacks together.

## 2. Get just the models you need

Run from the checkout on a networked machine. Choose an external directory with room for the weights; use the same absolute root throughout each guide. The default below prints immutable source URLs and target paths without writing anything:

```bash
export RUNTIME_ROOT="$HOME/mcit-runtime"
export MODEL_ROOT="$RUNTIME_ROOT/models"
python tools/public_models.py llava --root "$MODEL_ROOT"
```

Replace `llava` with one or more groups. Review each upstream model card and its terms before downloading:

| Group | Primary checkpoint | Additional downloads |
| --- | --- | --- |
| `llava` | [LLaVA-1.5-7B](https://huggingface.co/liuhaotian/llava-v1.5-7b) | [CLIP ViT-L/14-336](https://huggingface.co/openai/clip-vit-large-patch14-336) |
| `internvl` | [InternVL-Chat-ViT-6B-Vicuna-7B](https://huggingface.co/OpenGVLab/InternVL-Chat-ViT-6B-Vicuna-7B), **LLaVA-format**, not InternVL2 | [InternViT-6B-224px](https://huggingface.co/OpenGVLab/InternViT-6B-224px); CLIP for guidance methods |
| `videollava` | [Video-LLaVA-7B](https://huggingface.co/LanguageBind/Video-LLaVA-7B) | [LanguageBind_Image](https://huggingface.co/LanguageBind/LanguageBind_Image), [LanguageBind_Video_merge](https://huggingface.co/LanguageBind/LanguageBind_Video_merge), CLIP for auxiliary methods |
| `videollama2` | [VideoLLaMA2-7B](https://huggingface.co/DAMO-NLP-SG/VideoLLaMA2-7B), Mistral/STC | CLIP |
| `judge` | [Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) | Only needed for the official video scoring path; not downloaded with either video worker |

Allow approximately **14.2 GiB** for the LLaVA group, **36.3 GiB** for InternVL, **19.1 GiB** for Video-LLaVA, **16.6 GiB** for VideoLLaMA2, or **56.9 GiB** for the judge alone. These are upstream file sizes at the pinned revisions with the helper's exclusions, not GPU memory requirements. Leave additional room for downloads in progress, environments, checkpoints and data.

To actually download, use the control environment or a separate download environment, **not** the pinned GPU environment:

```bash
python -m pip install huggingface_hub==0.36.0
python tools/public_models.py llava --root "$MODEL_ROOT" --download
# Example for official video scoring:
# python tools/public_models.py videollava judge --root "$MODEL_ROOT" --download
```

The helper uses immutable Hub revisions, deduplicates shared towers, and links the model names expected by MCITlib into the model root. It does not overwrite existing model directories or fetch benchmark data. Keep the hidden `.hub-cache` with its links when transferring to an offline host, and do not edit the downloaded configs in place. Keep the pinned revisions with your own run metadata so others can reproduce the inputs.

## 3. Train, reload and get the first score

For **image LoRA-FT** on a CUDA 11.8 development host, the reference family is Python 3.10, PyTorch 2.0.1+cu118 and Transformers 4.31. Start a fresh shell with Conda initialized, ensure `nvcc --version` reports CUDA 11.8 and select a compatible host C++ compiler. The Python package install does not supply `nvcc`. From the checkout:

```bash
conda create -y -n mcit-image-lora python=3.10 pip
conda activate mcit-image-lora
python -m pip install 'setuptools>=68,<81' wheel packaging ninja psutil 'numpy==1.23.5'
python -m pip install --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.0.1+cu118 torchvision==0.15.2+cu118
python -m pip install --no-build-isolation -r envs/llava.txt
MAX_JOBS=2 python -m pip install --no-build-isolation flash-attn==2.6.3
python -m pip install -e .
python -m pip check
```

A prebuilt FlashAttention wheel must match Python, CUDA, Torch and the C++ ABI. The other image methods may need a different environment; see the [method table](../README.md#installation). For videos, use the [separate worker environments](video-runtime.md#install-the-runtimes), not the image recipe above.

| I want to... | Next guide |
| --- | --- |
| Try LLaVA-1.5 or InternVL with a tiny image example | [First image score](first-score-images.md) |
| Try Video-LLaVA or VideoLLaMA2 with a tiny video example | [First video score](first-score-video.md) |
| Prepare real UCIT, MLLM-DCL / ACL or CL-VISTA benchmark data | [Benchmark sources and layout](benchmark-inputs.md) |
| Train and evaluate the eight-task video sequence with its dedicated judge | [Video runtime](video-runtime.md) |

For the first run, keep the checkout revision, environment versions, predictions, denominator and score together. A score on a tiny generated example proves the path ran; it is **not** a benchmark result. Full benchmarks have separate annotation and media requirements (including access-controlled media in CL-VISTA). We do not redistribute pretrained model weights, datasets or credentials with the source.

The long-running queues additionally need Linux process tracking: a readable `/proc` and permitted `pidfd_open` / `pidfd_send_signal` for their own children. A restrictive container can run the CLI yet fail these queue checks with `Operation not permitted`. Use a supported job environment; do not bypass the ownership checks or grant the container host-wide privileges.
