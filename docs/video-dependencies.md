# Video dependencies for CL-VISTA

## Prepare the source

The release contains recipes, not installable copies of the video packages.
Use Python 3.10+ and Git. Run from the MCITlib release root, before starting
workers. Set `RUNTIME_ROOT` to the absolute, node-local runtime directory
from the node binding, outside this checkout.

```sh
python3 -B tools/prepare_video_sources.py --fetch --output "$RUNTIME_ROOT/sources"
python3 -B tools/prepare_video_sources.py --check --output "$RUNTIME_ROOT/sources"
```

For an offline preparation, replace `--fetch` with `--from-git /path/to/MCITlib`
(the local Git object database must already contain the pinned v3 commit **and**
its selected blobs). No upstream working tree is read or restored. A fresh build
verifies every input hash, applies the patches, verifies the complete output,
then publishes the directory. A repeat verifies an existing directory without
fetching or overwriting it. An incomplete or dirty output is rejected; use a
fresh destination rather than combining versions. Hidden files, extra packages,
symbolic links and bytecode caches are not allowed in the output.

For release validation, `python3 -B tools/check_vendor_video.py --from-git .`
tests a fresh offline build, repeat runs, bounded download retries, cleanup
after failure, syntax and detection of missing, modified or unexpected files.
Use the path to a complete v3 object database instead of `.` if necessary.

The two import roots are `$RUNTIME_ROOT/sources/Video-LLaVA` and
`$RUNTIME_ROOT/sources/VideoLLaMA2`. To relocate them together, set
`native_sources` in the node binding or export `MCITLIB_VIDEO_SOURCES`; the
environment variable takes precedence. Both name the **parent** of the two
directories, not a package or a checkout. The worker must see only its own
backbone on `PYTHONPATH`; PEFT comes from that worker's environment, **not**
MCITlib's image PEFT tree. Do not install the outer projects or let an import
trigger source, weight or package acquisition.

## What is pinned

The direct parent is [MCITlib v3 at a3a622c](https://github.com/Ghy0501/MCITlib/tree/a3a622c9beb690e84c2923785279f179a6aac07e).
The LoRA-FT implementations and the MR-LoRA router helpers are not
interchangeable with current upstream branches. Each recipe records the
v3 paths and SHA-256 of the original and patched bytes. Its `UPSTREAM.md`
identifies which v3 inputs differ from the corresponding original upstream.

| Prepared runtime | Contents | Original upstream license evidence |
| --- | --- | --- |
| [Video-LLaVA recipe](../third_party/Video-LLaVA/UPSTREAM.md) | 50 Python files and LICENSE. Includes both QA helpers and eagerly imported MPT and LanguageBind modules. Omits checkpoint conversion and alternate training launchers. | [PKU-YuanGroup at b6534e7](https://github.com/PKU-YuanGroup/Video-LLaVA/blob/b6534e7e1b1cdf674811466c5581582a993213dc/LICENSE), Apache-2.0 |
| [VideoLLaMA2 recipe](../third_party/VideoLLaMA2/UPSTREAM.md) | 15 Python files and LICENSE. Includes the Mistral training entry, model registrations, projector, processor and router prompt helper. Ordinary inference uses the package API. | [DAMO-NLP-SG at 1c89aeb](https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/1c89aeb9bd55c594c03e960e0934dd90cb0fac7f/LICENSE), Apache-2.0 |

Both original trees have an Apache-2.0 root LICENSE and no separate NOTICE in
the audited Git trees. The selected v3 method directories also carry Apache-2.0
LICENSE files. Neither fact independently clears every embedded dependency:
Video-LLaVA describes **most** of its project as Apache-2.0, and each project's
service and model terms are distinct from the source license. See the pinned
[Video-LLaVA terms](https://github.com/Ghy0501/MCITlib/blob/a3a622c9beb690e84c2923785279f179a6aac07e/Video-LLaVA/LoRA-FT/README.md#-license)
and [VideoLLaMA2 terms](https://github.com/Ghy0501/MCITlib/blob/a3a622c9beb690e84c2923785279f179a6aac07e/VideoLLaMA2/LoRA-FT/README.md#-license).
Only metadata and narrow local patches ship here; do not redistribute a
prepared tree as part of the release. Retain original headers. Weights,
model configs, media, task data, caches and method-specific binary assets
are separate inputs, not covered by the source recipes.

## Configure the workers

Use separate Linux environments with compatible CUDA wheels; the training stacks are not interchangeable.

| Worker | Python | Torch / CUDA | Transformers | PEFT | DeepSpeed | Flash Attention | Accelerate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Video-LLaVA | 3.10 | 2.0.1+cu117 / 11.7 | 4.31.0 | 0.4.0 | 0.13.1 | 2.6.3 | 0.21.0 |
| VideoLLaMA2 | 3.10 | 2.2.0+cu118 / 11.8 | 4.40.0 | 0.4.0 | 0.13.1 | 2.5.8 | 0.26.1 |

- Both need Decord (0.6.0), Pillow, NumPy, torchvision matched to
  Torch, OpenCV, einops, SentencePiece, tokenizers, tqdm and Hugging Face Hub.
  Provision TensorBoard or tensorboardX for training logs and bitsandbytes
  for quantized paths. Prebuild or locally supply the required native
  extensions, compiler and Ninja, especially DeepSpeed CPUAdam for offload.
- Video-LLaVA also imports PyTorchVideo (0.1.5). Configure local
  Video-LLaVA-7B tokenizer/config/weights and **both** LanguageBind_Image and
  LanguageBind_Video_merge towers in the model view. The video doctor expects
  eight frames in channel-first order.
- VideoLLaMA2 uses timm (1.0.3) and imageio. Bind a local Mistral-based
  VideoLLaMA2-7B model view and the CLIP ViT-L/14-336 vision tower. The video
  doctor expects eight frames in time-first order.
- Set the selected model's `conda_prefix`, `model_view` and `towers`; set
  `media` and task annotation paths separately. Bind `auxiliary_clip` and
  verified method assets where required. No assets or data are acquired by
  the preparation tool.
- Keep `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `PYTHONNOUSERSITE=1`
  and `PYTHONDONTWRITEBYTECODE=1` on worker processes, including subprocesses.
  Native loaders have Hub fallbacks when adapter files are missing. Supply
  complete local checkpoints and towers, run the video doctor with egress
  disabled, and fail on missing inputs rather than populating caches at runtime.

## Native judge

The public scorer is MCITlib's own wrapper. Do **not** acquire or execute a
third judge package. The historical prompt and scoring reference was the
[Video-LLaVA v3 Qwen script](https://github.com/Ghy0501/MCITlib/blob/a3a622c9beb690e84c2923785279f179a6aac07e/Video-LLaVA/LoRA-FT/videollava/eval/video/eval_video_qa_qwen.py)
and its [identical VideoLLaMA2 counterpart](https://github.com/Ghy0501/MCITlib/blob/a3a622c9beb690e84c2923785279f179a6aac07e/VideoLLaMA2/LoRA-FT/videollama2/eval/eval_video_qa_qwen.py):
Git blob `6766642ae9393cfe0101a2b80d659434ba7b2e8f` in both cases, under
the respective v3 LoRA-FT [LICENSE](https://github.com/Ghy0501/MCITlib/blob/a3a622c9beb690e84c2923785279f179a6aac07e/Video-LLaVA/LoRA-FT/LICENSE).
The original standalone parser is not a release dependency.

Use the separate judge interpreter and a trusted, local
`Qwen3-30B-A3B-Instruct-2507` checkpoint with its tokenizer and chat template.
Install the [judge environment](../envs/video-judge.txt). With the offline flags above and the release importable, launch:

```sh
python -m mcitlib.backbones.video.judge_service --runtime-root "$RUNTIME_ROOT" \
  --model-path "$JUDGE_MODEL" --attn-implementation flash_attention_2
```

When predictions are outside the default `$RUNTIME_ROOT/results`, pass the
bound `--results-root`. The scoring client connects to the local Unix socket
`$RUNTIME_ROOT/control/judge.sock`. A successful generation is not a score.
