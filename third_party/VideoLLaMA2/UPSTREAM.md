# VideoLLaMA2 source recipe

- Source: https://github.com/Ghy0501/MCITlib
- Commit: `a3a622c9beb690e84c2923785279f179a6aac07e` (v3 runtime parent).
- Distribution: acquisition metadata and local changes only. No Python package,
  pretrained model, media, environment or upstream checkout is shipped here.
- License evidence: the selected v3 component directories contain Apache-2.0
  LICENSE (Git blob `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64`).
  The prepared tree retains this file and the original per-file headers.
- `SOURCES.json` pins each original path and SHA-256; `prepared_sha256` is
  recorded when the output differs. `MANIFEST.json` covers the release recipe,
  not the external output. `runtime.diff` is applied only during preparation.

[Original backbone](https://github.com/DAMO-NLP-SG/VideoLLaMA2/tree/1c89aeb9bd55c594c03e960e0934dd90cb0fac7f) and
[license](https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/1c89aeb9bd55c594c03e960e0934dd90cb0fac7f/LICENSE).
Compared v3 input blobs against the original commit by path (before local patches):
10 of 16 match exactly, including LICENSE.
These paths instead use the pinned v3 implementation:

- `videollama2/__init__.py`
- `videollama2/eval/inference_video_qa_router.py`
- `videollama2/model/__init__.py`
- `videollama2/model/encoder.py`
- `videollama2/train.py`
- `videollama2/videollama2_trainer.py`

The MR-LoRA router helper comes from its own v3 method directory. Other inputs
come from LoRA-FT.
The v4 patch adds trainer callbacks and an option to load without merging the
adapter. The release also drops legacy source-search paths; no v3 checkout
is needed to run the result.

See [video dependencies](../../docs/video-dependencies.md) for offline use and scope.
