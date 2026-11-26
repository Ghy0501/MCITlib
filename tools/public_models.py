# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""List pinned upstream model snapshots; download only on explicit request."""
import argparse
from pathlib import Path

REVISIONS = {
    "liuhaotian/llava-v1.5-7b": "4481d270cc22fd5c4d1bb5df129622006ccd9234",
    "openai/clip-vit-large-patch14-336": "ce19dc912ca5cd21c8a653c79e251e808ccabcd1",
    "OpenGVLab/InternVL-Chat-ViT-6B-Vicuna-7B": "5a62edecca67848bea28f66156d02e11270cbbd9",
    "OpenGVLab/InternViT-6B-224px": "084bfb607eb04f373bf0349c49eb6bc3ba094919",
    "LanguageBind/Video-LLaVA-7B": "aecae02b7dee5c249e096dcb0ce546eb6f811806",
    "LanguageBind/LanguageBind_Image": "d8c2e37b439f4fc47c649dc8b90cdcd3a4e0c80e",
    "LanguageBind/LanguageBind_Video_merge": "efc40ec6ba6b2081276c11e7e19b24f08a099e79",
    "DAMO-NLP-SG/VideoLLaMA2-7B": "b39f289a67cf0aacd935d646de3bc945e63c3a0f",
    "Qwen/Qwen3-30B-A3B-Instruct-2507": "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
}
CLIP = "openai/clip-vit-large-patch14-336"
MODEL_GROUPS = {
    "llava": ("liuhaotian/llava-v1.5-7b", CLIP),
    "internvl": ("OpenGVLab/InternVL-Chat-ViT-6B-Vicuna-7B", "OpenGVLab/InternViT-6B-224px", CLIP),
    "videollava": ("LanguageBind/Video-LLaVA-7B", "LanguageBind/LanguageBind_Image",
                   "LanguageBind/LanguageBind_Video_merge", CLIP),
    "videollama2": ("DAMO-NLP-SG/VideoLLaMA2-7B", CLIP),
    "judge": ("Qwen/Qwen3-30B-A3B-Instruct-2507",),
}


def assets(groups):
    names = dict.fromkeys(name for group in groups for name in MODEL_GROUPS[group])
    return [(name, REVISIONS[name]) for name in names]


def install(root, name, revision, fetch):
    target = root / name.split("/")[-1]
    if target.exists() and not target.is_symlink():
        raise FileExistsError(f"Refusing to replace {target}. Use an empty model root.")
    # Do not download duplicate TF weights, training logs, or the redundant legacy
    # Video-LLaVA shards. Other models (including both image models) need .bin.
    exclude = ["*.h5", "*.msgpack", "runs/*"]
    if name == "LanguageBind/Video-LLaVA-7B":
        exclude += ["pytorch_model*.bin", "pytorch_model.bin.index.json"]
    snapshot = Path(fetch(repo_id=name, revision=revision, cache_dir=str(root / ".hub-cache"),
                          ignore_patterns=exclude)).resolve(strict=True)
    if target.is_symlink() and target.resolve() == snapshot:
        return
    target.symlink_to(snapshot.relative_to(root), target_is_directory=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("groups", choices=tuple(MODEL_GROUPS), nargs="+")
    parser.add_argument("--root", type=Path, required=True, help="local model directory")
    parser.add_argument("--download", action="store_true", help="fetch and link these upstream snapshots")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    rows = assets(args.groups)
    for name, revision in rows:
        print(f"https://huggingface.co/{name}/tree/{revision} -> {root / name.split('/')[-1]}", flush=True)
    if args.download:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            parser.error("Install huggingface_hub in a separate download environment (docs/public-start.md).")
        root.mkdir(parents=True, exist_ok=True)
        for name, revision in rows:
            install(root, name, revision, snapshot_download)


if __name__ == "__main__":
    main()
