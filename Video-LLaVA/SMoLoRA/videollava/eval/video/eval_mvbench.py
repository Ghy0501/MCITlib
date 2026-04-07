# evaluate_mvbench_videollava.py
import os
import json
import math
import argparse
import re
from collections import defaultdict

import torch
from tqdm import tqdm
from PIL import Image

from decord import VideoReader, cpu
from decord._ffi.base import DECORDError
import sys
sys.path.append('./')
from CoIN.peft.tuners.smolora import SMoLoraLinear
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.constants import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
    DEFAULT_VID_START_TOKEN,
    DEFAULT_VID_END_TOKEN,
)
from videollava.mm_utils import (
    get_model_name_from_path,
    tokenizer_image_token,
    KeywordsStoppingCriteria,
)
from videollava.model.builder import load_pretrained_model


# ----------------------------
# Utils
# ----------------------------
def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    return split_list(lst, n)[k]


def qa_template(data):
    question = f"Question: {data['question']}\n"
    question += "Options:\n"
    answer = data["answer"]
    answer_idx = -1
    for idx, c in enumerate(data["candidates"]):
        question += f"({chr(ord('A') + idx)}) {c}\n"
        if c == answer:
            answer_idx = idx
    question = question.rstrip()
    answer = f"({chr(ord('A') + answer_idx)}) {answer}"
    return question, answer


def check_ans(pred: str, gt: str) -> bool:
    """
    Robust answer checker:
    - Correctly handles pred like: "B", "(B)", "B.", "Answer: B", "The answer is (B) ..."
    - Falls back to content substring match if option letter isn't reliably present
    """
    if pred is None:
        return False
    pred = pred.strip()
    gt = gt.strip()
    if not pred:
        return False

    pred_l = pred.lower().replace("\n", " ").strip()
    gt_l = gt.lower().replace("\n", " ").strip()

    # ---------
    # 1) Parse option letter robustly
    # ---------
    def extract_option(s: str) -> str:
        s = s.strip().lower()

        # case: single letter like "b"
        if len(s) == 1 and s.isalpha():
            return s

        # case: starts with "(b)" or "b." or "b)" or "b:" or "b "
        m = re.match(r"^\(?\s*([a-z])\s*\)?\s*[\.\):]?\s*(.*)$", s)
        if m:
            return m.group(1)

        # case: anywhere in text like "answer is (b)" or "i choose c"
        m2 = re.search(r"(?:answer|option|choose|choice)\s*(?:is|:)?\s*\(?\s*([a-z])\s*\)?", s)
        if m2:
            return m2.group(1)

        # standalone "(b)" somewhere
        m3 = re.search(r"\(\s*([a-z])\s*\)", s)
        if m3:
            return m3.group(1)

        return ""

    def extract_gt_option_and_content(s: str):
        s = s.strip().lower()
        opt = extract_option(s)
        # remove leading option markup to get content
        content = re.sub(r"^\(?\s*[a-z]\s*\)?\s*[\.\):]?\s*", "", s).strip()
        if content.endswith("."):
            content = content[:-1].strip()
        return opt, content

    pred_opt = extract_option(pred_l)
    gt_opt, gt_content = extract_gt_option_and_content(gt_l)

    # option letter match is enough
    if pred_opt and gt_opt and pred_opt == gt_opt:
        return True

    # ---------
    # 2) Content fallback
    # ---------
    if gt_content and gt_content in pred_l:
        return True

    pred_content = re.sub(r"^\(?\s*[a-z]\s*\)?\s*[\.\):]?\s*", "", pred_l).strip()
    if pred_content and gt_content and pred_content == gt_content:
        return True

    return False


# ----------------------------
# Frame indices (MVBench style)
# ----------------------------
def get_frame_indices(num_segments, fps, max_frame, bound=None, first_idx=0):
    if bound is not None:
        start, end = bound
    else:
        start, end = -1e9, 1e9

    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)

    if end_idx <= start_idx:
        start_idx = first_idx
        end_idx = max_frame

    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = [
        int(start_idx + (seg_size / 2) + round(seg_size * i))
        for i in range(num_segments)
    ]
    frame_indices = [min(max(first_idx, idx), max_frame) for idx in frame_indices]
    return frame_indices


def load_frame_dir_frames(frame_dir, num_segments=8, fps=3, bound=None):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    all_files = [f for f in os.listdir(frame_dir) if f.lower().endswith(exts)]
    all_files.sort()
    if len(all_files) == 0:
        raise FileNotFoundError(f"No frames found in: {frame_dir}")

    max_frame = len(all_files) - 1
    idxs = get_frame_indices(num_segments, fps, max_frame, bound=bound, first_idx=0)
    frames = []
    for i in idxs:
        img_path = os.path.join(frame_dir, all_files[i])
        frames.append(Image.open(img_path).convert("RGB"))
    return frames


# ----------------------------
# Robust video tensor loader
# ----------------------------
def video_tensor_from_video_path_with_fallback(video_path, video_processor, num_segments=8, bound=None):
    """
    1) Try official: video_processor.preprocess(video_path)  (fast)
    2) If decord/ffmpeg crashes: fallback to manual decode with VideoReader(num_threads=1) + per-frame transform
    Return: torch.Tensor [T,C,H,W] or None
    """
    # fast path
    try:
        return video_processor.preprocess(video_path, return_tensors="pt")["pixel_values"][0]
    except (DECORDError, RuntimeError, OSError, ValueError, TypeError):
        # fallback path: manual safe decode
        try:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            max_frame = len(vr) - 1
            fps = float(vr.get_avg_fps()) if vr.get_avg_fps() is not None else 30.0
            idxs = get_frame_indices(num_segments, fps, max_frame, bound=bound, first_idx=0)

            # get_batch is faster but may still fail on bad videos -> we wrap again
            try:
                batch = vr.get_batch(idxs).asnumpy()  # [T,H,W,3]
                frames = [Image.fromarray(batch[i]).convert("RGB") for i in range(batch.shape[0])]
            except Exception:
                frames = []
                for i in idxs:
                    frames.append(Image.fromarray(vr[i].asnumpy()).convert("RGB"))

            feats = [
                video_processor.image_processor(img, video_processor.transform, return_tensors="pt")["pixel_values"][0]
                for img in frames
            ]
            return torch.stack(feats, dim=0)
        except Exception:
            return None


# ----------------------------
# VideoLLaVA Inference
# ----------------------------
def get_model_output(
    model,
    video_processor,
    tokenizer,
    media,
    qs,
    device,
    media_type="video",
    num_segments=8,
    tvqa_fps=3,
    bound=None,
):
    # prepend video tokens
    if getattr(model.config, "mm_use_im_start_end", False):
        qs = DEFAULT_VID_START_TOKEN + "".join([DEFAULT_IMAGE_TOKEN] * 8) + DEFAULT_VID_END_TOKEN + "\n" + qs
    else:
        qs = "".join([DEFAULT_IMAGE_TOKEN] * 8) + "\n" + qs

    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # preprocess -> video_tensor
    if media_type == "video":
        video_tensor = video_tensor_from_video_path_with_fallback(
            media, video_processor, num_segments=num_segments, bound=bound
        )
        if video_tensor is None:
            return None  # signal decode failure

    elif media_type == "frame":
        frames = load_frame_dir_frames(media, num_segments=num_segments, fps=tvqa_fps, bound=bound)
        feats = [
            video_processor.image_processor(img, video_processor.transform, return_tensors="pt")["pixel_values"][0]
            for img in frames
        ]
        video_tensor = torch.stack(feats, dim=0)
    else:
        raise ValueError(f"Unknown media_type: {media_type}")

    video_tensor = video_tensor.half().to(device)

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[video_tensor],
            do_sample=False,
            temperature=0.1,
            max_new_tokens=256,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    input_token_len = input_ids.shape[1]
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    return outputs.strip()


# ----------------------------
# MVBench Loader
# ----------------------------
def build_mvbench_samples(data_dir, data_list):
    samples = []
    for task_type, (json_name, prefix, data_type, has_bound) in data_list.items():
        json_path = os.path.join(data_dir, json_name)
        with open(json_path, "r") as f:
            json_data = json.load(f)
        for data in json_data:
            samples.append({
                "task_type": task_type,
                "prefix": prefix,
                "data_type": data_type,
                "has_bound": has_bound,
                "data": data,
            })
    return samples


# ----------------------------
# Main
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--video_dir", type=str, required=True, help="Root directory of MVBench videos")
    parser.add_argument('--ins-type', type=int, default=0)
    parser.add_argument("--mvbench_json_dir", type=str, required=True)
    parser.add_argument("--num_segments", type=int, default=8)
    parser.add_argument("--tvqa_fps", type=int, default=3)

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_name", type=str, default="videollava_mvbench")

    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    video_dir = args.video_dir
    # ---- MVBench task config ----
    data_list = {
    "Action Sequence": ("action_sequence.json", os.path.join(video_dir, "videos"), "video", True),
    "Action Prediction": ("action_prediction.json", os.path.join(video_dir, "videos"), "video", True),
    "Action Antonym": ("action_antonym.json", os.path.join(video_dir, "videos"), "video", False),
    "Fine-grained Action": ("fine_grained_action.json", os.path.join(video_dir, "videos"), "video", False),
    "Unexpected Action": ("unexpected_action.json", os.path.join(video_dir, "videos"), "video", False),
    "Object Existence": ("object_existence.json", os.path.join(video_dir, "videos"), "video", False),
    "Object Interaction": ("object_interaction.json", os.path.join(video_dir, "videos"), "video", True),
    "Object Shuffle": ("object_shuffle.json", os.path.join(video_dir, "videos"), "video", False),
    "Moving Direction": ("moving_direction.json", os.path.join(video_dir, "videos"), "video", False),
    "Action Localization": ("action_localization.json", os.path.join(video_dir, "videos"), "video", True),
    "Scene Transition": ("scene_transition.json", os.path.join(video_dir, "videos"), "video", False),
    "Action Count": ("action_count.json", os.path.join(video_dir, "videos"), "video", False),
    "Moving Count": ("moving_count.json", os.path.join(video_dir, "videos"), "video", False),
    "Moving Attribute": ("moving_attribute.json", os.path.join(video_dir, "videos"), "video", False),
    "State Change": ("state_change.json", os.path.join(video_dir, "videos"), "video", False),
    "Character Order": ("character_order.json", os.path.join(video_dir, "videos"), "video", False),
    "Egocentric Navigation": ("egocentric_navigation.json", os.path.join(video_dir, "videos"), "video", False),
    "Episodic Reasoning": ("episodic_reasoning.json", os.path.join(video_dir, "videos"), "video", True),
    "Counterfactual Inference": ("counterfactual_inference.json", os.path.join(video_dir, "videos"), "video", False),
}

    # ---- load model ----
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )
    model = model.to(args.device).eval()
    for name, module in model.named_modules():
        if isinstance(module, SMoLoraLinear):
            module.ins_type = args.ins_type

    print("ins_type", args.ins_type)
    # ---- load samples ----
    samples = build_mvbench_samples(args.mvbench_json_dir, data_list)
    samples = get_chunk(samples, args.num_chunks, args.chunk_idx)

    # ---- outputs ----
    out_path = os.path.join(args.output_dir, f"{args.output_name}.jsonl")
    err_path = os.path.join(args.output_dir, f"{args.output_name}_errors.jsonl")
    acc = defaultdict(lambda: [0, 0])
    total_correct, total_cnt = 0, 0
    skipped = 0

    with open(out_path, "w", encoding="utf-8") as f_out, open(err_path, "w", encoding="utf-8") as f_err:
        for s in tqdm(samples, desc="Evaluating MVBench"):
            task_type = s["task_type"]
            data = s["data"]

            bound = None
            if s["has_bound"]:
                bound = (data["start"], data["end"])

            question, gt_answer = qa_template(data)
            media_path = os.path.join(s["prefix"], data["video"])

            # inference (robust)
            try:
                pred = get_model_output(
                    model=model,
                    video_processor=processor["video"],
                    tokenizer=tokenizer,
                    media=media_path,
                    qs=question,
                    device=args.device,
                    media_type=s["data_type"],
                    num_segments=args.num_segments,
                    tvqa_fps=args.tvqa_fps,
                    bound=bound,
                )
            except Exception as e:
                pred = None
                f_err.write(json.dumps({
                    "task_type": task_type,
                    "video": data["video"],
                    "media_path": media_path,
                    "error": repr(e),
                }, ensure_ascii=False) + "\n")

            # decode failure (video_tensor None) -> skip
            if pred is None:
                skipped += 1
                f_err.write(json.dumps({
                    "task_type": task_type,
                    "video": data["video"],
                    "media_path": media_path,
                    "error": "decode_failed_or_pred_none",
                }, ensure_ascii=False) + "\n")
                continue

            correct = check_ans(pred, gt_answer)

            acc[task_type][1] += 1
            if correct:
                acc[task_type][0] += 1

            total_cnt += 1
            total_correct += int(correct)

            record = {
                "task_type": task_type,
                "video": data["video"],
                "question": question,
                "answer": gt_answer,
                "pred": pred,
                "correct": bool(correct),
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ---- summary ----
    summary = {}
    for task, (c, t) in acc.items():
        summary[task] = (c / t * 100.0) if t > 0 else 0.0
    summary["Avg"] = (total_correct / total_cnt * 100.0) if total_cnt > 0 else 0.0
    summary["Evaluated"] = total_cnt
    summary["Skipped"] = skipped

    summary_path = os.path.join(args.output_dir, f"{args.output_name}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== MVBench Results =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved predictions to: {out_path}")
    print(f"Saved errors to:      {err_path}")
    print(f"Saved summary to:     {summary_path}")


if __name__ == "__main__":
    main()
