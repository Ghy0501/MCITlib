# evaluate_mvbench_videollava_router.py
import os
import json
import math
import argparse

import torch
from tqdm import tqdm
from PIL import Image

from decord import VideoReader, cpu
from decord._ffi.base import DECORDError
import sys
sys.path.append("./")

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
# Router Prompt (same as your first code)
# ----------------------------
ROUTER_PROMPT_PREFIX = """You are a helpful assistant router for Video Question Answering (VideoQA) tasks. There are eight expert models, each specializing in a specific domain of video understanding.

Your task is to select the most suitable model based on the provided video content, user question, and model descriptions. Consider the temporal and visual expertise of each model carefully.

Important Instructions:
• Respond only with the letter (A, B, C, D, E, F, G, H) corresponding to the most suitable model.
• Do not attempt to answer the user's question directly.

Model Pool:
• A: A movie expert. This model is adept at interpreting cinematic sequences, understanding narrative plots, character interactions, and subtitles within movies or TV shows.
• B: A reasoning expert. This model is designed for complex video logic, analyzing cause-and-effect relationships (e.g., "why did the glass break?"), inferring intent, and temporal deduction.
• C: A GUI (Graphical User Interface) expert. This model specializes in understanding screen recordings, tracking cursor movements, identifying software interactions, and analyzing web or app navigation flows.
• D: A traffic analysis expert. This model focuses on dashcam or surveillance footage, analyzing vehicle behavior, traffic flow, road accidents, and driving scenarios.
• E: A science expert. This model has proficiency in interpreting videos of scientific experiments, physical phenomena, chemical reactions, and dynamic educational visualizations.
• F: A video counting expert. This model excels at quantifying objects or actions within a video clip. It answers "how many times" an action occurred or counts objects moving across frames.
• G: A spatial understanding expert. This model specializes in analyzing the relative positions and movements of objects over time, tracking trajectories, and understanding 3D geometry in dynamic scenes.
• H: A sports expert. This model excels at recognizing specific sports, analyzing athletes' movements, understanding game rules, and interpreting scoreboards in match highlights.

Here is the user's question: """

ROUTER_PROMPT_SUFFIX = """

You only need to select the suitable model and do not answer the question. JUST answer with the model's letter from the given choices directly."""


# ----------------------------
# Utils
# ----------------------------
def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    return split_list(lst, n)[k]


def qa_template(data):
    """
    Keep MVBench multiple-choice question formatting.
    Router will see the whole question+options text (fine).
    """
    question = f"Question: {data['question']}\n"
    question += "Options:\n"
    answer = data["answer"]
    answer_idx = -1
    for idx, c in enumerate(data["candidates"]):
        question += f"({chr(ord('A') + idx)}) {c}\n"
        if c == answer:
            answer_idx = idx
    question = question.rstrip()

    # (Optional) keep GT in output record for debugging
    gt_answer = f"({chr(ord('A') + answer_idx)}) {answer}"
    return question, gt_answer


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
    try:
        return video_processor.preprocess(video_path, return_tensors="pt")["pixel_values"][0]
    except (DECORDError, RuntimeError, OSError, ValueError, TypeError):
        try:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            max_frame = len(vr) - 1
            fps = float(vr.get_avg_fps()) if vr.get_avg_fps() is not None else 30.0
            idxs = get_frame_indices(num_segments, fps, max_frame, bound=bound, first_idx=0)

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
# VideoLLaVA Router Inference
# ----------------------------
def get_router_output(
    model,
    video_processor,
    tokenizer,
    media,
    user_qs,
    device,
    media_type="video",
    num_segments=8,
    tvqa_fps=3,
    bound=None,
):
    # build router prompt
    routing_qs = ROUTER_PROMPT_PREFIX + user_qs + ROUTER_PROMPT_SUFFIX

    # prepend video tokens + router prompt
    if getattr(model.config, "mm_use_im_start_end", False):
        qs = DEFAULT_VID_START_TOKEN + "".join([DEFAULT_IMAGE_TOKEN] * 8) + DEFAULT_VID_END_TOKEN + "\n" + routing_qs
    else:
        qs = "".join([DEFAULT_IMAGE_TOKEN] * 8) + "\n" + routing_qs

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
            max_new_tokens=16,  # router only needs 1 letter
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

    parser.add_argument("--mvbench_json_dir", type=str, required=True)
    parser.add_argument("--num_segments", type=int, default=8)
    parser.add_argument("--tvqa_fps", type=int, default=3)

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_name", type=str, default="videollava_mvbench_router")

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
    tokenizer, model, processor, _context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )
    model = model.to(args.device).eval()

    # ---- load samples ----
    samples = build_mvbench_samples(args.mvbench_json_dir, data_list)
    samples = get_chunk(samples, args.num_chunks, args.chunk_idx)

    # ---- outputs ----
    out_path = os.path.join(args.output_dir, f"{args.output_name}.jsonl")
    err_path = os.path.join(args.output_dir, f"{args.output_name}_errors.jsonl")

    skipped = 0
    with open(out_path, "w", encoding="utf-8") as f_out, open(err_path, "w", encoding="utf-8") as f_err:
        for s in tqdm(samples, desc="Routing MVBench"):
            task_type = s["task_type"]
            data = s["data"]

            bound = None
            if s["has_bound"]:
                bound = (data["start"], data["end"])

            question, gt_answer = qa_template(data)  # keep for record/debug
            media_path = os.path.join(s["prefix"], data["video"])

            try:
                pred = get_router_output(
                    model=model,
                    video_processor=processor["video"],
                    tokenizer=tokenizer,
                    media=media_path,
                    user_qs=question,
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

            if pred is None:
                skipped += 1
                f_err.write(json.dumps({
                    "task_type": task_type,
                    "video": data["video"],
                    "media_path": media_path,
                    "error": "decode_failed_or_pred_none",
                }, ensure_ascii=False) + "\n")
                continue

            record = {
                "task_type": task_type,
                "video": data["video"],
                "question": question,
                "answer": gt_answer,  # optional; you can delete this field if you want
                "pred": pred,          # router letter (A-H) ideally
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n===== MVBench Router Finished =====")
    print(f"Saved router predictions to: {out_path}")
    print(f"Saved errors to:            {err_path}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()