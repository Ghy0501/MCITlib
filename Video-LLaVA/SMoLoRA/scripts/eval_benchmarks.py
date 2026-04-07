import os
import sys
sys.path.append('/your_path/MCITlib_v3/Video-LLaVA/SMoLoRA')
import torch
import argparse
from videollava.eval.video.run_inference_video_qa import get_model_output
from videollava.mm_utils import get_model_name_from_path
from videollava.model.builder import load_pretrained_model
from difflib import SequenceMatcher
import json
from tqdm import tqdm

def normalize_text(text):
    text = text.strip().lower()
    if text and text[-1] in ".!?":
        text = text[:-1]
    return text

def clean_and_match_candidate(pred, candidates):
    if not candidates:
        return pred
    pred_clean = pred.strip().lower()
    for cand in candidates:
        if cand.strip().lower() in pred_clean:
            return cand.strip()
    best_match, best_ratio = candidates[0].strip(), 0
    for cand in candidates:
        ratio = SequenceMatcher(None, pred_clean, cand.strip().lower()).ratio()
        if ratio > best_ratio:
            best_match, best_ratio = cand.strip(), ratio
    return best_match

def evaluate_task(model, processor, tokenizer, args, json_path, video_dir, output_dir):
    task_name = os.path.splitext(os.path.basename(json_path))[0]
    with open(json_path, 'r') as f:
        gt_data = json.load(f)

    results, correct_count, total = [], 0, 0

    for sample in tqdm(gt_data, desc=f"Evaluating {task_name}"):
        video_name = sample["video"]
        question = sample["question"]
        candidates = sample.get("candidates", [])
        answer = sample.get("answer", "").strip()

        video_path = os.path.join(video_dir, video_name)
        if not os.path.exists(video_path):
            print(f" Video not found: {video_name}")
            continue

        if candidates:
            prompt = (
                f"{question}\n\n"
                "Here are the answer choices:\n" +
                "\n".join(f"- {c.strip()}" for c in candidates) +
                "\n\nPlease answer by copying only one of the choices above, exactly as it appears."
            )
        else:
            prompt = question

        try:
            raw_pred = get_model_output(model, processor['video'], tokenizer, video_path, prompt, args)
            pred = clean_and_match_candidate(raw_pred, candidates)
            results.append({
                "video": video_name,
                "question": question,
                "pred": pred,
                "candidates": candidates,
                "answer": answer
            })
            total += 1
            if normalize_text(pred) == normalize_text(answer):
                correct_count += 1
        except Exception as e:
            print(f"⚠️ Error processing {video_name}: {e}")

    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, f"{task_name}_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=4)

    accuracy = correct_count / total if total > 0 else 0
    print(f"{task_name}: {accuracy:.4f} ({correct_count}/{total})")
    return task_name, accuracy, correct_count, total

def run_benchmark_all(args):
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, _ = load_pretrained_model(args.model_path, args.model_base, model_name)

    if args.gpu_num > 1:
        print(f"Using {args.gpu_num} GPUs for training!")
        model = torch.nn.DataParallel(model)
    model = model.to(args.device)

    task_files = sorted([os.path.join(args.json_dir, f) for f in os.listdir(args.json_dir) if f.endswith(".json")])
    summary, total_correct, total_samples = [], 0, 0

    for json_path in task_files:
        task_name, acc, c, t = evaluate_task(
            model, processor, tokenizer,
            args=args,
            json_path=json_path,
            video_dir=args.video_dir,
            output_dir=args.output_dir
        )
        summary.append({"task": task_name, "accuracy": acc, "correct": c, "total": t})
        total_correct += c
        total_samples += t

    for item in summary:
        print(f"{item['task']:<30} {item['accuracy']:.4f} ({item['correct']}/{item['total']})")
    overall_acc = total_correct / total_samples if total_samples > 0 else 0
    print("-------------------------------------------------")
    print(f"Overall Accuracy: {overall_acc:.4f} ({total_correct}/{total_samples})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_base", type=str, required=False)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda") 
    parser.add_argument("--gpu_num", type=int, default=1)  
    args = parser.parse_args()

    run_benchmark_all(args)
