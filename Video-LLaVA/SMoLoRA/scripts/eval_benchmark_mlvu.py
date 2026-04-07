import os
import json
from tqdm import tqdm
from difflib import SequenceMatcher
from videollava.eval.video.run_inference_video_qa import get_model_output
from videollava.mm_utils import get_model_name_from_path
from videollava.model.builder import load_pretrained_model
from openai import OpenAI  # 假设你正在使用的API客户端

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
        cand_clean = cand.strip().lower()
        if cand_clean in pred_clean:
            return cand.strip()

    best_match = candidates[0].strip()
    best_ratio = 0.0
    for cand in candidates:
        ratio = SequenceMatcher(None, pred_clean, cand.strip().lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = cand.strip()
    return best_match

def evaluate_task(model, processor, tokenizer, args, json_path, video_dir, output_dir, skip_accuracy=False):
    task_name = os.path.splitext(os.path.basename(json_path))[0]

    with open(json_path, 'r') as f:
        gt_data = json.load(f)

    results = []
    correct_count = 0
    total = 0
    video_formats = ['.mp4', '.webm', '.avi', '.mov', '.mkv']

    for sample in tqdm(gt_data, desc=f"Evaluating {task_name}"):
        video_name = sample["video"]
        question = sample["question"]
        candidates = sample.get("candidates", [])
        answer = sample.get("answer", "").strip()

        video_path = os.path.join(video_dir, video_name)
        if not os.path.exists(video_path):
            print(f" 视频未找到: {video_name}")
            continue

        if candidates:
            prompt = (
                f"{question}\n\n以下是答案选项：\n"
                + "\n".join(f"- {c.strip()}" for c in candidates)
                + "\n\n请只选择一个答案，保持原样。"
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

            if not skip_accuracy and candidates:
                total += 1
                if normalize_text(pred) == normalize_text(answer):
                    correct_count += 1

        except Exception as e:
            print(f"⚠️ 处理视频 {video_name} 时出错: {e}")

    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, f"{task_name}_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=4)

    if not skip_accuracy:
        accuracy = correct_count / total if total > 0 else 0
        print(f" {task_name}: {accuracy:.4f} ({correct_count}/{total})")
        return task_name, accuracy, correct_count, total
    else:
        print(f" {task_name}: 跳过准确率计算（主观题）")
        return task_name, None, 0, 0

def convert_json_to_jsonl(input_files, output_dir):
    for input_path in input_files:
        output_path = os.path.join(output_dir, input_path.replace(".json", ".jsonl"))

        with open(input_path, "r") as f:
            data = json.load(f)

        converted = []
        for idx, item in enumerate(data):
            video_name = os.path.basename(item["video"]).split(".")[0]
            new_item = {
                "id": f"{video_name}_{idx:05d}",
                "question": item["question"],
                "answer": item["answer"],
                "pred": item["pred"]
            }
            converted.append(new_item)

        with open(output_path, "w") as f:
            for obj in converted:
                json.dump(obj, f, ensure_ascii=False)
                f.write("\n")

        print(f"✅ 已转换: {input_path} → {output_path}")

def evaluate_subjective_tasks(api_key, model_id, api_base, pred_path, output_json, result_path):
    client = OpenAI(api_key=api_key, base_url=api_base, timeout=1800)
    with open(pred_path, 'r') as file:
        predictions = [eval(i.strip()) for i in file.readlines()]

    id_list = [x['id'] for x in predictions]
    completed_files = set(os.listdir(output_json))
    incomplete_files = [f"{id}.json" for id in id_list if f"{id}.json" not in completed_files]

    # 处理主观题，调用OpenAI API
    # 这里需要填写处理主观题的具体代码

def run_model_and_convert(model_path, cache_dir, video_dir, json_dir, output_dir, subjective_files, device="cuda:0"):
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, model_base=None, model_name=model_name)
    model = model.to(device)

    task_files = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith(".json")]
    task_files.sort()

    summary = []
    total_correct = 0
    total_samples = 0

    for json_path in task_files:
        filename = os.path.basename(json_path)
        skip_accuracy = filename in subjective_files

        task_name, acc, c, t = evaluate_task(
            model, processor, tokenizer,
            args=type("obj", (object,), {
                "device": device,
                "model_path": model_path,
                "cache_dir": cache_dir,
                "model_max_length": 2048
            }),
            json_path=json_path,
            video_dir=video_dir,
            output_dir=output_dir,
            skip_accuracy=skip_accuracy
        )

        if not skip_accuracy:
            summary.append({"task": task_name, "accuracy": acc, "correct": c, "total": t})
            total_correct += c
            total_samples += t

    print("\n================= Summary =================")
    for item in summary:
        print(f"{item['task']:<30} {item['accuracy']:.4f} ({item['correct']}/{item['total']})")
    print("-------------------------------------------------")
    overall_acc = total_correct / total_samples if total_samples > 0 else 0
    print(f"Overall Accuracy: {overall_acc:.4f} ({total_correct}/{total_samples})")

    # Convert to JSONL
    convert_json_to_jsonl(subjective_files, output_dir)

if __name__ == "__main__":
    model_path = "/mnt/ShareDB_6TB/shiyichen2025/Video-LLaVA/Video-LLaVA-7B"
    cache_dir = "/mnt/ShareDB_6TB/shiyichen2025/Video-LLaVA/.cache"
    video_dir = "/mnt/ShareDB-3TB/syc/benchmark/MLVU/video"
    json_dir = "/mnt/ShareDB-3TB/syc/benchmark/MLVU/json"
    output_dir = "/mnt/ShareDB-3TB/syc/benchmark/MLVU/results"
    subjective_files = ["8_sub_scene_results.json", "9_summary_results.json"]  # 主观题文件名
    device = "cuda:0"

    run_model_and_convert(model_path, cache_dir, video_dir, json_dir, output_dir, subjective_files, device)
