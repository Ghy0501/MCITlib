import os
import argparse
import json
import ast
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 保持全局常量
BATCH_SIZE = 128
MAX_NEW_TOKENS = 128

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to the model checkpoint")
    parser.add_argument("--pred_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--result_path", required=True)
    return parser.parse_args()

def clean_response_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 1)[-1]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def build_prompt(question, answer, pred):
    return (
        f"Please evaluate the following video-based question-answer pair:\n\n"
        f"Question: {question}\n"
        f"Correct Answer: {answer}\n"
        f"Predicted Answer: {pred}\n\n"
        "Provide your evaluation only as a score where the score is an integer value between 0 and 5, "
        "with 5 indicating the highest meaningful match. "
        "Please generate the response in the form of a Python dictionary string with key 'score' only, "
        "where value of 'score' is an INTEGER between 0 and 5. "
        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. "
        "Only provide the Python dictionary string. "
        "For example: {'score': 4}."
    )

def process_batch(file_names, prediction_set, output_dir, model, tokenizer):
    messages_list = []
    keys = []

    for fname in file_names:
        key = fname[:-5]
        if key not in prediction_set:
            continue

        qa = prediction_set[key]
        
        # === 修改 1: 强制转换为字符串，防止 None 或数字混入 ===
        q_text = str(qa.get("q", ""))
        a_text = str(qa.get("a", ""))
        pred_text = str(qa.get("pred", ""))
        
        prompt = build_prompt(q_text, a_text, pred_text)

        messages = [
            {
                "role": "system",
                "content":
                    "You are an intelligent chatbot designed for evaluating the correctness of generative outputs "
                    "for question-answer pairs. "
                    "Your task is to compare the predicted answer with the correct answer and determine the matching score. "
                    "Here's how you can accomplish the task:"
                    "------"
                    "##INSTRUCTIONS: "
                    "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                    "- Consider synonyms or paraphrases as valid matches.\n"
                    "- Evaluate the correctness of the prediction compared to the answer and provide a score from 0 to 5.\n"
                    "- 0: No match at all\n"
                    "- 1: Very poor match\n"
                    "- 2: Poor match\n"
                    "- 3: Fair match\n"
                    "- 4: Good match\n"
                    "- 5: Excellent/Perfect match"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        messages_list.append(messages)
        keys.append(key)

    if not messages_list:
        return

    # --- 修改后的逻辑 ---
    raw_texts = [
        tokenizer.apply_chat_template(
            m,
            tokenize=False,
            add_generation_prompt=True
        )
        for m in messages_list
    ]

    # 2. 【核心拦截层】强行物理清洗
    final_texts = []
    for t in raw_texts:
        if t is None:
            final_texts.append("")
            continue
        
        # 将对象转为纯字符串
        t_str = str(t)
        
        # 核心清洗逻辑：使用 'surrogateescape' 捕获并 'ignore' 丢弃所有非法代理项
        # 这一步能解决 UnicodeEncodeError: surrogates not allowed
        clean_t = t_str.encode('utf-8', 'replace').decode('utf-8', 'ignore')
        
        # 再次确认剔除 U+D800 - U+DFFF
        clean_t = "".join(c for c in clean_t if not (0xD800 <= ord(c) <= 0xDFFF))
        
        final_texts.append(clean_t)

    # 3. 最终调用（建议把 use_fast 改回 True 以提高速度，因为数据已经洗干净了）
    model_inputs = tokenizer(
        final_texts,
        padding=True,
        truncation=True,
        max_length=4096,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=MAX_NEW_TOKENS
        )

    for i, key in enumerate(keys):
        # 这里的切片是为了只获取新生成的 token
        output_ids = outputs[i][len(model_inputs.input_ids[i]):]
        content = tokenizer.decode(output_ids, skip_special_tokens=True)

        try:
            cleaned = clean_response_text(content)
            resp = ast.literal_eval(cleaned)
            
            # Ensure resp contains score key
            if "score" not in resp:
                print(f"[WARNING] {key}: Response missing 'score' key: {resp}")
                score = 0
            else:
                score = int(resp["score"])
                # Ensure score is within 0-5 range
                score = max(0, min(5, score))
            
            # Only save score
            result = [{"score": score}, prediction_set[key]]

            with open(os.path.join(output_dir, f"{key}.json"), "w") as f:
                json.dump(result, f)

        except Exception as e:
            # 打印错误但不中断，方便 debug
            print(f"[ERROR] {key}: Failed to parse '{content}'. Error: {e}")
            # Default score of 0 on error
            result = [{"score": 0}, prediction_set[key]]
            with open(os.path.join(output_dir, f"{key}.json"), "w") as f:
                json.dump(result, f)

def main():
    args = parse_args()

    # 1. Load tokenizer
    print(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        local_files_only=True,
        max_length=4096
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load model
    print(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        dtype="auto",
        attn_implementation="flash_attention_2",
        local_files_only=True
    )
    model.eval()

    # Configure generation parameters
    gen_cfg = model.generation_config
    gen_cfg.do_sample = False
    gen_cfg.temperature = None
    gen_cfg.top_p = None
    gen_cfg.top_k = None
    gen_cfg.num_beams = 1

    # 3. Load prediction data
    print(f"Loading predictions from {args.pred_path}")
    with open(args.pred_path, "r") as f:
        samples = [eval(line.strip()) for line in f]

    prediction_set = {}
    id_files = []

    for s in samples:
        # === 修改开始：强制转为字符串 ===
        s_id = str(s["id"]) 
        
        prediction_set[s_id] = {  # 使用字符串作为 key
            "q": s["question"],
            "a": s["answer"],
            "pred": s["pred"]
        }
        id_files.append(f"{s_id}.json")
        # === 修改结束 ===

    os.makedirs(args.output_dir, exist_ok=True)

    completed = set(os.listdir(args.output_dir))
    remaining = [f for f in id_files if f not in completed]

    print(f"Remaining to process: {len(remaining)}")

    # 4. Batch processing
    for i in tqdm(range(0, len(remaining), BATCH_SIZE), desc="Batch Processing"):
        batch_files = remaining[i:i + BATCH_SIZE]
        process_batch(batch_files, prediction_set, args.output_dir, model, tokenizer)

    # 5. Combine results and calculate metrics
    combined = {}
    for fname in os.listdir(args.output_dir):
        if fname.endswith(".json"):
            with open(os.path.join(args.output_dir, fname), "r") as f:
                combined[fname[:-5]] = json.load(f)

    with open(args.output_json, "w") as f:
        json.dump(combined, f)

    # 6. Calculate statistics
    scores = []
    for key, result in combined.items():
        try:
            # Extract score
            score = result[0].get("score", 0)
            # Ensure it's numeric
            score = float(score)
            scores.append(score)
        except (KeyError, ValueError, TypeError) as e:
            print(f"[ERROR] Invalid score for {key}: {result[0]}, error: {e}")
            scores.append(0)
    
    if scores:
        # Calculate average score
        avg_score_original = sum(scores) / len(scores)
        
        # Normalize to 0-100 scale (multiply by 20)
        avg_score_normalized = avg_score_original * 20
        
        # Print only the normalized score
        print(f"Normalized Average Score (0-100): {avg_score_normalized:.4f}")
        
        # Save results
        with open(args.result_path, "w") as f:
            f.write(f"Original Average Score (0-5): {avg_score_original:.4f}\n")
            f.write(f"Normalized Average Score (0-100): {avg_score_normalized:.4f}\n")
            f.write(f"Number of samples: {len(scores)}\n")
            
    else:
        print("No valid evaluation results")
        with open(args.result_path, "w") as f:
            f.write("No valid evaluation results\n")

if __name__ == "__main__":
    main()