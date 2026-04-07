import os
import argparse
import json
import ast
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI  # 使用官方 OpenAI 库

# ================= 配置区域 =================
# 3. API 配置
API_KEY = "48b61489-fb36-48e2-9db4-d4b37301a6c7"
MODEL_ID = "deepseek-v3-1-250821"
API_BASE = "https://ark.cn-beijing.volces.com/api/v3" # 火山引擎的 Base URL

# 4. 并发配置
MAX_WORKERS = 20

if not API_KEY:
    raise ValueError("请设置 API_KEY")

# 初始化客户端 (OpenAI 客户端连接火山引擎)
client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE,
    timeout=1800, 
)
# ===========================================

def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-ark")
    parser.add_argument("--pred_path", default=r'', help="The path to file containing prediction.")
    parser.add_argument("--output_dir", default=r'', help="The path to save annotation json files.")
    parser.add_argument("--output_json", default=r'', help="The path to save annotation final combined json file.")
    parser.add_argument("--result_path", default=r'', help="The path to save evaluation results txt file.")
    parser.add_argument("--api_key", default="", help="Deprecated.")
    parser.add_argument("--api_base", default="", type=str, help="Deprecated.")
    parser.add_argument("--num_tasks", default=1, type=int, help="Deprecated.")
    args = parser.parse_args()
    return args

def clean_response_text(text):
    text = text.strip()
    if text.startswith("```python"):
        text = text[9:]
    elif text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def process_single_file(file_name, prediction_set, output_dir):
    key = file_name[:-5]
    output_path = os.path.join(output_dir, f"{key}.json")
    
    if os.path.exists(output_path):
        return "Skipped"

    if key not in prediction_set:
        return f"Error: Key {key} not found"

    qa_set = prediction_set[key]
    question = qa_set['q']
    answer = qa_set['a']
    pred = qa_set['pred']

    max_retries = 5
    for attempt in range(max_retries):
        try:
            # 调用方式与 Ark SDK 完全一致
            completion = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {
                        "role": "system",
                        "content":
                            "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                            "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                            "------"
                            "##INSTRUCTIONS: "
                            "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                            "- Consider synonyms or paraphrases as valid matches.\n"
                            "- Evaluate the correctness of the prediction compared to the answer."
                    },
                    {
                        "role": "user",
                        "content":
                            "Please evaluate the following video-based question-answer pair:\n\n"
                            f"Question: {question}\n"
                            f"Correct Answer: {answer}\n"
                            f"Predicted Answer: {pred}\n\n"
                            "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                            "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                            "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
                    }
                ]
            )
            
            response_message = completion.choices[0].message.content
            cleaned_response = clean_response_text(response_message)
            response_dict = ast.literal_eval(cleaned_response)
            result_qa_pair = [response_dict, qa_set]

            with open(output_path, "w") as f:
                json.dump(result_qa_pair, f)
            
            return "Success"

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed processing file '{key}': {e}")
                return f"Failed: {e}"
            time.sleep((2 ** attempt) + 1)

def main():
    args = parse_args()

    print(f"Loading predictions from {args.pred_path}...")
    with open(args.pred_path, 'r') as file:
        new_pred_contents = [eval(i.strip()) for i in file.readlines()]

    id_list = [x['id'] for x in new_pred_contents]
    caption_files = [f"{id}.json" for id in id_list]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    prediction_set = {}
    for sample in new_pred_contents:
        id = sample['id']
        qa_set = {"q": sample['question'], "a": sample['answer'], "pred": sample['pred']}
        prediction_set[id] = qa_set

    completed_files = set(os.listdir(args.output_dir))
    incomplete_files = [f for f in caption_files if f not in completed_files]
    
    print(f"Remaining to process: {len(incomplete_files)}")

    if incomplete_files:
        print(f"Starting processing with {MAX_WORKERS} workers...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(process_single_file, file_name, prediction_set, args.output_dir): file_name 
                for file_name in incomplete_files
            }
            for future in tqdm(as_completed(futures), total=len(incomplete_files), desc="Processing"):
                pass

    print("Combining results...")
    combined_contents = {}
    for file_name in os.listdir(args.output_dir):
        if file_name.endswith(".json"):
            try:
                with open(os.path.join(args.output_dir, file_name), "r") as f:
                    combined_contents[file_name[:-5]] = json.load(f)
            except: pass

    with open(args.output_json, "w") as f:
        json.dump(combined_contents, f)
    
    score_sum = 0
    count = 0
    yes_count = 0
    no_count = 0
    
    for key, result in combined_contents.items():
        try:
            count += 1
            score_sum += float(result[0]['score'])
            if "yes" in str(result[0]['pred']).lower():
                yes_count += 1
            elif "no" in str(result[0]['pred']).lower():
                no_count += 1
        except: pass

    if count > 0:
        acc = yes_count / (yes_count + no_count) if (yes_count + no_count) > 0 else 0
        avg = score_sum / count
        print(f"Accuracy: {acc:.4f}, Avg Score: {avg:.4f}")
        with open(args.result_path, "w") as f:
            f.write(f"Accuracy: {acc:.4f}\nAverage score: {avg:.4f}\n")

if __name__ == "__main__":
    main()