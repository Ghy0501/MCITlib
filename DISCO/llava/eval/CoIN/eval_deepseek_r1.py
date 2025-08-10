import os
import argparse
import json
import re
import time
from openai import OpenAI
from multiprocessing import Pool, cpu_count


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str, default='./LLaVA/cl_dataset/TextVQA/TextVQA_0.5.1_val.json')
    parser.add_argument('--result-file', type=str, default='./LLaVA/results/Instructions/TextVQA/Zero_shot/merge.jsonl')
    parser.add_argument('--output-dir', type=str)
    return parser.parse_args()

def prompt_processor(prompt):
    if prompt.startswith('OCR tokens:'):
        pattern = r"Question: (.*?) Short answer:"
        match = re.search(pattern, prompt, re.DOTALL)
        question = match.group(1)
    elif 'Reference OCR token:' in prompt and len(prompt.split('\n')) == 3:
        if prompt.startswith('Reference OCR token:'):
            question = prompt.split('\n')[1]
        else:
            question = prompt.split('\n')[0]
    elif len(prompt.split('\n')) == 2:
        question = prompt.split('\n')[0]
    else:
        assert False

    return question.lower()


def eval_single(annotation_file, result_file):
    annotations = json.load(open(annotation_file))
    annotations = {annotation['question_id']: annotation for annotation in annotations}
    results = [json.loads(line) for line in open(result_file)]

    total = len(results)
    right = 0
    answer_gt_file = []
    for result in results:
        annotation = annotations[result['question_id']]
        pred = result['text']
        ground_truth = annotation['answer']
        if pred.upper() == ground_truth.upper():
            right += 1
        answer_gt_file.append({
        "pred": pred,
        "ground_truth": ground_truth
        })
        # if ground_truth.upper() in pred.upper():
        #     right += 1
    ans_gt_file = os.path.join(args.output_dir, 'ans_gt.json')
    with open(ans_gt_file, "w", encoding="utf-8") as f:
        json.dump(answer_gt_file, f, ensure_ascii=False, indent=4)

    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(total, 100. * right / total))
    #将结果写入文件
    if args.output_dir is not None:
        output_file = os.path.join(args.output_dir, 'Result.text')
        with open(output_file, 'w') as f:
            f.write('Samples: {}\nAccuracy: {:.2f}%\n'.format(total, 100. * right / total))

    return ans_gt_file

def process_batch(api_key, batch):
    message = (
        "You are an expert evaluator assessing the semantic similarity between model-generated responses and ground truth answers. "
        "For each pair, provide a similarity score between 0 and 10 based on meaning, where 10 means the two responses are identical in meaning, "
        "and 0 means they are completely unrelated. Use the format 'Score: X' for each pair without explanations."
        "\n\nPairs:\n" +
        "\n".join([f"{i+1}. Model Response: {item['pred']}\n   Ground Truth: {item['ground_truth']}" for i, item in enumerate(batch)])
    )

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model='deepseek-chat',
        messages=[
            {"role": "system", "content": "You are an AI assistant evaluating the semantic similarity of responses."},
            {"role": "user", "content": message},
        ],
        stream=False
    )

    evaluation_text = response.choices[0].message.content

    # 提取 "Score: X" 形式的数值
    scores = []
    for line in evaluation_text.splitlines():
        if "Score:" in line:
            try:
                score = float(line.split(":")[1].strip())
                scores.append(score)
            except ValueError:
                continue  # 跳过无法解析的行

    average_score = sum(scores) / len(scores) if scores else 0  # 避免除零错误

    return average_score, len(batch)


def deepseek_chat_final(api_key, path, batch_size=10):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    num_batches = len(batches)

    print(f"Total data: {len(data)}, Total batches: {num_batches}, Batch size: {batch_size}")

    total_score = 0
    total_samples = 0

    with Pool(cpu_count()) as pool:
        results = pool.starmap(
            process_batch, [(api_key, batch) for batch in batches]
        )

        for batch_score, batch_total in results:
            total_score += batch_score * batch_total  # 还原该批次总分
            total_samples += batch_total

    overall_average_score = total_score / total_samples if total_samples > 0 else 0
    return overall_average_score
    

if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        ans_gt_file = eval_single(args.annotation_file, args.result_file)

        # api_key = "sk-d9e2eec1fa154fa78d4d934bb6bba976"   # deepseek  sk-d9e2eec1fa154fa78d4d934bb6bba976

        # batch_size = 2 
        # overall_accuracy = deepseek_chat_final(api_key, ans_gt_file, batch_size=batch_size)
        # print(f"Overall Accuracy: {overall_accuracy*10:.2f}")
        # if args.output_dir is not None:
        #     output_file = os.path.join(args.output_dir, 'Result_api.text')
        #     with open(output_file, 'w') as f:
        #         f.write('Accuracy: {:.2f}%\n'.format(overall_accuracy*10))
