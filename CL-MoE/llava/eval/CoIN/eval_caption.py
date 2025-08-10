import os
import argparse
import json
import re
from openai import OpenAI
from multiprocessing import Pool, cpu_count
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from collections import Counter
from collections import defaultdict

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str, default='./playground/Instructions_slim/VizWiz/val_new.json')
    parser.add_argument('--result-file', type=str, default='./results/CoIN_slim_new/VizWiz/Zero_shot/merge.jsonl')
    parser.add_argument('--output-dir', type=str)
    return parser.parse_args()

def create_coco_type(annotation_file, result_file, output_dir):
    results = [json.loads(line) for line in open(result_file)]

    pred_list = []
    total = len(results)
    right = 0
    coco_results = []
    image_id = 1
    for result in results:
        pred = result['text']

        coco_results.append({
            "image_id": int(image_id),  # 确保 image_id 是整数类型
            "caption": pred
        })
        image_id += 1
    output_file = 'pred_coco_type.json'
    output_path = os.path.join(output_dir, output_file)
    with open(output_path, 'w') as f_out:
        json.dump(coco_results, f_out, indent=4)
    return output_path, total

def load_json(file_path):
    """加载 JSON 文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def merge_captions(pred_file, val_file, output_file):
    # 加载预测文件和验证文件
    pred_data = load_json(pred_file)
    val_data = load_json(val_file)

    # 将验证数据中的 captions 根据 image_id 组织成字典，image_id -> [captions]
    val_dict = defaultdict(list)
    for item in val_data['annotations']:
        val_dict[item['image_id']].append(item['caption'])

    # 合并预测数据与验证数据
    merged_data = []
    for pred_item in pred_data:
        image_id = pred_item['image_id']
        pred_caption = pred_item['caption']

        # 获取真实值 captions
        gt_captions = val_dict.get(image_id, [])

        # 将预测与真实值合并
        merged_data.append({
            "pred": pred_caption,
            "ground_truth": gt_captions
        })

    # 将合并结果保存为 JSON 文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)

def eval_single(output_file, annotation_file, total):
    coco = COCO(annotation_file)  # Ground truth JSON file
    coco_res = coco.loadRes(output_file)  # Prediction JSON file

    coco_eval = COCOEvalCap(coco, coco_res)

    coco_eval.evaluate()

    metrics_to_print = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr"]
    results = []
    for metric, score in coco_eval.eval.items():
        if metric in metrics_to_print:
            score_percentage = score * 100.
            print(f"{metric}: {score_percentage:.2f}")
            results.append(score_percentage)
        

    print('Samples: {}\nAverage: {:.2f}%\n'.format(total, sum(results) / len(results)))
    #将结果写入文件
    if args.output_dir is not None:
        output_file = os.path.join(args.output_dir, 'Result.text')
        with open(output_file, 'w') as f:
            f.write('Samples: {}\nBleu_1: {:.2f}\nBleu_2: {:.2f}\nBleu_3: {:.2f}\nBleu_4: {:.2f}\nMETEOR: {:.2f}\nROUGE_L: {:.2f}\nCIDEr: {:.2f}\nAverage: {:.2f}\n'.format(
                total, results[0], results[1], results[2], results[3], results[4], results[5], results[6], sum(results) / len(results)))
    

def process_batch(api_key, batch):
    """
    对一个批次的数据进行评分，返回该批次所有样本的评分列表。
    """
    from openai import OpenAI  # 确保每个子进程加载必要的模块

    client = OpenAI(api_key=api_key, base_url="https://platform.llmprovider.ai/v1")

    message = (
        "Below are the model's predictions and the ground truth answers for a task. "
        "For each case, provide a semantic similarity score between 0 and 10 in the format 'Score: X', "
        "where X is your score. Always use the format 'Score:' and do not explain anything."
        "\n\nResults:\n" +
        "\n".join([f"{i+1}. Pred: {item['pred']}, Ground Truth: {item['ground_truth']}" for i, item in enumerate(batch)])
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are an AI assistant evaluating a model's prediction quality"}, {"role": "user", "content": message}],
        stream=False
    )

    evaluation_text = response.choices[0].message.content

    # 提取评分
    scores = []
    for line in evaluation_text.splitlines():
        score = float(line.split(":")[1].strip())
        scores.append(score)
    return scores

def deepseek_chat_final(api_key, path, batch_size=10):
    """
    使用多进程评估，返回所有样本的最终平均准确率。
    """
    # 加载 JSON 文件
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 分批处理
    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    num_batches = len(batches)

    print(f"Total data: {len(data)}, Total batches: {num_batches}, Batch size: {batch_size}")

    # 使用多进程池处理所有批次
    total_score = 0
    total_samples = 0
    with Pool(cpu_count()) as pool:
        results = pool.starmap(
            process_batch, [(api_key, batch) for batch in batches]
        )

        # 累计每个批次的评分总和和样本数
        for batch_scores in results:
            total_score += sum(batch_scores)
            total_samples += len(batch_scores)

    # 计算总体评分平均值
    overall_average_score = total_score / total_samples if total_samples > 0 else 0
    return overall_average_score
    



if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        output_file, total = create_coco_type(args.annotation_file, args.result_file, args.output_dir)
        ans_gt_file = os.path.join(args.output_dir, 'ans_gt.json')
        merge_captions(output_file, args.annotation_file, ans_gt_file)
        eval_single(output_file, args.annotation_file, total)

        # api_key = "sk-GdmqmU6fFWv5N0HlvYluLFzIbXIPNg3MHzPGeeV247092807Ba2e4487B9D5796cA3Be7dD4"

        # batch_size = 8 
        # overall_accuracy = deepseek_chat_final(api_key, ans_gt_file, batch_size=batch_size)
        # print(f"Overall Accuracy: {overall_accuracy*10:.2f}")
        # if args.output_dir is not None:
        #     output_file = os.path.join(args.output_dir, 'Result_api.text')
        #     with open(output_file, 'w') as f:
        #         f.write('Accuracy: {:.2f}%\n'.format(overall_accuracy*10))
