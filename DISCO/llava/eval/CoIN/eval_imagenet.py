import os
import argparse
import json
import re
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-file', type=str, default='./playground/Instructions_slim/ImageNet/test.json')
    parser.add_argument('--result-file', type=str, default='./results/CoIN_normaltrain_testslim/ImageNet/OCRVQA/merge.jsonl')
    parser.add_argument('--output-dir', type=str, default='./results/CoIN_normaltrain_testslim/ImageNet/OCRVQA')
    return parser.parse_args()


def eval_single(test_file, result_file):
    annotations = json.load(open(test_file))
    answers = [test['answer'] for test in annotations]
    results = [json.loads(line) for line in open(result_file)]

    total = len(results)
    right = 0
    false_answers = []
    for index in tqdm(range(total)):
        text = answers[index]
        label = results[index]
        if (text in label['text']) or (label['text'] in text):
            right += 1
        else:
            label['ground_truth'] = text
            false_answers.append(label)

    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(total, 100. * right / total))
    # 将结果写入文件
    if args.output_dir is not None:
        output_file = os.path.join(args.output_dir, 'Result.text')
        with open(output_file, 'w') as f:
            f.write('Samples: {}\nAccuracy: {:.2f}%\n'.format(total, 100. * right / total))
            json.dump(false_answers, f, indent=4)


if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_single(args.test_file, args.result_file)
