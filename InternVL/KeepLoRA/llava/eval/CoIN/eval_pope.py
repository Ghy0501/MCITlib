import os
import json
import argparse

def eval_pope(answers, label_file, output_file):

    def write_and_print(text):
        print(text)
        output_file.write(text + "\n")

    label_list = [json.loads(q)['label'] for q in open(label_file, 'r')]

    for answer in answers:
        text = answer['text']

        # Only keep the first sentence
        if text.find('.') != -1:
            text = text.split('.')[0]

        text = text.replace(',', '')
        words = text.split(' ')
        if 'No' in words or 'not' in words or 'no' in words:
            answer['text'] = 'no'
        else:
            answer['text'] = 'yes'

    for i in range(len(label_list)):
        if label_list[i] == 'no':
            label_list[i] = 0
        else:
            label_list[i] = 1

    pred_list = []
    for answer in answers:
        if answer['text'] == 'no':
            pred_list.append(0)
        else:
            pred_list.append(1)

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    write_and_print('TP\tFP\tTN\tFN\t')
    write_and_print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    write_and_print('Accuracy: {}'.format(acc))
    write_and_print('Precision: {}'.format(precision))
    write_and_print('Recall: {}'.format(recall))
    write_and_print('F1 score: {}'.format(f1))
    write_and_print('Yes ratio: {}'.format(yes_ratio))
    write_and_print('%.3f, %.3f, %.3f, %.3f, %.3f' % (f1, acc, precision, recall, yes_ratio))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-dir", type=str, required=True, help="Directory containing annotation files")
    parser.add_argument("--question-file", type=str, required=True, help="Path to the question file")
    parser.add_argument("--result-file", type=str, required=True, help="Path to the result file")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the result file")
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 打开输出文件（追加模式）
    with open(os.path.join(args.output_dir, "result.txt"), "a") as output_file:

        def write_and_print(text):
            """辅助函数：同时写入文件和打印"""
            print(text)
            output_file.write(text + "\n")

        # 加载问题和答案数据
        questions = [json.loads(line) for line in open(args.question_file)]
        questions = {question['question_id']: question for question in questions}
        answers = [json.loads(q) for q in open(args.result_file)]

        # 遍历注释目录中的文件
        for file in os.listdir(args.annotation_dir):
            # 过滤文件，确保符合命名规则
            assert file.startswith('coco_pope_')
            assert file.endswith('.json')
            category = file[10:-5]

            # 获取属于当前分类的答案
            cur_answers = [x for x in answers if questions[x['question_id']]['category'] == category]
            write_and_print('Category: {}, # samples: {}'.format(category, len(cur_answers)))

            # 调用 eval_pope，并将输出写入文件
            eval_pope(cur_answers, os.path.join(args.annotation_dir, file), output_file)
            write_and_print("====================================")