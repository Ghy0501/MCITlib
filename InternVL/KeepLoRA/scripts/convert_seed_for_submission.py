import os
import json
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", type=str)
    parser.add_argument("--result-file", type=str)
    parser.add_argument("--result-upload-file", type=str)
    parser.add_argument("--output-dir", type=str)
    return parser.parse_args()


def eval_single(result_file, eval_only_type=None, output_file=None):
    results = {}
    for line in open(result_file, encoding="utf-8"):
        row = json.loads(line)
        results[row['question_id']] = row

    type_counts = {}
    correct_counts = {}
    for question_data in data['questions']:
        if eval_only_type is not None and question_data.get('data_type') != eval_only_type:
            continue
        data_type = question_data['question_type_id']
        type_counts[data_type] = type_counts.get(data_type, 0) + 1
        try:
            question_id = int(question_data['question_id'])
        except:
            question_id = question_data['question_id']
        if question_id not in results:
            correct_counts[data_type] = correct_counts.get(data_type, 0)
            continue
        row = results[question_id]
        if row.get('text') == question_data.get('answer'):
            correct_counts[data_type] = correct_counts.get(data_type, 0) + 1
        else:
            correct_counts[data_type] = correct_counts.get(data_type, 0)

    total_count = 0
    total_correct = 0
    for data_type in sorted(type_counts.keys()):
        cnt = type_counts[data_type]
        cor = correct_counts.get(data_type, 0)
        accuracy = (cor / cnt * 100) if cnt > 0 else 0.0
        if eval_only_type is None:
            msg = f"{ques_type_id_to_name.get(data_type, str(data_type))}: {accuracy:.2f}%"
            print(msg)
            if output_file is not None:
                output_file.write(msg + "\n")

        total_count += cnt
        total_correct += cor

    total_accuracy = (total_correct / total_count * 100) if total_count > 0 else 0.0
    if eval_only_type is None:
        msg = f"Total accuracy: {total_accuracy:.2f}%"
    else:
        msg = f"{eval_only_type} accuracy: {total_accuracy:.2f}%"
    print(msg)
    if output_file is not None:
        output_file.write(msg + "\n")

    return results


if __name__ == "__main__":
    args = get_args()
    data = json.load(open(args.annotation_file, encoding="utf-8"))
    ques_type_id_to_name = {id_: name for name, id_ in data['question_type'].items()}

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "result.txt"), "w", encoding="utf-8") as output_file:
        results = eval_single(args.result_file, output_file=output_file)
        eval_single(args.result_file, eval_only_type='image', output_file=output_file)
        eval_single(args.result_file, eval_only_type='video', output_file=output_file)

    with open(args.result_upload_file, 'w', encoding="utf-8") as fp:
        for question in data['questions']:
            qid = question['question_id']
            if qid in results:
                result = results[qid]
            else:
                try:
                    result = results[int(qid)]
                except:
                    result = {'text': ''}
            fp.write(json.dumps({
                'question_id': qid,
                'prediction': result.get('text', '')
            }, ensure_ascii=False) + '\n')