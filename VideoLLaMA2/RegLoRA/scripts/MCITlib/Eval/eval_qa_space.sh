#!/bin/bash

TRAIN_CONFIG=$1

read_config() {
    python3 -c "import json; print(json.load(open('$1'))['$2'])"
}

TASK="space"
output_dir=$(read_config "$TRAIN_CONFIG" result_path)
STAGE=$(read_config "$TRAIN_CONFIG" stage)
judge_model_path=$(read_config "$TRAIN_CONFIG" judge_path)

output_dir="$output_dir/$TASK/$STAGE"
pred_path="$output_dir/merge.jsonl"
output_json="$output_dir/results.json"
result_path="$output_dir/results.txt"
output_dir="$output_dir/qwen3"


num_tasks=20

python3 videollama2/eval/eval_video_qa_qwen.py \
   --model_path ${judge_model_path} \
   --pred_path ${pred_path} \
   --output_dir ${output_dir} \
   --output_json ${output_json} \
   --result_path ${result_path}