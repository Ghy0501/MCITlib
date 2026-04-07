#!/bin/bash

TRAIN_CONFIG=$1

read_config() {
    python3 -c "import json; print(json.load(open('$1'))['$2'])"
}

TASK="longvideobench"
output_dir=$(read_config "$TRAIN_CONFIG" result_path)
STAGE=$(read_config "$TRAIN_CONFIG" stage)

output_dir="$output_dir/$TASK/$STAGE"
pred_path="$output_dir/merge.jsonl"
output_json="$output_dir/results.json"
result_path="$output_dir/results.txt"
output_dir="$output_dir/Qwen3-30B-A3B-Instruct-2507"



python3 videollava/eval/video/eval_video_qa_benchmark.py \
   --pred_path ${pred_path} \
   --output_dir ${output_dir} \
   --output_json ${output_json} \
   --result_path ${result_path} 