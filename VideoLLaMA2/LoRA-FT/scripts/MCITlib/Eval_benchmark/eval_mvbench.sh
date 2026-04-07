TRAIN_CONFIG=$1

read_config() {
    python3 -c "import json; print(json.load(open('$1'))['$2'])"
}

TASK="mvbench"

output_dir=$(read_config "$TRAIN_CONFIG" result_path)
STAGE=$(read_config "$TRAIN_CONFIG" stage)


output_file=${output_dir}/${TASK}/${STAGE}/answers/merge.json



python3 videollama2/eval/eval_video_mcqa_mvbench.py \
    --pred_path ${output_file} \
