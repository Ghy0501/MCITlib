#!/bin/bash

MODEL_CONFIG=$1
DATA_CONFIG=$2
TRAIN_CONFIG=$3

read_config() {
    python3 -c "import json; print(json.load(open('$1'))['$2'])"
}

TASK="counting"
GPU_NUM=$(read_config "$TRAIN_CONFIG" gpu_num)
STAGE=$(read_config "$TRAIN_CONFIG" stage)
MODELPATH=$(read_config "$TRAIN_CONFIG" model_path)
MODELBASE=$(read_config "$MODEL_CONFIG" model_name)
video_dir=$(read_config "$DATA_CONFIG" test_folder)
TEXT_TOWER=$(read_config "$TRAIN_CONFIG" text_tower)
PREFIX_LEN=$(read_config "$TRAIN_CONFIG" prefix_len)
CUR_TASK=$(read_config "$TRAIN_CONFIG" cur_task)
NUM_TASK=$(read_config "$TRAIN_CONFIG" num_tasks)
gt_file_question=$(read_config "$DATA_CONFIG" test_q_path)
gt_file_answers=$(read_config "$DATA_CONFIG" test_a_path)
output_dir=$(read_config "$TRAIN_CONFIG" result_path)

output_dir="$output_dir/$TASK/$STAGE"

gpu_list=""
for ((i=0; i<GPU_NUM; i++)); do
    gpu_list+="$i,"
done
gpu_list=${gpu_list%,}

IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

echo $TEXT_TOWER

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 videollama2/eval/inference_video_qa.py \
        --model_path $MODELPATH \
        --model_base $MODELBASE \
        --text-tower $TEXT_TOWER\
        --prefix-len $PREFIX_LEN \
        --cur-task $CUR_TASK \
        --num-tasks $NUM_TASK \
        --video_dir $video_dir \
        --gt_file_question $gt_file_question \
        --gt_file_answers $gt_file_answers \
        --output_dir $output_dir \
        --output_name ${CHUNKS}_${IDX} \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX &
done

wait


output_file=${output_dir}/merge.jsonl


> "$output_file"


for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${output_dir}/${CHUNKS}_${IDX}.json >> "$output_file"
done

