#!/bin/bash

MODEL_CONFIG=$1
DATA_CONFIG=$2

read_config() {
    python3 -c "import json; print(json.load(open('$1'))['$2'])"
}

GPU_NUM=$(read_config "$MODEL_CONFIG" gpu_num)
STAGE=$(read_config "$MODEL_CONFIG" stage)
MODELPATH=$(read_config "$MODEL_CONFIG" model_path)
MODELBASE=$(read_config "$MODEL_CONFIG" model_base)
DATA_PATH=$(read_config "$DATA_CONFIG" test_path)
IMAGE=$(read_config "$DATA_CONFIG" image_folder)
TEXT_TOWER=$(read_config "$MODEL_CONFIG" text_tower)
PREFIX_LEN=$(read_config "$MODEL_CONFIG" prefix_len)
CUR_TASK=$(read_config "$MODEL_CONFIG" cur_task)
NUM_TASK=$(read_config "$MODEL_CONFIG" num_task)

gpu_list=""
for ((i=0; i<GPU_NUM; i++)); do
    gpu_list+="$i,"
done
gpu_list=${gpu_list%,}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$gpu_list}"

IFS=',' read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"
CHUNKS=${#GPULIST[@]}

RESULT_DIR="./results/UCIT/each_dataset/ArxivQA"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.ModalPrompt.model_others \
        --model-path $MODELPATH \
        --model-base $MODELBASE \
        --question-file $DATA_PATH \
        --image-folder $IMAGE \
        --text-tower $TEXT_TOWER\
        --prefix-len $PREFIX_LEN \
        --cur-task $CUR_TASK \
        --num-tasks $NUM_TASK \
        --answers-file $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=$RESULT_DIR/$STAGE/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python -m llava.eval.ModalPrompt.eval_deepseek_r1 \
    --annotation-file $DATA_PATH \
    --result-file $output_file \
    --output-dir $RESULT_DIR/$STAGE \

# /mnt/cache/guohaiyang/miniconda3/envs/coin/bin/python -m llava.eval.LLaVA.ModalPrompt.create_prompt \
#     --rule ./ETrain/Eval/LLaVA/ModalPrompt/rule.json \
#     --questions ./playground/Instructions_Original/ScienceQA/test.json \
#     --results $output_file \