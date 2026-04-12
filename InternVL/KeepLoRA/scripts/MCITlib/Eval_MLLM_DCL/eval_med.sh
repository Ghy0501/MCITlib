#!/bin/bash

MODEL_CONFIG=$1
DATA_CONFIG=$2
TRAIN_CONFIG=$3

read_config() {
    python3 -c "import json; print(json.load(open('$1'))['$2'])"
}

TASK="Med"
GPU_NUM=$(read_config "$TRAIN_CONFIG" gpu_num)
STAGE=$(read_config "$TRAIN_CONFIG" stage)
MODELPATH=$(read_config "$TRAIN_CONFIG" model_path)
MODELBASE=$(read_config "$MODEL_CONFIG" model_name)
DATA_PATH=$(read_config "$DATA_CONFIG" test_path)
IMAGE=$(read_config "$DATA_CONFIG" test_folder)
RESULT_PATH=$(read_config "$TRAIN_CONFIG" result_path)

gpu_list=""
for ((i=0; i<GPU_NUM; i++)); do
    gpu_list+="$i,"
done
gpu_list=${gpu_list%,}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$gpu_list}"

IFS=',' read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"
CHUNKS=${#GPULIST[@]}

RESULT_DIR="$RESULT_PATH/$TASK"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.CoIN.model_pvqa \
        --model-path $MODELPATH \
        --question-file $DATA_PATH \
        --image-folder $IMAGE \
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

python -m llava.eval.CoIN.eval_pvqa \
    --annotation-file $DATA_PATH \
    --result-file $output_file \
    --output-dir $RESULT_DIR/$STAGE \

# /mnt/cache/guohaiyang/miniconda3/envs/coin/bin/python -m llava.eval.LLaVA.CoIN.create_prompt \
#     --rule ./ETrain/Eval/LLaVA/CoIN/rule.json \
#     --questions ./playground/Instructions_Original/ScienceQA/test.json \
#     --results $output_file \