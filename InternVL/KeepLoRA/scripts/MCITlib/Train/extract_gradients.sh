#!/bin/bash

MODEL_CONFIG=$1
DATA_CONFIG=$2
TRAIN_CONFIG=$3
DATA_RATIO=$4
SVD_MODE=$5  # fixed_rank or energy

read_config() {
    python3 -c "import json; print(json.load(open('$1'))['$2'])"
}

GPU_NUM=$(read_config "$TRAIN_CONFIG" gpu_num)
OUTPUT_DIR=$(read_config "$TRAIN_CONFIG" output_dir)

gpu_list=""
for ((i=0; i<GPU_NUM; i++)); do
    gpu_list+="$i,"
done
gpu_list=${gpu_list%,}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$gpu_list}"

IFS=',' read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.train.extract_gradients \
        --data-config $DATA_CONFIG \
        --model-config $MODEL_CONFIG \
        --train-config $TRAIN_CONFIG \
        --output-dir $OUTPUT_DIR \
        --data-ratio $DATA_RATIO \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX &
done

wait

python -m llava.train.extract_gradients \
    --data-config $DATA_CONFIG \
    --model-config $MODEL_CONFIG \
    --train-config $TRAIN_CONFIG \
    --output-dir $OUTPUT_DIR \
    --num-chunks $CHUNKS \
    --merge-only \
    --svd-mode $SVD_MODE
