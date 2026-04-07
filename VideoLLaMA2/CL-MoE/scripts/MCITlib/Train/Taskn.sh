#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <model_config.json> <data_config.json> <train_config.json>"
    exit 1
fi

MODEL_CONFIG=$1
DATA_CONFIG=$2
TRAIN_CONFIG=$3

read_config() {
    python3 -c "import json; print(json.load(open('$1'))['$2'])"
}

GPU_NUM=$(read_config "$TRAIN_CONFIG" gpu_num)
LORA_RANK=$(read_config "$TRAIN_CONFIG" rank)
EXPERT=$(read_config "$TRAIN_CONFIG" expert_num)
OUTPUT_DIR=$(read_config "$TRAIN_CONFIG" output_dir)
EPOCH=$(read_config "$TRAIN_CONFIG" epoch)
BATCH_SIZE=$(read_config "$TRAIN_CONFIG" batch_size)
GRAD_ACC=$(read_config "$TRAIN_CONFIG" grad_acc)
LR=$(read_config "$TRAIN_CONFIG" lr)
PREVIOUS=$(read_config "$TRAIN_CONFIG" previous_model)
TASK=$(read_config "$TRAIN_CONFIG" task)

MODEL_PATH=$(read_config "$MODEL_CONFIG" model_name)
VISION_TOWER=$(read_config "$MODEL_CONFIG" vision_tower)

DATA_PATH=$(read_config "$DATA_CONFIG" train_path)
DATA_FOLDER=$(read_config "$DATA_CONFIG" train_folder)

GPU_LIST=""
for i in $(seq 0 $((GPU_NUM-1))); do
    GPU_LIST+="$i,"
done
GPU_LIST=${GPU_LIST%,}

echo "Using GPUs: $GPU_LIST"
echo "Gradient Accumulation Steps: $GRAD_ACC"

export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=$WANDB_PROJECT

deepspeed --include localhost:$GPU_LIST --master_port 2326 videollama2/train.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True --lora_r $LORA_RANK --lora_alpha $((LORA_RANK * 2)) --mm_projector_lr 1e-5 \
    --expert_num $EXPERT \
    --model_type videollama2_mistral \
    --model_path $MODEL_PATH \
    --previous_task_model_path $PREVIOUS \
    --vision_tower $VISION_TOWER \
    --mm_projector_type stc_connector \
    --data_path $DATA_PATH \
    --data_folder $DATA_FOLDER \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio pad \
    --num_frames 8 \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRAD_ACC \
    --save_strategy "steps" \
    --save_steps 50000 \
    --learning_rate $LR \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to tensorboard \
    --task $TASK