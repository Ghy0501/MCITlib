#!/bin/bash


MODEL_CONFIG=$1
DATA_CONFIG=$2
TRAIN_CONFIG=$3

read_config() {
    python3 -c "import json; print(json.load(open('$1'))['$2'])"
}

GPU_NUM=$(read_config "$TRAIN_CONFIG" gpu_num)
RANK=$(read_config "$TRAIN_CONFIG" rank)
EXPERT=$(read_config "$TRAIN_CONFIG" expert_num)
MODEL_PATH=$(read_config "$MODEL_CONFIG" model_name)
PREVIOUS=$(read_config "$TRAIN_CONFIG" previous_model)
JSON_FOLDER=$(read_config "$DATA_CONFIG" train_path)
VIDEO_FOLDER=$(read_config "$DATA_CONFIG" train_folder)
VIDEO_TOWER=$(read_config "$MODEL_CONFIG" vision_tower)
OUTPUT_DIR=$(read_config "$TRAIN_CONFIG" output_dir)
EPOCH=$(read_config "$TRAIN_CONFIG" epoch)
BATCH_SIZE=$(read_config "$TRAIN_CONFIG" batch_size)
GRAD_ACC=$(read_config "$TRAIN_CONFIG" grad_acc)
LR=$(read_config "$TRAIN_CONFIG" lr)
TASK=$(read_config "$TRAIN_CONFIG" task)

GPU_LIST=""
for i in $(seq 0 $((GPU_NUM-1))); do
    GPU_LIST+="$i,"
done
GPU_LIST=${GPU_LIST%,}


HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed --include localhost:$GPU_LIST --master_port 2321 videollava/train/train_mem.py \
    --lora_enable True --lora_r $RANK --lora_alpha $((RANK * 2)) --mm_projector_lr 1e-5 \
    --expert_num $EXPERT \
    --learning_rate $LR \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps $GRAD_ACC \
    --deepspeed ./scripts/zero2_offload.json \
    --model_name_or_path $MODEL_PATH \
    --previous_task_model_path $PREVIOUS \
    --version v1 \
    --data_path $JSON_FOLDER \
    --video_folder $VIDEO_FOLDER \
    --video_tower $VIDEO_TOWER \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --tokenizer_model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir" \
    --task $TASK