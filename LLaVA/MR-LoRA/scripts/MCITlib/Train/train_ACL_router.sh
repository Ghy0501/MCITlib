#!/bin/bash

HARD_PATH=/your_path/MCITlib_v3

bash scripts/MCITlib/Train/Task1_router.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/MLLM-ACL/OCR.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/MLLM-ACL/train_router/task1.json
bash scripts/MCITlib/Eval_MLLM_ACL/Eval_finetune1.sh 1 $HARD_PATH

bash scripts/MCITlib/Train/Task1_router.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/MLLM-ACL/Math.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/MLLM-ACL/train_router/task2.json
bash scripts/MCITlib/Eval_MLLM_ACL/Eval_finetune1.sh 2 $HARD_PATH

bash scripts/MCITlib/Train/Task1_router.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/MLLM-ACL/VP.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/MLLM-ACL/train_router/task3.json
bash scripts/MCITlib/Eval_MLLM_ACL/Eval_finetune1.sh 3 $HARD_PATH

bash scripts/MCITlib/Train/Task1_router.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/MLLM-ACL/APP.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/MLLM-ACL/train_router/task4.json
bash scripts/MCITlib/Eval_MLLM_ACL/Eval_finetune1.sh 4 $HARD_PATH