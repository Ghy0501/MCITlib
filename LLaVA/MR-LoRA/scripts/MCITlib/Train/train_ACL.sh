#!/bin/bash

HARD_PATH=/your_path/MCITlib_v3

pip install -e .
bash scripts/MCITlib/Train/Task1.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/MLLM-ACL/OCR.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/MLLM-ACL/train/task1.json
bash scripts/MCITlib/Eval_MLLM_ACL/Eval_acl.sh 1

pip install -e .
bash scripts/MCITlib/Train/Task1.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/MLLM-ACL/Math.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/MLLM-ACL/train/task2.json
bash scripts/MCITlib/Eval_MLLM_ACL/Eval_acl.sh 2

pip install -e .
bash scripts/MCITlib/Train/Task1.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/MLLM-ACL/VP.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/MLLM-ACL/train/task3.json
bash scripts/MCITlib/Eval_MLLM_ACL/Eval_acl.sh 3

pip install -e .
bash scripts/MCITlib/Train/Task1.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/MLLM-ACL/APP.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/MLLM-ACL/train/task4.json
bash scripts/MCITlib/Eval_MLLM_ACL/Eval_acl.sh 4

python scripts/ACL_extract_router_result.py