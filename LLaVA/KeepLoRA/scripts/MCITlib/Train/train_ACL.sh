#!/bin/bash

HARD_PATH=/your_path/MCITlib_v3

bash scripts/MCITlib/Train/extract_weights.sh \
   $HARD_PATH/configs/model_configs/llava.json \
   $HARD_PATH/configs/train_configs/KeepLoRA/LLaVA/MLLM-ACL/train_pre/task0.json \
   0.6

bash scripts/MCITlib/Train/extract_gradients.sh \
   $HARD_PATH/configs/model_configs/llava.json \
   $HARD_PATH/configs/data_configs/MLLM-ACL/OCR.json \
   $HARD_PATH/configs/train_configs/KeepLoRA/LLaVA/MLLM-ACL/train_pre/task1.json \
   0.2 \
   fixed_rank

bash scripts/MCITlib/Train/Task1.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/MLLM-ACL/OCR.json \
    $HARD_PATH/configs/train_configs/KeepLoRA/LLaVA/MLLM-ACL/train/task1.json

bash scripts/MCITlib/Train/extract_gradients.sh \
   $HARD_PATH/configs/model_configs/llava.json \
   $HARD_PATH/configs/data_configs/MLLM-ACL/OCR.json \
   $HARD_PATH/configs/train_configs/KeepLoRA/LLaVA/MLLM-ACL/train_post/task1.json \
   0.2 \
   energy
bash scripts/MCITlib/Eval_MLLM_ACL/Eval_finetune1.sh 1


bash scripts/MCITlib/Train/extract_gradients.sh \
   $HARD_PATH/configs/model_configs/llava.json \
   $HARD_PATH/configs/data_configs/MLLM-ACL/Math.json \
   $HARD_PATH/configs/train_configs/KeepLoRA/LLaVA/MLLM-ACL/train_pre/task2.json \
   0.2 \
   fixed_rank

bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/MLLM-ACL/Math.json \
    $HARD_PATH/configs/train_configs/KeepLoRA/LLaVA/MLLM-ACL/train/task2.json

bash scripts/MCITlib/Train/extract_gradients.sh \
   $HARD_PATH/configs/model_configs/llava.json \
   $HARD_PATH/configs/data_configs/MLLM-ACL/Math.json \
   $HARD_PATH/configs/train_configs/KeepLoRA/LLaVA/MLLM-ACL/train_post/task2.json \
   0.2 \
   energy
bash scripts/MCITlib/Eval_MLLM_ACL/Eval_finetune1.sh 2


bash scripts/MCITlib/Train/extract_gradients.sh \
   $HARD_PATH/configs/model_configs/llava.json \
   $HARD_PATH/configs/data_configs/MLLM-ACL/VP.json \
   $HARD_PATH/configs/train_configs/KeepLoRA/LLaVA/MLLM-ACL/train_pre/task3.json \
   0.2 \
   fixed_rank

bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/MLLM-ACL/VP.json \
    $HARD_PATH/configs/train_configs/KeepLoRA/LLaVA/MLLM-ACL/train/task3.json

bash scripts/MCITlib/Train/extract_gradients.sh \
   $HARD_PATH/configs/model_configs/llava.json \
   $HARD_PATH/configs/data_configs/MLLM-ACL/VP.json \
   $HARD_PATH/configs/train_configs/KeepLoRA/LLaVA/MLLM-ACL/train_post/task3.json \
   0.2 \
   energy
bash scripts/MCITlib/Eval_MLLM_ACL/Eval_finetune1.sh 3


bash scripts/MCITlib/Train/extract_gradients.sh \
   $HARD_PATH/configs/model_configs/llava.json \
   $HARD_PATH/configs/data_configs/MLLM-ACL/APP.json \
   $HARD_PATH/configs/train_configs/KeepLoRA/LLaVA/MLLM-ACL/train_pre/task4.json \
   0.2 \
   fixed_rank

bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/MLLM-ACL/APP.json \
    $HARD_PATH/configs/train_configs/KeepLoRA/LLaVA/MLLM-ACL/train/task4.json

bash scripts/MCITlib/Train/extract_gradients.sh \
   $HARD_PATH/configs/model_configs/llava.json \
   $HARD_PATH/configs/data_configs/MLLM-ACL/APP.json \
   $HARD_PATH/configs/train_configs/KeepLoRA/LLaVA/MLLM-ACL/train_post/task4.json \
   0.2 \
   energy
bash scripts/MCITlib/Eval_MLLM_ACL/Eval_finetune1.sh 4