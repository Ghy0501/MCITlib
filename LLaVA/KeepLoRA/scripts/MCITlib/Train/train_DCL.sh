#!/bin/bash

HARD_PATH=/your_path/MCITlib_v3

bash scripts/MCITlib/Train/extract_weights.sh \
   $HARD_PATH/configs/model_configs/llava.json \
   $HARD_PATH/configs/train_configs/KeepLoRA/LLaVA/MLLM-DCL/train_pre/task0.json \
   0.6

bash scripts/MCITlib/Train/extract_gradients.sh \
   $HARD_PATH/configs/model_configs/llava.json \
   $HARD_PATH/configs/data_configs/MLLM-DCL/RS.json \
   $HARD_PATH/configs/train_configs/KeepLoRA/LLaVA/MLLM-DCL/train_pre/task1.json \
   0.2 \
   fixed_rank

bash scripts/MCITlib/Train/Task1.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/MLLM-DCL/RS.json \
    $HARD_PATH/configs/train_configs/KeepLoRA/LLaVA/MLLM-DCL/train/task1.json

bash scripts/MCITlib/Train/extract_gradients.sh \
   $HARD_PATH/configs/model_configs/llava.json \
   $HARD_PATH/configs/data_configs/MLLM-DCL/RS.json \
   $HARD_PATH/configs/train_configs/KeepLoRA/LLaVA/MLLM-DCL/train_post/task1.json \
   0.2 \
   energy
bash scripts/MCITlib/Eval_MLLM_DCL/Eval_finetune1.sh 1


bash scripts/MCITlib/Train/extract_gradients.sh \
   $HARD_PATH/configs/model_configs/llava.json \
   $HARD_PATH/configs/data_configs/MLLM-DCL/Med.json \
   $HARD_PATH/configs/train_configs/KeepLoRA/LLaVA/MLLM-DCL/train_pre/task2.json \
   0.2 \
   fixed_rank

bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/MLLM-DCL/Med.json \
    $HARD_PATH/configs/train_configs/KeepLoRA/LLaVA/MLLM-DCL/train/task2.json

bash scripts/MCITlib/Train/extract_gradients.sh \
   $HARD_PATH/configs/model_configs/llava.json \
   $HARD_PATH/configs/data_configs/MLLM-DCL/Med.json \
   $HARD_PATH/configs/train_configs/KeepLoRA/LLaVA/MLLM-DCL/train_post/task2.json \
   0.2 \
   energy

bash scripts/MCITlib/Eval_MLLM_DCL/Eval_finetune1.sh 2


bash scripts/MCITlib/Train/extract_gradients.sh \
   $HARD_PATH/configs/model_configs/llava.json \
   $HARD_PATH/configs/data_configs/MLLM-DCL/AD.json \
   $HARD_PATH/configs/train_configs/KeepLoRA/LLaVA/MLLM-DCL/train_pre/task3.json \
   0.2 \
   fixed_rank

bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/MLLM-DCL/AD.json \
    $HARD_PATH/configs/train_configs/KeepLoRA/LLaVA/MLLM-DCL/train/task3.json

bash scripts/MCITlib/Train/extract_gradients.sh \
   $HARD_PATH/configs/model_configs/llava.json \
   $HARD_PATH/configs/data_configs/MLLM-DCL/AD.json \
   $HARD_PATH/configs/train_configs/KeepLoRA/LLaVA/MLLM-DCL/train_post/task3.json \
   0.2 \
   energy

bash scripts/MCITlib/Eval_MLLM_DCL/Eval_finetune1.sh 3


bash scripts/MCITlib/Train/extract_gradients.sh \
   $HARD_PATH/configs/model_configs/llava.json \
   $HARD_PATH/configs/data_configs/MLLM-DCL/Sci.json \
   $HARD_PATH/configs/train_configs/KeepLoRA/LLaVA/MLLM-DCL/train_pre/task4.json \
   0.2 \
   fixed_rank

bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/MLLM-DCL/Sci.json \
    $HARD_PATH/configs/train_configs/KeepLoRA/LLaVA/MLLM-DCL/train/task4.json

bash scripts/MCITlib/Train/extract_gradients.sh \
   $HARD_PATH/configs/model_configs/llava.json \
   $HARD_PATH/configs/data_configs/MLLM-DCL/Sci.json \
   $HARD_PATH/configs/train_configs/KeepLoRA/LLaVA/MLLM-DCL/train_post/task4.json \
   0.2 \
   energy

bash scripts/MCITlib/Eval_MLLM_DCL/Eval_finetune1.sh 4


bash scripts/MCITlib/Train/extract_gradients.sh \
   $HARD_PATH/configs/model_configs/llava.json \
   $HARD_PATH/configs/data_configs/MLLM-DCL/Fin.json \
   $HARD_PATH/configs/train_configs/KeepLoRA/LLaVA/MLLM-DCL/train_pre/task5.json \
   0.2 \
   fixed_rank

bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/MLLM-DCL/Fin.json \
    $HARD_PATH/configs/train_configs/KeepLoRA/LLaVA/MLLM-DCL/train/task5.json

bash scripts/MCITlib/Train/extract_gradients.sh \
   $HARD_PATH/configs/model_configs/llava.json \
   $HARD_PATH/configs/data_configs/MLLM-DCL/Fin.json \
   $HARD_PATH/configs/train_configs/KeepLoRA/LLaVA/MLLM-DCL/train_post/task5.json \
   0.2 \
   energy

bash scripts/MCITlib/Eval_MLLM_DCL/Eval_finetune1.sh 5