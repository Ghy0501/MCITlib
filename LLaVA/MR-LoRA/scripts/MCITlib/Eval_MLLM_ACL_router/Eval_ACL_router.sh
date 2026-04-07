# #!/bin/bash

TASK_ID=$1
HARD_PATH=/your_path/MCITlib_v3

if [ "$TASK_ID" == "1" ]; then
    bash scripts/MCITlib/Eval_MLLM_ACL_router/eval_OCR.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-ACL/OCR.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/MLLM-ACL/eval_router/task1.json
elif [ "$TASK_ID" == "2" ]; then
    bash scripts/MCITlib/Eval_MLLM_ACL_router/eval_OCR.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-ACL/OCR.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/MLLM-ACL/eval_router/task2.json
    bash scripts/MCITlib/Eval_MLLM_ACL_router/eval_Math.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-ACL/Math.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/MLLM-ACL/eval_router/task2.json
elif [ "$TASK_ID" == "3" ]; then
    bash scripts/MCITlib/Eval_MLLM_ACL_router/eval_OCR.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-ACL/OCR.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/MLLM-ACL/eval_router/task3.json
    bash scripts/MCITlib/Eval_MLLM_ACL_router/eval_Math.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-ACL/Math.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/MLLM-ACL/eval_router/task3.json
    bash scripts/MCITlib/Eval_MLLM_ACL_router/eval_VP.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-ACL/VP.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/MLLM-ACL/eval_router/task3.json
else
    bash scripts/MCITlib/Eval_MLLM_ACL_router/eval_OCR.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-ACL/OCR.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/MLLM-ACL/eval_router/task4.json
    bash scripts/MCITlib/Eval_MLLM_ACL_router/eval_Math.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-ACL/Math.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/MLLM-ACL/eval_router/task4.json
    bash scripts/MCITlib/Eval_MLLM_ACL_router/eval_VP.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-ACL/VP.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/MLLM-ACL/eval_router/task4.json
    bash scripts/MCITlib/Eval_MLLM_ACL_router/eval_APP.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-ACL/APP.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/MLLM-ACL/eval_router/task4.json
fi