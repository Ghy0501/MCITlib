# #!/bin/bash

TASK_ID=$1
HARD_PATH=/your_path/MCITlib_v3

if [ "$TASK_ID" == "1" ]; then
    pip install -e .
    bash scripts/MCITlib/Eval_MLLM_ACL/eval_OCR.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-ACL/OCR.json $HARD_PATH/configs/train_configs/SEFE/LLaVA/MLLM-ACL/eval/task1.json
elif [ "$TASK_ID" == "2" ]; then
    pip install -e .
    bash scripts/MCITlib/Eval_MLLM_ACL/eval_OCR.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-ACL/OCR.json $HARD_PATH/configs/train_configs/SEFE/LLaVA/MLLM-ACL/eval/task2.json
    pip install -e .
    bash scripts/MCITlib/Eval_MLLM_ACL/eval_Math.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-ACL/Math.json $HARD_PATH/configs/train_configs/SEFE/LLaVA/MLLM-ACL/eval/task2.json
elif [ "$TASK_ID" == "3" ]; then
    pip install -e .
    bash scripts/MCITlib/Eval_MLLM_ACL/eval_OCR.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-ACL/OCR.json $HARD_PATH/configs/train_configs/SEFE/LLaVA/MLLM-ACL/eval/task3.json
    pip install -e .
    bash scripts/MCITlib/Eval_MLLM_ACL/eval_Math.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-ACL/Math.json $HARD_PATH/configs/train_configs/SEFE/LLaVA/MLLM-ACL/eval/task3.json
    pip install -e .
    bash scripts/MCITlib/Eval_MLLM_ACL/eval_VP.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-ACL/VP.json $HARD_PATH/configs/train_configs/SEFE/LLaVA/MLLM-ACL/eval/task3.json
else
    pip install -e .
    bash scripts/MCITlib/Eval_MLLM_ACL/eval_OCR.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-ACL/OCR.json $HARD_PATH/configs/train_configs/SEFE/LLaVA/MLLM-ACL/eval/task4.json
    pip install -e .
    bash scripts/MCITlib/Eval_MLLM_ACL/eval_Math.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-ACL/Math.json $HARD_PATH/configs/train_configs/SEFE/LLaVA/MLLM-ACL/eval/task4.json
    pip install -e .
    bash scripts/MCITlib/Eval_MLLM_ACL/eval_VP.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-ACL/VP.json $HARD_PATH/configs/train_configs/SEFE/LLaVA/MLLM-ACL/eval/task4.json
    pip install -e .
    bash scripts/MCITlib/Eval_MLLM_ACL/eval_APP.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-ACL/APP.json $HARD_PATH/configs/train_configs/SEFE/LLaVA/MLLM-ACL/eval/task4.json
fi