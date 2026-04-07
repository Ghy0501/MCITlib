# #!/bin/bash

TASK_ID=$1
HARD_PATH=/your_path/MCITlib_v3

if [ "$TASK_ID" == "1" ]; then
    bash scripts/MCITlib/Eval_MLLM_DCL_router/eval_rs.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/RS.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/MLLM-DCL/eval_router/task1.json
elif [ "$TASK_ID" == "2" ]; then
    bash scripts/MCITlib/Eval_MLLM_DCL_router/eval_med.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/Med.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/MLLM-DCL/eval_router/task2.json
    bash scripts/MCITlib/Eval_MLLM_DCL_router/eval_rs.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/RS.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/MLLM-DCL/eval_router/task2.json
elif [ "$TASK_ID" == "3" ]; then
    bash scripts/MCITlib/Eval_MLLM_DCL_router/eval_med.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/Med.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/MLLM-DCL/eval_router/task3.json
    bash scripts/MCITlib/Eval_MLLM_DCL_router/eval_rs.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/RS.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/MLLM-DCL/eval_router/task3.json
    bash scripts/MCITlib/Eval_MLLM_DCL_router/eval_ad.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/AD.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/MLLM-DCL/eval_router/task3.json
elif [ "$TASK_ID" == "4" ]; then
    bash scripts/MCITlib/Eval_MLLM_DCL_router/eval_ad.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/AD.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/MLLM-DCL/eval_router/task4.json
    bash scripts/MCITlib/Eval_MLLM_DCL_router/eval_rs.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/RS.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/MLLM-DCL/eval_router/task4.json
    bash scripts/MCITlib/Eval_MLLM_DCL_router/eval_med.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/Med.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/MLLM-DCL/eval_router/task4.json
    bash scripts/MCITlib/Eval_MLLM_DCL_router/eval_sci.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/Sci.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/MLLM-DCL/eval_router/task4.json
else
    bash scripts/MCITlib/Eval_MLLM_DCL_router/eval_ad.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/AD.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/MLLM-DCL/eval_router/task5.json
    bash scripts/MCITlib/Eval_MLLM_DCL_router/eval_rs.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/RS.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/MLLM-DCL/eval_router/task5.json
    bash scripts/MCITlib/Eval_MLLM_DCL_router/eval_med.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/Med.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/MLLM-DCL/eval_router/task5.json
    bash scripts/MCITlib/Eval_MLLM_DCL_router/eval_sci.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/Sci.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/MLLM-DCL/eval_router/task5.json
    bash scripts/MCITlib/Eval_MLLM_DCL_router/eval_fin.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/Fin.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/MLLM-DCL/eval_router/task5.json
fi