# #!/bin/bash

HARD_PATH=/your_path/MCITlib_v3

pip install -e .
bash scripts/MCITlib/Eval_MLLM_ACL/eval_OCR.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-ACL/OCR.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/MLLM-ACL/eval/zero_shot.json
pip install -e .
bash scripts/MCITlib/Eval_MLLM_ACL/eval_Math.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-ACL/Math.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/MLLM-ACL/eval/zero_shot.json
pip install -e .
bash scripts/MCITlib/Eval_MLLM_ACL/eval_VP.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-ACL/VP.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/MLLM-ACL/eval/zero_shot.json
pip install -e .
bash scripts/MCITlib/Eval_MLLM_ACL/eval_APP.sh $HARD_PATH/configs/model_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-ACL/APP.json $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/MLLM-ACL/eval/zero_shot.json