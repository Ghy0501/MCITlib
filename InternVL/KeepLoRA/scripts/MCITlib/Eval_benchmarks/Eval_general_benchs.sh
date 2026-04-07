# #!/bin/bash

DATASET_ID=$1
TASK_ID=$2
HARD_PATH=/your_path/MCITlib_v3

pip install -e .
bash scripts/MCITlib/Eval_benchmarks/eval_mmbench.sh \
    $HARD_PATH/configs/model_configs/internvl.json \
    $HARD_PATH/configs/data_configs/General_benchmark/mm_bench.json \
    $HARD_PATH/configs/train_configs/KeepLoRA/InternVL/${DATASET_ID}/eval/task${TASK_ID}.json

pip install -e .
bash scripts/MCITlib/Eval_benchmarks/eval_mme.sh \
    $HARD_PATH/configs/model_configs/internvl.json \
    $HARD_PATH/configs/data_configs/General_benchmark/mme.json \
    $HARD_PATH/configs/train_configs/KeepLoRA/InternVL/${DATASET_ID}/eval/task${TASK_ID}.json

pip install -e .
bash scripts/MCITlib/Eval_benchmarks/eval_pope.sh \
    $HARD_PATH/configs/model_configs/internvl.json \
    $HARD_PATH/configs/data_configs/General_benchmark/pope.json \
    $HARD_PATH/configs/train_configs/KeepLoRA/InternVL/${DATASET_ID}/eval/task${TASK_ID}.json

pip install -e .
bash scripts/MCITlib/Eval_benchmarks/eval_seed.sh \
    $HARD_PATH/configs/model_configs/internvl.json \
    $HARD_PATH/configs/data_configs/General_benchmark/seed.json \
    $HARD_PATH/configs/train_configs/KeepLoRA/InternVL/${DATASET_ID}/eval/task${TASK_ID}.json