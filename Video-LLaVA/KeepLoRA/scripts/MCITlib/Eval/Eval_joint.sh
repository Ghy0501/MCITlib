#!/bin/bash

HARD_PATH=/your_path/MCITlib_v3
CONFIG_FILE="$HARD_PATH/configs/train_configs/KeepLoRA/Video-LLaVA/CL-VISTA/eval/joint.json"

source /your_conda_path/miniconda3/etc/profile.d/conda.sh

echo "=============================================="
echo "Joint evaluation script - all 6 datasets"
echo "Current user: $(whoami)"
echo "Config file: $CONFIG_FILE"
echo "=============================================="

echo "Phase 1: Activate 'videollava' and run all run_qa scripts"
echo "=============================================="

conda activate videollava
if [ $? -ne 0 ]; then
    echo "Error: failed to activate conda env 'videollava'"
    echo "Available environments:"
    conda env list
    exit 1
fi

echo "Active env: $(conda info --envs | grep '*' | awk '{print $1}')"

bash scripts/MCITlib/Eval/run_qa_counting.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/counting.json $CONFIG_FILE
bash scripts/MCITlib/Eval/run_qa_space.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/space.json $CONFIG_FILE
bash scripts/MCITlib/Eval/run_qa_traffic.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/traffic.json $CONFIG_FILE
bash scripts/MCITlib/Eval/run_qa_movie.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/movie.json $CONFIG_FILE
bash scripts/MCITlib/Eval/run_qa_gui.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/gui.json $CONFIG_FILE
bash scripts/MCITlib/Eval/run_qa_science.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/science.json $CONFIG_FILE
bash scripts/MCITlib/Eval/run_qa_sports.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/sports.json $CONFIG_FILE
bash scripts/MCITlib/Eval/run_qa_star.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/star.json $CONFIG_FILE

echo ""
echo "=============================================="
echo "All run_qa scripts finished."
echo "Switching to eval environment..."
echo "=============================================="

conda deactivate

EVAL_ENV="transformers"
conda activate $EVAL_ENV
if [ $? -ne 0 ]; then
    echo "Warning: failed to activate eval env '$EVAL_ENV'"
    echo "Available environments:"
    conda env list
    echo "Enter the correct eval env name: "
    read EVAL_ENV
    conda activate $EVAL_ENV
    if [ $? -ne 0 ]; then
        echo "Error: still failed to activate env '$EVAL_ENV'"
        exit 1
    fi
fi

echo "Active env: $(conda info --envs | grep '*' | awk '{print $1}')"

echo "=============================================="
echo "Phase 2: Run all eval_qa scripts in '$EVAL_ENV'"
echo "=============================================="

bash scripts/MCITlib/Eval/eval_qa_counting.sh $CONFIG_FILE
bash scripts/MCITlib/Eval/eval_qa_space.sh $CONFIG_FILE
bash scripts/MCITlib/Eval/eval_qa_traffic.sh $CONFIG_FILE
bash scripts/MCITlib/Eval/eval_qa_movie.sh $CONFIG_FILE
bash scripts/MCITlib/Eval/eval_qa_gui.sh $CONFIG_FILE
bash scripts/MCITlib/Eval/eval_qa_science.sh $CONFIG_FILE
bash scripts/MCITlib/Eval/eval_qa_sports.sh $CONFIG_FILE
bash scripts/MCITlib/Eval/eval_qa_star.sh $CONFIG_FILE

echo "=============================================="
echo "Joint evaluation finished."
echo "Final active env: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "=============================================="