#!/bin/bash

TASK_ID=$1
HARD_PATH=/your_path/MCITlib_v3

# Source conda shell hook
source /your_conda_path/miniconda3/etc/profile.d/conda.sh

# Map TASK_ID to eval task JSON
get_task_config() {
    if [ "$TASK_ID" == "1" ]; then
        echo "$HARD_PATH/configs/train_configs/Replay/Video-LLaVA/CL-VISTA/eval/task1.json"
    elif [ "$TASK_ID" == "2" ]; then
        echo "$HARD_PATH/configs/train_configs/Replay/Video-LLaVA/CL-VISTA/eval/task2.json"
    elif [ "$TASK_ID" == "3" ]; then
        echo "$HARD_PATH/configs/train_configs/Replay/Video-LLaVA/CL-VISTA/eval/task3.json"
    elif [ "$TASK_ID" == "4" ]; then
        echo "$HARD_PATH/configs/train_configs/Replay/Video-LLaVA/CL-VISTA/eval/task4.json"
    elif [ "$TASK_ID" == "5" ]; then
        echo "$HARD_PATH/configs/train_configs/Replay/Video-LLaVA/CL-VISTA/eval/task5.json"
    elif [ "$TASK_ID" == "6" ]; then
        echo "$HARD_PATH/configs/train_configs/Replay/Video-LLaVA/CL-VISTA/eval/task6.json"
    elif [ "$TASK_ID" == "7" ]; then
        echo "$HARD_PATH/configs/train_configs/Replay/Video-LLaVA/CL-VISTA/eval/task7.json"
    else
        echo "$HARD_PATH/configs/train_configs/Replay/Video-LLaVA/CL-VISTA/eval/task8.json"
    fi
}

TASK_CONFIG=$(get_task_config)

echo "=============================================="
echo "CVU evaluation (switches conda envs automatically)"
echo "Current user: $(whoami)"
echo "Conda root: /your_conda_path/miniconda3"
echo "=============================================="

echo "Phase 1: Activate 'videollava' and run all run_qa scripts"
echo "=============================================="

# Activate inference env
conda activate videollava
if [ $? -ne 0 ]; then
    echo "Error: failed to activate conda env 'videollava'"
    echo "Available environments:"
    conda env list
    exit 1
fi

echo "Active env: $(conda info --envs | grep '*' | awk '{print $1}')"

# Run all run_qa scripts
if [ "$TASK_ID" == "1" ]; then
    bash scripts/MCITlib/Eval/run_qa_counting.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/counting.json $TASK_CONFIG
    
elif [ "$TASK_ID" == "2" ]; then
    bash scripts/MCITlib/Eval/run_qa_counting.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/counting.json $TASK_CONFIG
    bash scripts/MCITlib/Eval/run_qa_space.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/space.json $TASK_CONFIG
    
elif [ "$TASK_ID" == "3" ]; then
    bash scripts/MCITlib/Eval/run_qa_counting.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/counting.json $TASK_CONFIG
    bash scripts/MCITlib/Eval/run_qa_space.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/space.json $TASK_CONFIG
    bash scripts/MCITlib/Eval/run_qa_traffic.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/traffic.json $TASK_CONFIG
    
elif [ "$TASK_ID" == "4" ]; then
    bash scripts/MCITlib/Eval/run_qa_counting.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/counting.json $TASK_CONFIG
    bash scripts/MCITlib/Eval/run_qa_space.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/space.json $TASK_CONFIG
    bash scripts/MCITlib/Eval/run_qa_traffic.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/traffic.json $TASK_CONFIG
    bash scripts/MCITlib/Eval/run_qa_movie.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/movie.json $TASK_CONFIG
    
elif [ "$TASK_ID" == "5" ]; then
    bash scripts/MCITlib/Eval/run_qa_counting.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/counting.json $TASK_CONFIG
    bash scripts/MCITlib/Eval/run_qa_space.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/space.json $TASK_CONFIG
    bash scripts/MCITlib/Eval/run_qa_traffic.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/traffic.json $TASK_CONFIG
    bash scripts/MCITlib/Eval/run_qa_movie.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/movie.json $TASK_CONFIG
    bash scripts/MCITlib/Eval/run_qa_gui.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/gui.json $TASK_CONFIG

elif [ "$TASK_ID" == "6" ]; then
    bash scripts/MCITlib/Eval/run_qa_counting.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/counting.json $TASK_CONFIG
    bash scripts/MCITlib/Eval/run_qa_space.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/space.json $TASK_CONFIG
    bash scripts/MCITlib/Eval/run_qa_traffic.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/traffic.json $TASK_CONFIG
    bash scripts/MCITlib/Eval/run_qa_movie.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/movie.json $TASK_CONFIG
    bash scripts/MCITlib/Eval/run_qa_gui.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/gui.json $TASK_CONFIG
    bash scripts/MCITlib/Eval/run_qa_science.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/science.json $TASK_CONFIG

elif [ "$TASK_ID" == "7" ]; then
    bash scripts/MCITlib/Eval/run_qa_counting.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/counting.json $TASK_CONFIG
    bash scripts/MCITlib/Eval/run_qa_space.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/space.json $TASK_CONFIG
    bash scripts/MCITlib/Eval/run_qa_traffic.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/traffic.json $TASK_CONFIG
    bash scripts/MCITlib/Eval/run_qa_movie.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/movie.json $TASK_CONFIG
    bash scripts/MCITlib/Eval/run_qa_gui.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/gui.json $TASK_CONFIG
    bash scripts/MCITlib/Eval/run_qa_science.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/science.json $TASK_CONFIG
    bash scripts/MCITlib/Eval/run_qa_sports.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/sports.json $TASK_CONFIG

else
    bash scripts/MCITlib/Eval/run_qa_counting.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/counting.json $TASK_CONFIG
    bash scripts/MCITlib/Eval/run_qa_space.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/space.json $TASK_CONFIG
    bash scripts/MCITlib/Eval/run_qa_traffic.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/traffic.json $TASK_CONFIG
    bash scripts/MCITlib/Eval/run_qa_movie.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/movie.json $TASK_CONFIG
    bash scripts/MCITlib/Eval/run_qa_gui.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/gui.json $TASK_CONFIG
    bash scripts/MCITlib/Eval/run_qa_science.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/science.json $TASK_CONFIG
    bash scripts/MCITlib/Eval/run_qa_sports.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/sports.json $TASK_CONFIG
    bash scripts/MCITlib/Eval/run_qa_star.sh $HARD_PATH/configs/model_configs/videollava.json $HARD_PATH/configs/data_configs/CL-VISTA/star.json $TASK_CONFIG

fi

echo ""
echo "=============================================="
echo "All run_qa scripts finished."
echo "Switching to eval environment..."
echo "=============================================="

conda deactivate

# Scoring / metrics env (change if yours differs)
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

# Run all eval_qa scripts
if [ "$TASK_ID" == "1" ]; then
    bash scripts/MCITlib/Eval/eval_qa_counting.sh $TASK_CONFIG
    
elif [ "$TASK_ID" == "2" ]; then
    bash scripts/MCITlib/Eval/eval_qa_counting.sh $TASK_CONFIG
    bash scripts/MCITlib/Eval/eval_qa_space.sh $TASK_CONFIG
    
elif [ "$TASK_ID" == "3" ]; then
    bash scripts/MCITlib/Eval/eval_qa_counting.sh $TASK_CONFIG
    bash scripts/MCITlib/Eval/eval_qa_space.sh $TASK_CONFIG
    bash scripts/MCITlib/Eval/eval_qa_traffic.sh $TASK_CONFIG
    
elif [ "$TASK_ID" == "4" ]; then
    bash scripts/MCITlib/Eval/eval_qa_counting.sh $TASK_CONFIG
    bash scripts/MCITlib/Eval/eval_qa_space.sh $TASK_CONFIG
    bash scripts/MCITlib/Eval/eval_qa_traffic.sh $TASK_CONFIG
    bash scripts/MCITlib/Eval/eval_qa_movie.sh $TASK_CONFIG
    
elif [ "$TASK_ID" == "5" ]; then
    bash scripts/MCITlib/Eval/eval_qa_counting.sh $TASK_CONFIG
    bash scripts/MCITlib/Eval/eval_qa_space.sh $TASK_CONFIG
    bash scripts/MCITlib/Eval/eval_qa_traffic.sh $TASK_CONFIG
    bash scripts/MCITlib/Eval/eval_qa_movie.sh $TASK_CONFIG
    bash scripts/MCITlib/Eval/eval_qa_gui.sh $TASK_CONFIG

elif [ "$TASK_ID" == "6" ]; then
    bash scripts/MCITlib/Eval/eval_qa_counting.sh $TASK_CONFIG
    bash scripts/MCITlib/Eval/eval_qa_space.sh $TASK_CONFIG
    bash scripts/MCITlib/Eval/eval_qa_traffic.sh $TASK_CONFIG
    bash scripts/MCITlib/Eval/eval_qa_movie.sh $TASK_CONFIG
    bash scripts/MCITlib/Eval/eval_qa_gui.sh $TASK_CONFIG
    bash scripts/MCITlib/Eval/eval_qa_science.sh $TASK_CONFIG

elif [ "$TASK_ID" == "7" ]; then
    bash scripts/MCITlib/Eval/eval_qa_counting.sh $TASK_CONFIG
    bash scripts/MCITlib/Eval/eval_qa_space.sh $TASK_CONFIG
    bash scripts/MCITlib/Eval/eval_qa_traffic.sh $TASK_CONFIG
    bash scripts/MCITlib/Eval/eval_qa_movie.sh $TASK_CONFIG
    bash scripts/MCITlib/Eval/eval_qa_gui.sh $TASK_CONFIG
    bash scripts/MCITlib/Eval/eval_qa_science.sh $TASK_CONFIG
    bash scripts/MCITlib/Eval/eval_qa_sports.sh $TASK_CONFIG

else
    bash scripts/MCITlib/Eval/eval_qa_counting.sh $TASK_CONFIG
    bash scripts/MCITlib/Eval/eval_qa_space.sh $TASK_CONFIG
    bash scripts/MCITlib/Eval/eval_qa_traffic.sh $TASK_CONFIG
    bash scripts/MCITlib/Eval/eval_qa_movie.sh $TASK_CONFIG
    bash scripts/MCITlib/Eval/eval_qa_gui.sh $TASK_CONFIG
    bash scripts/MCITlib/Eval/eval_qa_science.sh $TASK_CONFIG
    bash scripts/MCITlib/Eval/eval_qa_sports.sh $TASK_CONFIG
    bash scripts/MCITlib/Eval/eval_qa_star.sh $TASK_CONFIG

fi

echo "=============================================="
echo "Evaluation pipeline finished."
echo "Final active env: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "=============================================="