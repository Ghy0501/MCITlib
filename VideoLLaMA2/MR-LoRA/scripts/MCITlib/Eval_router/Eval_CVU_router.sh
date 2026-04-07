#!/bin/bash

TASK_ID=$1
HARD_PATH=/your_path/MCITlib_v3

source /your_conda_path/miniconda3/etc/profile.d/conda.sh

get_task_config() {
    if [ "$TASK_ID" == "1" ]; then
        echo "$HARD_PATH/configs/train_configs/MR-LoRA/VideoLLaMA2/CL-VISTA/eval_router/task1.json"
    elif [ "$TASK_ID" == "2" ]; then
        echo "$HARD_PATH/configs/train_configs/MR-LoRA/VideoLLaMA2/CL-VISTA/eval_router/task2.json"
    elif [ "$TASK_ID" == "3" ]; then
        echo "$HARD_PATH/configs/train_configs/MR-LoRA/VideoLLaMA2/CL-VISTA/eval_router/task3.json"
    elif [ "$TASK_ID" == "4" ]; then
        echo "$HARD_PATH/configs/train_configs/MR-LoRA/VideoLLaMA2/CL-VISTA/eval_router/task4.json"
    elif [ "$TASK_ID" == "5" ]; then
        echo "$HARD_PATH/configs/train_configs/MR-LoRA/VideoLLaMA2/CL-VISTA/eval_router/task5.json"
    elif [ "$TASK_ID" == "6" ]; then
        echo "$HARD_PATH/configs/train_configs/MR-LoRA/VideoLLaMA2/CL-VISTA/eval_router/task6.json"
    elif [ "$TASK_ID" == "7" ]; then
        echo "$HARD_PATH/configs/train_configs/MR-LoRA/VideoLLaMA2/CL-VISTA/eval_router/task7.json"
    else
        echo "$HARD_PATH/configs/train_configs/MR-LoRA/VideoLLaMA2/CL-VISTA/eval_router/task8.json"
    fi
}

TASK_CONFIG=$(get_task_config)

echo "=============================================="
echo "CVU router evaluation (switches conda envs automatically)"
echo "Current user: $(whoami)"
echo "Conda root: /your_conda_path/miniconda3"
echo "=============================================="

echo "Phase 1: Activate 'videollama2' and run all Eval_router run_qa scripts"
echo "=============================================="

conda activate videollama2
if [ $? -ne 0 ]; then
    echo "Error: failed to activate conda env 'videollama2'"
    echo "Available environments:"
    conda env list
    exit 1
fi

echo "Active env: $(conda info --envs | grep '*' | awk '{print $1}')"

if [ "$TASK_ID" == "1" ]; then
    bash scripts/MCITlib/Eval_router/run_qa_counting.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/counting.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_space.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/space.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_traffic.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/traffic.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_movie.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/movie.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_gui.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/gui.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_science.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/science.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_sports.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/sports.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_star.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/star.json $TASK_CONFIG
    
elif [ "$TASK_ID" == "2" ]; then
    bash scripts/MCITlib/Eval_router/run_qa_counting.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/counting.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_space.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/space.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_traffic.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/traffic.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_movie.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/movie.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_gui.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/gui.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_science.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/science.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_sports.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/sports.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_star.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/star.json $TASK_CONFIG
    
elif [ "$TASK_ID" == "3" ]; then
    bash scripts/MCITlib/Eval_router/run_qa_counting.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/counting.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_space.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/space.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_traffic.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/traffic.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_movie.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/movie.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_gui.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/gui.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_science.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/science.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_sports.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/sports.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_star.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/star.json $TASK_CONFIG
    
elif [ "$TASK_ID" == "4" ]; then
    bash scripts/MCITlib/Eval_router/run_qa_counting.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/counting.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_space.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/space.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_traffic.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/traffic.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_movie.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/movie.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_gui.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/gui.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_science.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/science.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_sports.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/sports.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_star.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/star.json $TASK_CONFIG
    
elif [ "$TASK_ID" == "5" ]; then
    bash scripts/MCITlib/Eval_router/run_qa_counting.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/counting.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_space.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/space.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_traffic.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/traffic.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_movie.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/movie.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_gui.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/gui.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_science.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/science.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_sports.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/sports.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_star.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/star.json $TASK_CONFIG

elif [ "$TASK_ID" == "6" ]; then
    bash scripts/MCITlib/Eval_router/run_qa_counting.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/counting.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_space.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/space.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_traffic.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/traffic.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_movie.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/movie.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_gui.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/gui.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_science.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/science.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_sports.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/sports.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_star.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/star.json $TASK_CONFIG

elif [ "$TASK_ID" == "7" ]; then
    bash scripts/MCITlib/Eval_router/run_qa_counting.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/counting.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_space.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/space.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_traffic.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/traffic.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_movie.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/movie.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_gui.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/gui.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_science.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/science.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_sports.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/sports.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_star.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/star.json $TASK_CONFIG

else
    bash scripts/MCITlib/Eval_router/run_qa_counting.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/counting.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_space.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/space.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_traffic.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/traffic.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_movie.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/movie.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_gui.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/gui.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_science.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/science.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_sports.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/sports.json $TASK_CONFIG
    bash scripts/MCITlib/Eval_router/run_qa_star.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/star.json $TASK_CONFIG

fi