#!/bin/bash

HARD_PATH=/your_path/MCITlib_v3
CONFIG_FILE="$HARD_PATH/configs/train_configs/LoRA-FT/VideoLLaMA2/CL-VISTA/eval/joint.json"

source /your_conda_path/miniconda3/etc/profile.d/conda.sh

echo "=============================================="
echo "当前用户: $(whoami)"
echo "配置文件: $CONFIG_FILE"
echo "=============================================="

echo "阶段1: 激活 'videollama2' 环境并运行所有 run_qa 脚本"
echo "=============================================="

conda activate videollama2
if [ $? -ne 0 ]; then
    echo "错误: 无法激活 conda 环境 'videollama2'"
    echo "可用环境列表:"
    conda env list
    exit 1
fi

echo "当前环境: $(conda info --envs | grep '*' | awk '{print $1}')"

bash scripts/MCITlib/Eval/run_qa_counting.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/counting.json $CONFIG_FILE
bash scripts/MCITlib/Eval/run_qa_space.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/space.json $CONFIG_FILE
bash scripts/MCITlib/Eval/run_qa_traffic.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/traffic.json $CONFIG_FILE
bash scripts/MCITlib/Eval/run_qa_movie.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/movie.json $CONFIG_FILE
bash scripts/MCITlib/Eval/run_qa_gui.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/gui.json $CONFIG_FILE
bash scripts/MCITlib/Eval/run_qa_science.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/science.json $CONFIG_FILE
bash scripts/MCITlib/Eval/run_qa_sports.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/sports.json $CONFIG_FILE
bash scripts/MCITlib/Eval/run_qa_star.sh $HARD_PATH/configs/model_configs/videollama2.json $HARD_PATH/configs/data_configs/CL-VISTA/star.json $CONFIG_FILE

echo ""
echo "=============================================="
echo "所有 run_qa 脚本执行完成！"
echo "切换到 eval 环境..."
echo "=============================================="

conda deactivate

EVAL_ENV="transformers"
conda activate $EVAL_ENV
if [ $? -ne 0 ]; then
    echo "警告: 无法激活 eval 环境 '$EVAL_ENV'"
    echo "可用环境列表:"
    conda env list
    echo "请输入正确的 eval 环境名称: "
    read EVAL_ENV
    conda activate $EVAL_ENV
    if [ $? -ne 0 ]; then
        echo "错误: 仍然无法激活环境 '$EVAL_ENV'"
        exit 1
    fi
fi

echo "当前环境: $(conda info --envs | grep '*' | awk '{print $1}')"

echo "=============================================="
echo "阶段2: 在 '$EVAL_ENV' 环境中运行所有 eval_qa 脚本"
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
echo "所有6个数据集的联合评估完成！"
echo "最终环境: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "=============================================="