#!/bin/bash

############################
# 显示 MVBench summary 文件内容
############################

# 输出目录，根据原来的脚本逻辑拼接

TRAIN_CONFIG=$1

# Utils: 从 JSON 配置读取字段
read_config() {
    python3 - <<EOF
import json
print(json.load(open("$1"))["$2"])
EOF
}

TASK="mvbench"

# 读取配置
MODELPATH=$(read_config "$TRAIN_CONFIG" model_path)
OUTPUT_ROOT=$(read_config "$TRAIN_CONFIG" result_path)
STAGE=$(read_config "$TRAIN_CONFIG" stage)

MODEL_NAME=$(basename "$MODELPATH")
OUTPUT_DIR=${OUTPUT_ROOT}/${TASK}/${STAGE}/${MODEL_NAME}

SUMMARY_FILE=${OUTPUT_DIR}/videollava_mvbench_1_0_summary.json

# 显示 summary 内容
if [ -f "$SUMMARY_FILE" ]; then
    cat "$SUMMARY_FILE"
else
    echo "[ERROR] Summary file not found: $SUMMARY_FILE"
fi