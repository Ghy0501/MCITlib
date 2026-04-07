#!/bin/bash

############################
# Print MVBench summary JSON
############################

# Output dir: same layout as the generation scripts (task / stage / model name)

TRAIN_CONFIG=$1

# Utils: read one field from a JSON config file
read_config() {
    python3 - <<EOF
import json
print(json.load(open("$1"))["$2"])
EOF
}

TASK="mvbench"

# Load paths from config
MODELPATH=$(read_config "$TRAIN_CONFIG" model_path)
OUTPUT_ROOT=$(read_config "$TRAIN_CONFIG" result_path)
STAGE=$(read_config "$TRAIN_CONFIG" stage)

MODEL_NAME=$(basename "$MODELPATH")
OUTPUT_DIR=${OUTPUT_ROOT}/${TASK}/${STAGE}/${MODEL_NAME}

SUMMARY_FILE=${OUTPUT_DIR}/videollava_mvbench_1_0_summary.json

# Print summary if present
if [ -f "$SUMMARY_FILE" ]; then
    cat "$SUMMARY_FILE"
else
    echo "[ERROR] Summary file not found: $SUMMARY_FILE"
fi