#!/bin/bash

############################
# Args
############################
MODEL_CONFIG=$1
DATA_CONFIG=$2
TRAIN_CONFIG=$3

############################
# Utils
############################
read_config() {
    python3 - <<EOF
import json
print(json.load(open("$1"))["$2"])
EOF
}

TASK="mvbench"

############################
# Read configs
############################
MODELPATH=$(read_config "$TRAIN_CONFIG" model_path)
MODELBASE=$(read_config "$MODEL_CONFIG" model_name)
MVBENCH_JSON_DIR=$(read_config "$DATA_CONFIG" json_dir)
OUTPUT_ROOT=$(read_config "$TRAIN_CONFIG" result_path)
STAGE=$(read_config "$TRAIN_CONFIG" stage)
VIDEO_DIR=$(read_config "$DATA_CONFIG" video_dir)
############################
# GPU & Chunk config
############################
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

GPUS_PER_TASK=1
CHUNKS=$((${#GPULIST[@]} / $GPUS_PER_TASK))

############################
# Paths
############################
MODEL_NAME=$(basename "$MODELPATH")
OUTPUT_DIR=${OUTPUT_ROOT}/${TASK}/${STAGE}/${MODEL_NAME}

mkdir -p ${OUTPUT_DIR}

############################
# Merge results
############################
MERGE_FILE=${OUTPUT_DIR}/videollava_mvbench_merge.jsonl
SUMMARY_FILE=${OUTPUT_DIR}/videollava_mvbench_summary_merge.json


> ${MERGE_FILE}

for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${OUTPUT_DIR}/videollava_mvbench_${CHUNKS}_${IDX}.jsonl >> ${MERGE_FILE}
done

echo "[INFO] Merge done: ${MERGE_FILE}"

############################
# (Optional) Merge summaries
############################
python3 - <<EOF
import json, glob, os
summaries = glob.glob("${OUTPUT_DIR}/videollava_mvbench_*_summary.json")
if summaries:
    print("[INFO] Individual summaries:")
    for s in summaries:
        print(s)
EOF

echo "[DONE] MVBench evaluation finished."
