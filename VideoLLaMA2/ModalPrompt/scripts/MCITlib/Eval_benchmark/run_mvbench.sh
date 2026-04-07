MODEL_CONFIG=$1
DATA_CONFIG=$2
TRAIN_CONFIG=$3

read_config() {
    python3 -c "import json; print(json.load(open('$1'))['$2'])"
}

TASK="mvbench"

MODELPATH=$(read_config "$TRAIN_CONFIG" model_path)
MODELBASE=$(read_config "$MODEL_CONFIG" model_name)
VIDEO_DIR=$(read_config "$DATA_CONFIG" video_dir)
QUESTION_FILE=$(read_config "$DATA_CONFIG" json_dir)
output_dir=$(read_config "$TRAIN_CONFIG" result_path)
STAGE=$(read_config "$TRAIN_CONFIG" stage)
TEXT_TOWER=$(read_config "$TRAIN_CONFIG" text_tower)
PREFIX_LEN=$(read_config "$TRAIN_CONFIG" prefix_len)
CUR_TASK=$(read_config "$TRAIN_CONFIG" cur_task)
NUM_TASK=$(read_config "$TRAIN_CONFIG" num_tasks)
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

# divide data via the number of GPUs per task
GPUS_PER_TASK=1
CHUNKS=$((${#GPULIST[@]}/$GPUS_PER_TASK))

output_file=${output_dir}/${TASK}/${STAGE}/answers/merge.json



if [ ! -f "$output_file" ]; then
    for IDX in $(seq 0 $((CHUNKS-1))); do
        gpu_devices=$(IFS=,; echo "${GPULIST[*]:$(($IDX*$GPUS_PER_TASK)):$GPUS_PER_TASK}")
        TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${gpu_devices} python3 videollama2/eval/inference_video_mcqa_mvbench.py \
            --model_path $MODELPATH \
            --model_base $MODELBASE \
            --text-tower $TEXT_TOWER\
            --prefix-len $PREFIX_LEN \
            --cur-task $CUR_TASK \
            --num-tasks $NUM_TASK \
            --video-folder $VIDEO_DIR \
            --question-file $QUESTION_FILE \
            --answer-file ${output_dir}/${TASK}/${STAGE}/answers/${CHUNKS}_${IDX}.json \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX &
    done

    wait

    # Clear out the output file if it exists.
    > "$output_file"

    # Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ${output_dir}/${TASK}/${STAGE}/answers/${CHUNKS}_${IDX}.json >> "$output_file"
    done
fi
