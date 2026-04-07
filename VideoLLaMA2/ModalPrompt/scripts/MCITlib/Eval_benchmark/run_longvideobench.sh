MODEL_CONFIG=$1
DATA_CONFIG=$2
TRAIN_CONFIG=$3

read_config() {
    python3 -c "import json; print(json.load(open('$1'))['$2'])"
}

TASK="longvideobench"

MODELPATH=$(read_config "$TRAIN_CONFIG" model_path)
MODELBASE=$(read_config "$MODEL_CONFIG" model_name)
VIDEO_DIR=$(read_config "$DATA_CONFIG" video_dir)
SUBTITLE_DIR=$(read_config "$DATA_CONFIG" subtitle_dir)
QUESTION_FILE=$(read_config "$DATA_CONFIG" question_file)
output_dir=$(read_config "$TRAIN_CONFIG" result_path)
STAGE=$(read_config "$TRAIN_CONFIG" stage)
TEXT_TOWER=$(read_config "$TRAIN_CONFIG" text_tower)
PREFIX_LEN=$(read_config "$TRAIN_CONFIG" prefix_len)
CUR_TASK=$(read_config "$TRAIN_CONFIG" cur_task)
NUM_TASK=$(read_config "$TRAIN_CONFIG" num_tasks)
############################
# GPU & Chunk config
############################
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

GPUS_PER_TASK=1
CHUNKS=$((${#GPULIST[@]} / $GPUS_PER_TASK))


MODEL_NAME=$(basename "$MODELPATH")

ANSWER_DIR=${output_dir}/${TASK}/${STAGE}/answers
MERGE_FILE=${ANSWER_DIR}/merge.json

mkdir -p ${ANSWER_DIR}

############################
# Clean old results if needed
############################
if [ ! -f "$MERGE_FILE" ] || [ $(wc -l < "$MERGE_FILE") -eq 0 ]; then
    rm -f ${ANSWER_DIR}/*.json
fi

############################
# Inference
############################
if [ ! -f "$MERGE_FILE" ]; then
    for IDX in $(seq 0 $((CHUNKS-1))); do
        gpu_devices=$(IFS=,; echo "${GPULIST[*]:$(($IDX*$GPUS_PER_TASK)):$GPUS_PER_TASK}")

        TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${gpu_devices} \
        python3 videollama2/eval/inference_video_longvideobench.py \
            --model_path $MODELPATH \
            --model_base $MODELBASE \
            --text-tower $TEXT_TOWER\
            --prefix-len $PREFIX_LEN \
            --cur-task $CUR_TASK \
            --num-tasks $NUM_TASK \
            --video-folder ${VIDEO_DIR} \
            --subtitle-folder ${SUBTITLE_DIR} \
            --question-file ${QUESTION_FILE} \
            --answer-file ${ANSWER_DIR}/${CHUNKS}_${IDX}.json \
            --num-chunks ${CHUNKS} \
            --chunk-idx ${IDX} &
    done

    wait

    ############################
    # Merge results
    ############################
    > "$MERGE_FILE"
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ${ANSWER_DIR}/${CHUNKS}_${IDX}.json >> "$MERGE_FILE"
    done
fi
