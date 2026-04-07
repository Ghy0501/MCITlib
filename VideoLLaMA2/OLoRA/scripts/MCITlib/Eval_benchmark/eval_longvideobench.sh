TRAIN_CONFIG=$1

read_config() {
    python3 -c "import json; print(json.load(open('$1'))['$2'])"
}

TASK="longvideobench"

output_dir=$(read_config "$TRAIN_CONFIG" result_path)
STAGE=$(read_config "$TRAIN_CONFIG" stage)


ANSWER_DIR=${output_dir}/${TASK}/${STAGE}/answers
MERGE_FILE=${ANSWER_DIR}/merge.json

############################
# Evaluation
############################
python3 videollama2/eval/eval_longvideobench.py \
    --pred_path ${MERGE_FILE}
