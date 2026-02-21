#!/bin/bash

# ================= Parameters =================
BASE_DIR="./data"
GPU_ID="0"
STANCE="oppose"
BUDGET=10
TOPK=5
EMBEDDING_PREFIX="bge"
TARGET_MODEL="mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
# ==============================================

source $(dirname "$0")/topic.sh

echo "======================================================"
echo " STEP 4: Final Retrieval & Evaluation"
echo "======================================================"

execute_eval() {
    local category=$1
    local topic=$2

    echo "[Eval] Processing Category: $category | Topic: $topic"

    python src/eval/evaluate_retrieval.py \
        --category "$category" \
        --topic "$topic" \
        --base_dir "$BASE_DIR" \
        --stance "$STANCE" \
        --budget "$BUDGET" \
        --gpu_id "$GPU_ID" \
        --embedding_prefix "$EMBEDDING_PREFIX"

    python src/eval/evaluate_generation.py \
        --category "$category" \
        --topic "$topic" \
        --base_dir "$BASE_DIR" \
        --stance "$STANCE" \
        --budget "$BUDGET" \
        --gpu_id "$GPU_ID" \
        --target_model "$TARGET_MODEL" \
        --embedding_prefix "$EMBEDDING_PREFIX"
}

for topic in "${TOPICS_POLITICS[@]}"; do execute_eval "politics" "$topic"; done
for topic in "${TOPICS_SPORTS[@]}"; do execute_eval "sports" "$topic"; done
for topic in "${TOPICS_ENTERTAINMENT[@]}"; do execute_eval "entertainment" "$topic"; done
for topic in "${TOPICS_SOCIETY_BUSINESS[@]}"; do execute_eval "society_business" "$topic"; done

echo "======================================================"
echo " STEP 5: Generating Final Metric Tables"
echo "======================================================"

python src/eval/calculate_metrics.py \
    --base_dir "$BASE_DIR" \
    --budget "$BUDGET" \
    --topk "$TOPK" \
    --embed_prefix "$EMBEDDING_PREFIX" \
    --model_name "Llama-31"

echo "DiscourseFlip Pipeline Finished Successfully!"