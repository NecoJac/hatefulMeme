#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

python3 -m semantic_rahmd.train \
  --cues-jsonl "${CUES_JSONL:-data/facebook_semantic_cues.jsonl}" \
  --image-dir "${IMAGE_DIR:-data/images}" \
  --dataset-name "${DATASET_NAME:-cs5242-hateful-memes/hateful-memes-data}" \
  --cache-dir "${CACHE_DIR:-data/hf_cache}" \
  --output-dir "${OUTPUT_DIR:-outputs/frozen_llm}" \
  --train-split "${TRAIN_SPLIT:-train}" \
  --dev-split "${DEV_SPLIT:-dev_seen}" \
  --test-split ${TEST_SPLITS:-test_seen test_unseen} \
  --extra-eval-split ${EXTRA_EVAL_SPLITS:-dev_unseen} \
  --encoder-backend hf \
  --llm-model "${LLM_MODEL:-distilbert-base-uncased}" \
  --proj-dim "${PROJ_DIM:-768}" \
  --num-heads "${NUM_HEADS:-8}" \
  --num-layers "${NUM_LAYERS:-2}" \
  --epochs "${EPOCHS:-30}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --lr "${LR:-0.0001}" \
  --weight-decay "${WEIGHT_DECAY:-0.01}" \
  --contrastive-weight "${CONTRASTIVE_WEIGHT:-1.0}" \
  --temperature "${TEMPERATURE:-1.0}" \
  --topk "${TOPK:-20}" \
  --grad-clip "${GRAD_CLIP:-1.0}" \
  --seed "${SEED:-1}" \
  --save-embeddings \
  "$@"
