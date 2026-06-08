#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${CHECKPOINT:-/scratch/hateful_meme_semantic_retrieval/outputs/runai_qwen_vl_fields_image/best_model.pt}"
CUES_JSONL="${CUES_JSONL:-/scratch/hateful_meme_semantic_retrieval/data/facebook_semantic_cues.jsonl}"
IMAGE_DIR="${IMAGE_DIR:-/scratch/hateful_meme_semantic_retrieval/data/images}"
DATASET_NAME="${DATASET_NAME:-cs5242-hateful-memes/hateful-memes-data}"
CACHE_DIR="${CACHE_DIR:-/scratch/hateful_meme_semantic_retrieval/data/hf_cache}"
OUTPUT_JSON="${OUTPUT_JSON:-/scratch/hateful_meme_semantic_retrieval/outputs/runai_qwen_vl_fields_image/test_seen_unseen_eval.json}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
EVAL_SPLITS="${EVAL_SPLITS:-test_seen test_unseen}"
BATCH_SIZE="${BATCH_SIZE:-64}"
ENCODER_BATCH_SIZE="${ENCODER_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"

read -r -a EVAL_SPLIT_ARGS <<< "${EVAL_SPLITS}"

python3 -u -m semantic_rahmd.eval_checkpoint \
  --checkpoint "${CHECKPOINT}" \
  --cues-jsonl "${CUES_JSONL}" \
  --image-dir "${IMAGE_DIR}" \
  --dataset-name "${DATASET_NAME}" \
  --cache-dir "${CACHE_DIR}" \
  --train-split "${TRAIN_SPLIT}" \
  --eval-split "${EVAL_SPLIT_ARGS[@]}" \
  --output-json "${OUTPUT_JSON}" \
  --batch-size "${BATCH_SIZE}" \
  --encoder-batch-size "${ENCODER_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  "$@"
