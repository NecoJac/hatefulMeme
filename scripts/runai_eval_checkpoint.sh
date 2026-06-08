#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-registry.rcp.epfl.ch/ee559/environment-with-packages:latest}"
PVC="${PVC:-course-ee-559-scratch-g36}"
UID_TO_RUN="${UID_TO_RUN:-287685}"

REPO_DIR="${REPO_DIR:-/scratch/hateful_meme_semantic_retrieval}"
CHECKPOINT="${CHECKPOINT:-/scratch/hateful_meme_semantic_retrieval/outputs/runai_qwen_vl_fields_image/best_model.pt}"
CUES_JSONL="${CUES_JSONL:-/scratch/hateful_meme_semantic_retrieval/data/facebook_semantic_cues.jsonl}"
IMAGE_DIR="${IMAGE_DIR:-/scratch/hateful_meme_semantic_retrieval/data/images}"
DATASET_NAME="${DATASET_NAME:-cs5242-hateful-memes/hateful-memes-data}"
CACHE_DIR="${CACHE_DIR:-/scratch/hateful_meme_semantic_retrieval/data/hf_cache}"
OUTPUT_JSON="${OUTPUT_JSON:-/scratch/hateful_meme_semantic_retrieval/outputs/runai_qwen_vl_fields_image/test_seen_unseen_eval.json}"
VENDOR_PYTHONPATH="${VENDOR_PYTHONPATH:-${REPO_DIR}/vendor/python}"
PYTHONPATH_TO_RUN="${PYTHONPATH_TO_RUN:-${VENDOR_PYTHONPATH}:${REPO_DIR}/src}"

EVAL_SPLITS="${EVAL_SPLITS:-test_seen test_unseen}"
BATCH_SIZE="${BATCH_SIZE:-64}"
ENCODER_BATCH_SIZE="${ENCODER_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
GPU="${GPU:-1}"

CMD=(
  bash "${REPO_DIR}/scripts/eval_checkpoint.sh"
)

ENV_ARGS=(
  --environment "PYTHONPATH=${PYTHONPATH_TO_RUN}"
  --environment "PYTHONUNBUFFERED=1"
  --environment "CHECKPOINT=${CHECKPOINT}"
  --environment "CUES_JSONL=${CUES_JSONL}"
  --environment "IMAGE_DIR=${IMAGE_DIR}"
  --environment "DATASET_NAME=${DATASET_NAME}"
  --environment "CACHE_DIR=${CACHE_DIR}"
  --environment "OUTPUT_JSON=${OUTPUT_JSON}"
  --environment "EVAL_SPLITS=${EVAL_SPLITS}"
  --environment "BATCH_SIZE=${BATCH_SIZE}"
  --environment "ENCODER_BATCH_SIZE=${ENCODER_BATCH_SIZE}"
  --environment "NUM_WORKERS=${NUM_WORKERS}"
)
if [[ -n "${HF_TOKEN:-}" ]]; then
  ENV_ARGS+=(--environment "HF_TOKEN=${HF_TOKEN}")
fi

printf '[local] command:'
printf ' %q' "${CMD[@]}"
printf '\n'

runai submit --run-as-uid "${UID_TO_RUN}" \
  --image "${IMAGE}" \
  --gpu "${GPU}" \
  --existing-pvc "claimname=${PVC},path=/scratch" \
  "${ENV_ARGS[@]}" \
  --command -- "${CMD[@]}"
