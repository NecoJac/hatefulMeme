#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-registry.rcp.epfl.ch/ee559/environment-with-packages:latest}"
PVC="${PVC:-course-ee-559-scratch-g36}"
UID_TO_RUN="${UID_TO_RUN:-287685}"
REPO_DIR="${REPO_DIR:-/scratch/hateful_meme_semantic_retrieval}"
VENDOR_PYTHONPATH="${VENDOR_PYTHONPATH:-${REPO_DIR}/vendor/python}"
PYTHONPATH_TO_RUN="${PYTHONPATH_TO_RUN:-${VENDOR_PYTHONPATH}:${REPO_DIR}/src}"

SCRIPT="${SCRIPT:-${REPO_DIR}/scripts/extract_facebook_semantic_cues.py}"
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
CACHE_DIR="${CACHE_DIR:-${REPO_DIR}/data/hf_cache}"
OUTPUT="${OUTPUT:-${REPO_DIR}/data/facebook_semantic_cues.jsonl}"
SPLITS="${SPLITS:-train dev_seen dev_unseen test_seen test_unseen}"
LIMIT="${LIMIT:-}"
RESUME="${RESUME:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.0}"
GPU="${GPU:-2}"

read -r -a SPLIT_ARGS <<< "${SPLITS}"

CMD=(
  python3 "${SCRIPT}"
  --model "${MODEL}"
  --cache-dir "${CACHE_DIR}"
  --splits "${SPLIT_ARGS[@]}"
  --output "${OUTPUT}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
)

if [[ -n "${LIMIT}" ]]; then
  CMD+=(--limit "${LIMIT}")
fi
if [[ "${RESUME}" == "1" || "${RESUME}" == "true" || "${RESUME}" == "yes" ]]; then
  CMD+=(--resume)
fi
CMD+=("$@")

ENV_ARGS=(
  --environment "PYTHONPATH=${PYTHONPATH_TO_RUN}"
  --environment "PYTHONUNBUFFERED=1"
)
if [[ -n "${HF_TOKEN:-}" ]]; then
  ENV_ARGS+=(--environment "HF_TOKEN=${HF_TOKEN}")
fi

runai submit --run-as-uid "${UID_TO_RUN}"   --image "${IMAGE}"   --gpu "${GPU}"   --existing-pvc "claimname=${PVC},path=/scratch"   "${ENV_ARGS[@]}"   --command -- "${CMD[@]}"
