#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-registry.rcp.epfl.ch/ee559/environment-with-packages:latest}"
PVC="${PVC:-course-ee-559-scratch-g36}"
UID_TO_RUN="${UID_TO_RUN:-287685}"

REPO_DIR="${REPO_DIR:-/scratch/hateful_meme_semantic_retrieval}"
SCRIPT_MODULE="${SCRIPT_MODULE:-semantic_rahmd.train}"
CUES_JSONL="${CUES_JSONL:-/scratch/hateful_meme_semantic_retrieval/data/facebook_semantic_cues.jsonl}"
IMAGE_DIR="${IMAGE_DIR:-/scratch/hateful_meme_semantic_retrieval/data/images}"
DATASET_NAME="${DATASET_NAME:-cs5242-hateful-memes/hateful-memes-data}"
CACHE_DIR="${CACHE_DIR:-/scratch/hateful_meme_semantic_retrieval/data/hf_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/hateful_meme_semantic_retrieval/outputs/runai_qwen_vl_fields_image}"
VENDOR_PYTHONPATH="${VENDOR_PYTHONPATH:-${REPO_DIR}/vendor/python}"
PYTHONPATH_TO_RUN="${PYTHONPATH_TO_RUN:-${VENDOR_PYTHONPATH}:${REPO_DIR}/src}"

TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
DEV_SPLIT="${DEV_SPLIT:-dev_seen}"
TEST_SPLITS="${TEST_SPLITS:-test_seen test_unseen}"
EXTRA_EVAL_SPLITS="${EXTRA_EVAL_SPLITS:-dev_unseen}"
ENCODER_BACKEND="${ENCODER_BACKEND:-st_image}"
LLM_MODEL="${LLM_MODEL:-Qwen/Qwen3-VL-Embedding-8B}"
EMBEDDING_INSTRUCTION="${EMBEDDING_INSTRUCTION:-Represent the semantic cue field for hateful meme classification.}"
EMBEDDING_OUTPUT_DIM="${EMBEDDING_OUTPUT_DIM:-4096}"
ENCODER_BATCH_SIZE="${ENCODER_BATCH_SIZE:-8}"
USE_LORA="${USE_LORA:-0}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}"
HF_DEVICE_MAP="${HF_DEVICE_MAP:-}"
HF_TORCH_DTYPE="${HF_TORCH_DTYPE:-}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-64}"
PROJ_DIM="${PROJ_DIM:-768}"
NUM_HEADS="${NUM_HEADS:-8}"
NUM_LAYERS="${NUM_LAYERS:-2}"
CONTRASTIVE_WEIGHT="${CONTRASTIVE_WEIGHT:-1.0}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOPK="${TOPK:-20}"
LR="${LR:-0.0001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
SEED="${SEED:-1}"
GPU="${GPU:-1}"

read -r -a TEST_SPLIT_ARGS <<< "${TEST_SPLITS}"
read -r -a EXTRA_EVAL_SPLIT_ARGS <<< "${EXTRA_EVAL_SPLITS}"

EXTRA_ARGS=()
if [[ "${USE_LORA}" == "1" || "${USE_LORA}" == "true" || "${USE_LORA}" == "yes" ]]; then
  EXTRA_ARGS+=(--use-lora)
fi
if [[ -n "${HF_DEVICE_MAP}" ]]; then
  EXTRA_ARGS+=(--hf-device-map "${HF_DEVICE_MAP}")
fi
if [[ -n "${HF_TORCH_DTYPE}" ]]; then
  EXTRA_ARGS+=(--hf-torch-dtype "${HF_TORCH_DTYPE}")
fi
EXTRA_ARGS+=("$@")

ENV_ARGS=(
  --environment "PYTHONPATH=${PYTHONPATH_TO_RUN}"
  --environment "PYTHONUNBUFFERED=1"
)
if [[ -n "${HF_TOKEN:-}" ]]; then
  ENV_ARGS+=(--environment "HF_TOKEN=${HF_TOKEN}")
fi

CMD=(
  bash "${REPO_DIR}/scripts/run_train_checked.sh"
  --cues-jsonl "${CUES_JSONL}"
  --image-dir "${IMAGE_DIR}"
  --dataset-name "${DATASET_NAME}"
  --cache-dir "${CACHE_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --train-split "${TRAIN_SPLIT}"
  --dev-split "${DEV_SPLIT}"
  --test-split "${TEST_SPLIT_ARGS[@]}"
  --extra-eval-split "${EXTRA_EVAL_SPLIT_ARGS[@]}"
  --encoder-backend "${ENCODER_BACKEND}"
  --llm-model "${LLM_MODEL}"
  --embedding-instruction "${EMBEDDING_INSTRUCTION}"
  --embedding-output-dim "${EMBEDDING_OUTPUT_DIM}"
  --encoder-batch-size "${ENCODER_BATCH_SIZE}"
  --lora-r "${LORA_R}"
  --lora-alpha "${LORA_ALPHA}"
  --lora-dropout "${LORA_DROPOUT}"
  --lora-target-modules "${LORA_TARGET_MODULES}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --proj-dim "${PROJ_DIM}"
  --num-heads "${NUM_HEADS}"
  --num-layers "${NUM_LAYERS}"
  --contrastive-weight "${CONTRASTIVE_WEIGHT}"
  --temperature "${TEMPERATURE}"
  --topk "${TOPK}"
  --grad-clip "${GRAD_CLIP}"
  --seed "${SEED}"
)
CMD+=("${EXTRA_ARGS[@]}")

printf '[local] command:'
printf ' %q' "${CMD[@]}"
printf '\n'

runai submit --run-as-uid "${UID_TO_RUN}" \
  --image "${IMAGE}" \
  --gpu "${GPU}" \
  --existing-pvc "claimname=${PVC},path=/scratch" \
  "${ENV_ARGS[@]}" \
  --command -- "${CMD[@]}"
