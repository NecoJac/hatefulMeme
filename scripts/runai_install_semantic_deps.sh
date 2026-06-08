#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-registry.rcp.epfl.ch/ee559/environment-with-packages:latest}"
PVC="${PVC:-course-ee-559-scratch-g36}"
UID_TO_RUN="${UID_TO_RUN:-287685}"
REPO_DIR="${REPO_DIR:-/scratch/hateful_meme_semantic_retrieval}"
VENDOR_DIR="${VENDOR_DIR:-${REPO_DIR}/vendor/python}"
INSTALL_SCRIPT="${INSTALL_SCRIPT:-${REPO_DIR}/scripts/install_semantic_deps.sh}"

ENV_ARGS=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  ENV_ARGS+=(--environment "HF_TOKEN=${HF_TOKEN}")
fi

runai submit --run-as-uid "${UID_TO_RUN}" \
  --image "${IMAGE}" \
  --gpu 0 \
  --existing-pvc "claimname=${PVC},path=/scratch" \
  "${ENV_ARGS[@]}" \
  --environment "REPO_DIR=${REPO_DIR}" \
  --environment "VENDOR_DIR=${VENDOR_DIR}" \
  --command -- bash "${INSTALL_SCRIPT}"
