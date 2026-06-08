#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-registry.rcp.epfl.ch/ee559/environment-with-packages:latest}"
PVC="${PVC:-course-ee-559-scratch-g36}"
UID_TO_RUN="${UID_TO_RUN:-287685}"
REPO_DIR="${REPO_DIR:-/scratch/hateful_meme_semantic_retrieval}"
VENDOR_PYTHONPATH="${VENDOR_PYTHONPATH:-${REPO_DIR}/vendor/python}"
PYTHONPATH_TO_RUN="${PYTHONPATH_TO_RUN:-${VENDOR_PYTHONPATH}:${REPO_DIR}/src}"
GPU="${GPU:-1}"
PORT="${PORT:-8888}"

ENV_ARGS=(
  --environment "PYTHONPATH=${PYTHONPATH_TO_RUN}"
  --environment "PYTHONUNBUFFERED=1"
)
if [[ -n "${HF_TOKEN:-}" ]]; then
  ENV_ARGS+=(--environment "HF_TOKEN=${HF_TOKEN}")
fi

runai submit --interactive --attach   --run-as-uid "${UID_TO_RUN}"   --image "${IMAGE}"   --gpu "${GPU}"   --existing-pvc "claimname=${PVC},path=/scratch"   "${ENV_ARGS[@]}"   --command -- bash -lc "cd ${REPO_DIR} && python3 -m jupyter lab --ip=0.0.0.0 --port=${PORT} --no-browser --NotebookApp.token='' --NotebookApp.password=''"
