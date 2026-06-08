#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/scratch/hateful_meme_semantic_retrieval}"
VENDOR_DIR="${VENDOR_DIR:-${REPO_DIR}/vendor/python}"

mkdir -p "${VENDOR_DIR}"
python3 -m pip install \
  --target "${VENDOR_DIR}" \
  --upgrade \
  --no-deps \
  sentence-transformers==5.4.0 \
  peft \
  accelerate \
  Pillow \
  pyarrow \
  qwen-vl-utils

PYTHONPATH="${VENDOR_DIR}:${PYTHONPATH:-}" python3 - <<'PY'
import sentence_transformers
import peft
import accelerate
import PIL
import pyarrow
import qwen_vl_utils
print(sentence_transformers.__version__)
print(sentence_transformers.__file__)
print(peft.__version__)
print(peft.__file__)
print(accelerate.__version__)
print(accelerate.__file__)
print(PIL.__version__)
print(PIL.__file__)
print(pyarrow.__version__)
print(pyarrow.__file__)
print(qwen_vl_utils.__file__)
PY
