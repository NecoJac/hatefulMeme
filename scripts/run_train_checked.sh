#!/usr/bin/env bash
set -euo pipefail

ENCODER_BACKEND_VALUE=""
USE_LORA_VALUE="0"
ORIGINAL_ARGS=("$@")
while [[ $# -gt 0 ]]; do
  case "$1" in
    --encoder-backend)
      ENCODER_BACKEND_VALUE="${2:-}"
      shift 2
      ;;
    --use-lora)
      USE_LORA_VALUE="1"
      shift
      ;;
    *)
      shift
      ;;
  esac
done

export SEMANTIC_RAHMD_ENCODER_BACKEND="${ENCODER_BACKEND_VALUE}"
export SEMANTIC_RAHMD_USE_LORA="${USE_LORA_VALUE}"

python3 - <<'PY'
import importlib
import os
import sys
from pathlib import Path

print("[env] python", sys.version.replace("\n", " "))
print("[env] sys.path[:5]", sys.path[:5])

backend = os.environ.get("SEMANTIC_RAHMD_ENCODER_BACKEND", "")
use_lora = os.environ.get("SEMANTIC_RAHMD_USE_LORA", "0") == "1" or backend == "hf_lora"

if backend in {"st", "st_image"}:
    try:
        st = importlib.import_module("sentence_transformers")
    except Exception as exc:
        raise SystemExit(f"[env] failed to import sentence_transformers: {exc}") from exc

    version = getattr(st, "__version__", "unknown")
    path = Path(getattr(st, "__file__", "unknown"))
    print("[env] sentence_transformers", version, path)
    if version < "5.4.0":
        raise SystemExit(
            "[env] sentence-transformers>=5.4.0 is required for "
            "Qwen/Qwen3-VL-Embedding-8B. Run "
            "./scripts/runai_install_semantic_deps.sh first."
        )

if use_lora:
    try:
        peft = importlib.import_module("peft")
    except Exception as exc:
        raise SystemExit(f"[env] failed to import peft for LoRA: {exc}") from exc
    print("[env] peft", getattr(peft, "__version__", "unknown"), Path(getattr(peft, "__file__", "unknown")))
PY

exec python3 -u -m semantic_rahmd.train "${ORIGINAL_ARGS[@]}"
