#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

python3 -u -m semantic_rahmd.train   --cues-jsonl "${CUES_JSONL:-examples/example_semantic_cues.jsonl}"   --output-dir "${OUTPUT_DIR:-outputs/smoke_hash}"   --encoder-backend hash   --hash-dim 128   --proj-dim 128   --num-heads 4   --num-layers 1   --batch-size 4   --epochs "${EPOCHS:-2}"   --lr 0.001   --contrastive-weight 0.1   --test-split test_seen test_unseen   --extra-eval-split dev_unseen   "$@"
