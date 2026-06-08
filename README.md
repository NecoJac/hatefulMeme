# Hateful Meme Semantic Retrieval

This repository contains the final project code for hateful meme detection with a frozen multimodal semantic representation. Unless otherwise stated, commands below assume the repository root is the current directory:

```bash
cd hateful_meme_semantic_retrieval
```

Final model used in the report:

```text
meme image + original meme text + VLM-generated semantic cues
  -> frozen Qwen3-VL-Embedding-8B
  -> residual projection + 2-layer Transformer field aggregator
  -> meme embedding
  -> binary classifier + retrieval/RKC evaluation
```

Large artifacts may exist in this working copy under `data/`, `outputs/`, and `vendor/`, but they are ignored by git and should not be uploaded with the code submission. Small notebook demo samples under `notebooks/demo_samples/` are included only for the screencast demo.

## Repository Layout

```text
src/semantic_rahmd/              model, data loading, training, evaluation, inference
src/qwen_hatememe/               Qwen model identifiers and small runtime helpers
scripts/extract_facebook_semantic_cues.py
scripts/runai_extract_semantic_cues.sh
scripts/runai_semantic_rahmd.sh  generic RunAI training entrypoint
scripts/train_frozen_image.sh    local final-model training command
scripts/runai_eval_checkpoint.sh RunAI checkpoint evaluation
scripts/runai_notebook.sh        interactive Jupyter/Lab job for screencast
scripts/run_smoke_train.sh       tiny CPU smoke test using examples/example_semantic_cues.jsonl
notebooks/screencast_inference_demo.ipynb
results/ablation_test_seen.csv   compact ablation table
```

## External Dataset

We use the Facebook Hateful Memes dataset via Hugging Face datasets:

- Dataset: `cs5242-hateful-memes/hateful-memes-data`
- Link: https://huggingface.co/datasets/cs5242-hateful-memes/hateful-memes-data

The dataset is not part of the code submission. In this working copy, cached data can live at:

```text
data/hf_cache/
data/images/
```

Processed split counts used in the project:

| Split | Count |
|---|---:|
| Train | 8,500 |
| Dev seen | 398 |
| Dev unseen | 433 |
| Test seen | 1,000 |
| Test unseen | 1,593 |

## External Models

Model weights are not submitted. The code loads them through Hugging Face.

| Component | Model | Link |
|---|---|---|
| Semantic cue generator VLM | `Qwen/Qwen3-VL-8B-Instruct` | https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct |
| Frozen embedding encoder | `Qwen/Qwen3-VL-Embedding-8B` | https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B |

## Install

Local Python environment:

```bash
python3 -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
```

RunAI uses a repo-local dependency directory at `vendor/python`. Install it once before the real Qwen inference notebook:

```bash
bash scripts/runai_install_semantic_deps.sh
```

This is important because Qwen3-VL-Embedding requires `sentence-transformers>=5.4.0`; some course images contain an older version.

## 1. Extract Semantic Cues

This step runs the frozen VLM over each meme image and text, then writes structured neutral semantic cues.

```bash
SPLITS="train dev_seen dev_unseen test_seen test_unseen" \
OUTPUT="$PWD/data/facebook_semantic_cues.jsonl" \
bash scripts/runai_extract_semantic_cues.sh
```

Each JSONL row contains `split`, `id`, `text`, `label`, and `semantic_cues`.

The seven generated semantic cue fields are:

```text
GlobalDescription
TargetCandidateType
TargetCandidate
ProtectedTargetPossible
ProtectedTargetType
RelationOrAction
SafetyReasonCode
```

Together with the original meme text, the model uses eight textual semantic units plus the raw image embedding.

## 2. Train the Final Frozen Image + Text + Cues Model

RunAI command:

```bash
ENCODER_BACKEND=st_image \
OUTPUT_DIR="$PWD/outputs/runai_qwen_vl_fields_image" \
BATCH_SIZE=64 \
PROJ_DIM=768 \
LR=0.0001 \
EPOCHS=30 \
bash scripts/runai_semantic_rahmd.sh
```

Local command with the same model variant:

```bash
bash scripts/train_frozen_image.sh
```

Training writes the checkpoint and summary under:

```text
outputs/runai_qwen_vl_fields_image/best_model.pt
outputs/runai_qwen_vl_fields_image/summary.json
```

New runs of the current training code may also save metric-specific checkpoints such as `best_clf_auroc.pt` and `best_rkc_auroc.pt`. The screencast notebook uses the ResProj1024 + RKC run under `outputs/runai_qwen_vl_fields_image_resproj1024_rkc_lr5e5/`, preferring `best_clf_auroc.pt` for direct prediction display.

## 3. Evaluate a Checkpoint

```bash
CHECKPOINT="$PWD/outputs/runai_qwen_vl_fields_image/best_model.pt" \
OUTPUT_JSON="$PWD/outputs/runai_qwen_vl_fields_image/test_seen_unseen_eval.json" \
bash scripts/runai_eval_checkpoint.sh
```

## 4. Screencast Notebook Workflow

The notebook is designed for a short, stable screencast. It does not run training commands or shell scripts inside the notebook. It directly imports the submitted Python code, loads four fixed test-set demo examples from `notebooks/demo_samples/`, loads the real trained checkpoint, and displays image, text, semantic cues, ground truth, and prediction.

Real checkpoint used by the notebook:

```text
outputs/runai_qwen_vl_fields_image_resproj1024_rkc_lr5e5/best_clf_auroc.pt
```

If `best_clf_auroc.pt` is missing, the notebook falls back to:

```text
outputs/runai_qwen_vl_fields_image_resproj1024_rkc_lr5e5/best_model.pt
```

Checkpoint files under `outputs/` are local generated artifacts and are ignored by git. For a fresh clone, train/evaluate first or copy the saved checkpoint into the path above before running the real-model notebook cell.

### 4.1 Install Notebook Dependencies on RunAI

Run this once in a terminal from the repository root:

```bash
bash scripts/runai_install_semantic_deps.sh
```

Wait until the install job finishes. It creates or updates:

```text
vendor/python/
```

### 4.2 Start JupyterLab on RunAI

```bash
GPU=1 bash scripts/runai_notebook.sh
```

The terminal will show a job name like:

```text
Job job-xxxxxxxx submitted successfully.
```

Keep this terminal open.

### 4.3 Port-Forward JupyterLab

Open a second terminal and run, replacing `job-xxxxxxxx` with the job name printed above:

```bash
runai port-forward job-xxxxxxxx \
  --port 8888:8888 \
  --address localhost \
  -p course-ee-559-sjiang
```

If local port `8888` is busy, use `8890:8888` and open `http://127.0.0.1:8890/lab`.

Then open this URL in your local browser:

```text
http://127.0.0.1:8888/lab
```

### 4.4 Open and Run the Notebook

In JupyterLab, open:

```text
notebooks/screencast_inference_demo.ipynb
```

Select the `Python 3` kernel if prompted, then run the notebook cells in order:

```text
0. Environment setup
1. Load four test demo samples
2. Load the real model checkpoint
3. Run inference and display results
```

Expected successful environment output includes:

```text
semantic_rahmd import: OK
sentence_transformers: 5.4.0 .../vendor/python/...
```

The final cell displays each test sample's image, meme text, semantic cues, ground-truth label, predicted label, hateful probability, and field-attention weights.

## 5. Smoke Test Without External Data

```bash
bash scripts/run_smoke_train.sh
PYTHONPATH="$PWD/src:${PYTHONPATH:-}" python3 -m semantic_rahmd.infer   --checkpoint outputs/smoke_hash/best_model.pt   --cues-jsonl examples/example_semantic_cues.jsonl   --id toy_007   --device cpu
```

The smoke model uses a deterministic hash encoder, so it is only for checking that the code path runs.

## Method Details

Each meme is represented by:

```text
[original meme text,
 GlobalDescription,
 TargetCandidateType,
 TargetCandidate,
 ProtectedTargetPossible,
 ProtectedTargetType,
 RelationOrAction,
 SafetyReasonCode,
 raw image]
```

The frozen Qwen3-VL embedding model encodes the semantic text units and raw image. A residual bottleneck projection and 2-layer Transformer aggregator fuse these unit-level representations into one normalized meme embedding.

Training objective:

```text
L = L_BCE + lambda * L_LCL
```

Evaluation reports both direct classifier metrics and retrieval/RKC metrics over the learned meme embedding.

## Test Seen Ablation

| Variant | Clf AUROC | Clf Acc | Retrieval AUROC | Retrieval Acc |
|---|---:|---:|---:|---:|
| Frozen semantic fields | 0.805 | 0.706 | 0.796 | 0.727 |
| LoRA semantic fields | 0.807 | 0.733 | 0.747 | 0.728 |
| Frozen fields + image | 0.827 | 0.729 | 0.806 | 0.748 |
| ResProj1024 + RKC | **0.833** | 0.720 | **0.829** | 0.740 |

Main conclusion: semantic fields are useful and interpretable, but preserving the raw image improves the frozen model. The submitted final method is the frozen image + text + cues model described in the report; the other rows are kept as ablations.
