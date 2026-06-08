from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from semantic_rahmd.data import (
    SemanticCueDataset,
    collate_semantic_batch,
    records_for_split,
    records_from_jsonl,
)
from semantic_rahmd.model import build_model
from semantic_rahmd.train import evaluate


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Evaluate a saved Semantic RA-HMD checkpoint on named splits.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt.")
    parser.add_argument("--cues-jsonl", default=str(repo_root / "data" / "facebook_semantic_cues.jsonl"))
    parser.add_argument("--image-dir", default=str(repo_root / "data" / "images"))
    parser.add_argument("--dataset-name", default="cs5242-hateful-memes/hateful-memes-data")
    parser.add_argument("--cache-dir", default=str(repo_root / "data" / "hf_cache"))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", nargs="+", default=["test_seen", "test_unseen"])
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="Override checkpoint batch size for evaluation.")
    parser.add_argument("--encoder-batch-size", type=int, default=None, help="Override checkpoint encoder batch size.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def checkpoint_args(saved_args: dict[str, Any], cli_args: argparse.Namespace) -> argparse.Namespace:
    """Restore training-time args while allowing evaluation batch-size overrides."""
    defaults = {
        "encoder_backend": "hash",
        "llm_model": "distilbert-base-uncased",
        "embedding_instruction": "Represent the semantic cue field for hateful meme classification.",
        "embedding_output_dim": None,
        "encoder_batch_size": 16,
        "train_llm": False,
        "use_lora": False,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_target_modules": "q_proj,k_proj,v_proj,o_proj",
        "trust_remote_code": True,
        "hf_device_map": None,
        "hf_torch_dtype": None,
        "hash_dim": 384,
        "proj_dim": 768,
        "num_heads": 8,
        "num_layers": 2,
        "dropout": 0.1,
        "use_residual_projection": False,
        "batch_size": 64,
        "temperature": 1.0,
        "topk": 20,
    }
    defaults.update(saved_args)
    if cli_args.batch_size is not None:
        defaults["batch_size"] = cli_args.batch_size
    if cli_args.encoder_batch_size is not None:
        defaults["encoder_batch_size"] = cli_args.encoder_batch_size
    defaults["num_workers"] = cli_args.num_workers
    return argparse.Namespace(**defaults)


def make_loader(records, args: argparse.Namespace, shuffle: bool = False):
    """Use the same collate path as training so image/cue batches match checkpoint expectations."""
    return torch.utils.data.DataLoader(
        SemanticCueDataset(records),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=collate_semantic_batch,
    )


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = Path(args.checkpoint).expanduser()
    state = torch.load(checkpoint, map_location=device)
    model_args = checkpoint_args(state.get("args", {}), args)

    # Rebuild train memory because retrieval/RKC evaluation compares eval samples to train embeddings.
    records = records_from_jsonl(
        args.cues_jsonl,
        image_dir=args.image_dir,
        dataset_name=args.dataset_name,
        cache_dir=args.cache_dir,
        use_hf_images=model_args.encoder_backend == "st_image",
    )
    train_records = records_for_split(records, args.train_split)
    if not train_records:
        raise ValueError(f"No train records found for split {args.train_split!r}.")
    train_loader = make_loader(train_records, model_args)

    model = build_model(model_args)
    # Sharded HF backbones already manage their own device placement; move only trainable heads.
    if model_args.encoder_backend == "hf_lora" or model_args.hf_device_map:
        model.aggregator.to(device)
        model.classifier.to(device)
    else:
        model.to(device)
    model.load_state_dict(state["model"])
    model.eval()

    results: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "encoder_backend": model_args.encoder_backend,
        "train_split": args.train_split,
        "eval_by_split": {},
    }
    for split in args.eval_split:
        split_records = records_for_split(records, split)
        if not split_records:
            results["eval_by_split"][split] = None
            continue
        split_loader = make_loader(split_records, model_args)
        metrics = evaluate(model, train_loader, split_loader, device, model_args)
        results["eval_by_split"][split] = metrics
        print(json.dumps({"split": split, **metrics}, sort_keys=True), flush=True)

    if args.output_json:
        output_path = Path(args.output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[done] output={output_path}")
    else:
        print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
