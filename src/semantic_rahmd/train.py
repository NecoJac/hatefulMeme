from __future__ import annotations

import argparse
from collections import Counter
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from semantic_rahmd.data import SemanticCueDataset, collate_semantic_batch, records_for_split, records_from_jsonl, split_records
from semantic_rahmd.losses import rahmd_loss
from semantic_rahmd.metrics import binary_metrics, rkc_logits
from semantic_rahmd.model import build_model


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Train semantic-field RA-HMD on frozen VLM cues.")
    parser.add_argument("--cues-jsonl", default=str(repo_root / "data" / "facebook_semantic_cues.jsonl"))
    parser.add_argument("--image-dir", default=str(repo_root / "data" / "images"))
    parser.add_argument("--dataset-name", default="cs5242-hateful-memes/hateful-memes-data")
    parser.add_argument("--cache-dir", default=str(repo_root / "data" / "hf_cache"))
    parser.add_argument("--output-dir", default="outputs/semantic_rahmd")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--dev-split", default="dev_seen")
    parser.add_argument("--test-split", nargs="+", default=["test_seen", "test_unseen"])
    parser.add_argument("--extra-eval-split", nargs="*", default=["dev_unseen"])
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--encoder-backend", choices=["hf", "hash", "st", "st_image", "hf_lora"], default="hash")
    parser.add_argument("--llm-model", default="distilbert-base-uncased")
    parser.add_argument("--embedding-instruction", default="Represent the semantic cue field for hateful meme classification.")
    parser.add_argument("--embedding-output-dim", type=int, default=None)
    parser.add_argument("--encoder-batch-size", type=int, default=16)
    parser.add_argument("--train-llm", action="store_true")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated module names for HF LoRA. Defaults match common Qwen attention projections.",
    )
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--hf-device-map",
        default=None,
        help='Optional device_map for HF backbones, e.g. "auto" for multi-GPU model sharding.',
    )
    parser.add_argument(
        "--hf-torch-dtype",
        default=None,
        help='Optional torch dtype for HF backbones: "auto", "float16", "bfloat16", or "float32".',
    )
    parser.add_argument("--hash-dim", type=int, default=384)
    parser.add_argument("--proj-dim", type=int, default=768)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--use-residual-projection",
        action="store_true",
        help="Add a residual MLP projection before the Transformer aggregator. Disabled by default for compatibility with the final checkpoint.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--contrastive-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-embeddings", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(records, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    return DataLoader(
        SemanticCueDataset(records),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=collate_semantic_batch,
    )


@torch.no_grad()
def collect_outputs(model, loader: DataLoader, device: torch.device) -> dict[str, torch.Tensor | list[str]]:
    """Materialize logits/embeddings for retrieval memory and split evaluation."""
    model.eval()
    all_logits, all_embeddings, all_labels, all_ids = [], [], [], []
    for batch in loader:
        labels = batch["labels"].to(device)
        outputs = model(batch["fields"], image_paths=batch.get("image_paths"), images=batch.get("images"))
        all_logits.append(outputs["logits"].detach().cpu())
        all_embeddings.append(outputs["embedding"].detach().cpu())
        all_labels.append(labels.detach().cpu())
        all_ids.extend(batch["ids"])
    return {
        "ids": all_ids,
        "logits": torch.cat(all_logits),
        "embeddings": torch.cat(all_embeddings),
        "labels": torch.cat(all_labels),
    }


def evaluate(model, train_loader: DataLoader, eval_loader: DataLoader, device: torch.device, args: argparse.Namespace) -> dict[str, float | None]:
    """Evaluate both the classifier head and retrieval/RKC over train embeddings."""
    train_out = collect_outputs(model, train_loader, device)
    eval_out = collect_outputs(model, eval_loader, device)
    clf = binary_metrics(eval_out["logits"], eval_out["labels"])
    knn_logits = rkc_logits(
        train_out["embeddings"],
        train_out["labels"],
        eval_out["embeddings"],
        topk=args.topk,
        temperature=args.temperature,
    )
    rkc = binary_metrics(knn_logits, eval_out["labels"])
    return {
        "clf_acc": clf["acc"],
        "clf_f1": clf["f1"],
        "clf_auroc": clf["auroc"],
        "rkc_acc": rkc["acc"],
        "rkc_f1": rkc["f1"],
        "rkc_auroc": rkc["auroc"],
        "knn_acc": rkc["acc"],
        "knn_f1": rkc["f1"],
        "knn_auroc": rkc["auroc"],
    }


def train_epoch(
    model,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    args: argparse.Namespace,
    memory: dict[str, torch.Tensor | list[str]] | None = None,
) -> dict[str, float]:
    model.train()
    running = {"loss": 0.0, "bce": 0.0, "lcl": 0.0}
    # Train-memory negatives/positives make LCL closer to full-database retrieval than batch-only loss.
    memory_embeddings = memory["embeddings"].to(device) if memory is not None else None
    memory_labels = memory["labels"].to(device) if memory is not None else None
    memory_ids = memory["ids"] if memory is not None else None
    for batch in loader:
        labels = batch["labels"].to(device)
        outputs = model(batch["fields"], image_paths=batch.get("image_paths"), images=batch.get("images"))
        loss, parts = rahmd_loss(
            outputs["logits"],
            outputs["embedding"],
            labels,
            contrastive_weight=args.contrastive_weight,
            temperature=args.temperature,
            pos_weight=args.pos_weight,
            memory_embeddings=memory_embeddings,
            memory_labels=memory_labels,
            anchor_ids=batch["ids"],
            memory_ids=memory_ids,
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optimizer.step()
        running["loss"] += float(loss.detach())
        running["bce"] += parts["bce"]
        running["lcl"] += parts["lcl"]
    denom = max(1, len(loader))
    return {key: value / denom for key, value in running.items()}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = records_from_jsonl(
        args.cues_jsonl,
        image_dir=args.image_dir,
        dataset_name=args.dataset_name,
        cache_dir=args.cache_dir,
        use_hf_images=args.encoder_backend == "st_image",
    )
    split_counts = Counter(record.split for record in records)
    train_records, dev_records, test_records = split_records(
        records,
        train_split=args.train_split,
        dev_split=args.dev_split,
        test_split=args.test_split,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    train_loader = make_loader(train_records, args, shuffle=True)
    train_eval_loader = make_loader(train_records, args, shuffle=False)
    dev_loader = make_loader(dev_records, args, shuffle=False)
    test_loader = make_loader(test_records, args, shuffle=False) if test_records else None

    device = torch.device(args.device)
    model = build_model(args)
    if args.encoder_backend == "hf_lora" or args.hf_device_map:
        model.aggregator.to(device)
        model.classifier.to(device)
    else:
        model.to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)

    best_metrics = {"clf_auroc": -1.0, "rkc_auroc": -1.0}
    best_paths = {
        "clf_auroc": output_dir / "best_clf_auroc.pt",
        "rkc_auroc": output_dir / "best_rkc_auroc.pt",
    }
    best_path = output_dir / "best_model.pt"
    history = []
    print(f"[data] split_counts={dict(sorted(split_counts.items()))}")
    print(f"[data] train={len(train_records)} dev={len(dev_records)} test={len(test_records) if test_records else 0}")
    print(f"[model] backend={args.encoder_backend} trainable_params={sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    for epoch in range(1, args.epochs + 1):
        # Refresh retrieval memory once per epoch using the current encoder/aggregator.
        train_memory = collect_outputs(model, train_eval_loader, device)
        train_stats = train_epoch(model, train_loader, optimizer, device, args, memory=train_memory)
        dev_stats = evaluate(model, train_eval_loader, dev_loader, device, args)
        row = {"epoch": epoch, **train_stats, **dev_stats}
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        # Save separate checkpoints for the direct classifier and retrieval/RKC selection metrics.
        for metric_name, checkpoint_path in best_paths.items():
            metric_value = dev_stats.get(metric_name)
            if metric_value is None:
                continue
            if metric_value > best_metrics[metric_name]:
                best_metrics[metric_name] = metric_value
                payload = {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "selection_metric": metric_name,
                    "selection_value": metric_value,
                    "epoch": epoch,
                }
                torch.save(payload, checkpoint_path)
                if metric_name == "clf_auroc":
                    torch.save(payload, best_path)

    summary = {
        "best_dev_clf_auroc": best_metrics["clf_auroc"],
        "best_dev_rkc_auroc": best_metrics["rkc_auroc"],
        "best_dev_knn_auroc": best_metrics["rkc_auroc"],
        "best_model": str(best_path),
        "best_clf_auroc_model": str(best_paths["clf_auroc"]),
        "best_rkc_auroc_model": str(best_paths["rkc_auroc"]),
        "best_knn_auroc_model": str(best_paths["rkc_auroc"]),
        "history": history,
        "eval_by_checkpoint": {},
    }

    for checkpoint_name, checkpoint_path in best_paths.items():
        if not checkpoint_path.exists():
            continue
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model"])
        checkpoint_summary = {
            "selection_metric": state.get("selection_metric", checkpoint_name),
            "selection_value": state.get("selection_value"),
            "epoch": state.get("epoch"),
            "eval_by_split": {},
        }
        for split in [*args.extra_eval_split, *args.test_split]:
            eval_split_records = records_for_split(records, split)
            if not eval_split_records:
                continue
            split_loader = make_loader(eval_split_records, args, shuffle=False)
            checkpoint_summary["eval_by_split"][split] = evaluate(model, train_eval_loader, split_loader, device, args)
        if test_loader is not None:
            checkpoint_summary["test"] = evaluate(model, train_eval_loader, test_loader, device, args)
        summary["eval_by_checkpoint"][checkpoint_name] = checkpoint_summary

    if "clf_auroc" in summary["eval_by_checkpoint"]:
        summary["eval_by_split"] = summary["eval_by_checkpoint"]["clf_auroc"]["eval_by_split"]
        summary["test"] = summary["eval_by_checkpoint"]["clf_auroc"].get("test")

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if args.save_embeddings:
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state["model"])
        train_out = collect_outputs(model, train_eval_loader, device)
        torch.save(train_out, output_dir / "train_embeddings.pt")

    print(f"[done] summary={output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
