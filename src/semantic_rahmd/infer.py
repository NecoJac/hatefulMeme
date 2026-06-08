from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from semantic_rahmd.data import SEMANTIC_FIELDS, load_jsonl, parse_semantic_cues
from semantic_rahmd.model import build_model


FIELD_NAMES = ["Text", *SEMANTIC_FIELDS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run semantic RA-HMD inference on one meme example.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained best_model.pt checkpoint.")
    parser.add_argument("--example-json", help="JSON file with text and semantic_cues fields.")
    parser.add_argument("--cues-jsonl", help="JSONL file produced by extract_facebook_semantic_cues.py.")
    parser.add_argument("--id", help="Sample id to select from --cues-jsonl. Defaults to the first usable row.")
    parser.add_argument("--text", help="Raw meme text for manual inference.")
    parser.add_argument("--semantic-cues", help="Structured semantic cue text, or a path to a text file.")
    parser.add_argument("--image", help="Optional image path for checkpoints trained with --encoder-backend st_image.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def read_semantic_cues(value: str | None) -> str:
    if value is None:
        return ""
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return candidate.read_text(encoding="utf-8")
    return value


def load_example(args: argparse.Namespace) -> dict[str, Any]:
    """Accept one of three inference input styles: JSON, JSONL lookup, or manual text/cues."""
    sources = [args.example_json is not None, args.cues_jsonl is not None, args.text is not None or args.semantic_cues is not None]
    if sum(sources) != 1:
        raise ValueError("Choose exactly one input source: --example-json, --cues-jsonl, or --text/--semantic-cues.")

    if args.example_json:
        with Path(args.example_json).expanduser().open("r", encoding="utf-8") as f:
            return json.load(f)

    if args.cues_jsonl:
        for item in load_jsonl(args.cues_jsonl):
            if "error" in item or "semantic_cues" not in item:
                continue
            if args.id is None or str(item.get("id")) == str(args.id):
                return item
        selector = f"id={args.id}" if args.id is not None else "first usable row"
        raise ValueError(f"No example found in {args.cues_jsonl} for {selector}")

    return {
        "id": args.id or "manual",
        "text": args.text or "",
        "semantic_cues": read_semantic_cues(args.semantic_cues),
    }


def item_to_fields(item: dict[str, Any]) -> list[str]:
    """Convert raw cue text into the ordered field list expected by the model."""
    if "fields" in item:
        fields = [str(value) for value in item["fields"]]
        if len(fields) != len(FIELD_NAMES):
            raise ValueError(f"Expected {len(FIELD_NAMES)} fields, got {len(fields)}")
        return fields

    parsed = parse_semantic_cues(str(item.get("semantic_cues", "")))
    fields = [str(item.get("text", ""))]
    fields.extend(parsed[field] for field in SEMANTIC_FIELDS)
    return fields


def checkpoint_args(saved_args: dict[str, Any]) -> argparse.Namespace:
    """Merge checkpoint args with defaults for backwards-compatible loading."""
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
        "num_layers": 1,
        "dropout": 0.1,
        "use_residual_projection": False,
    }
    defaults.update(saved_args)
    return argparse.Namespace(**defaults)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    item = load_example(args)
    fields = item_to_fields(item)

    state = torch.load(Path(args.checkpoint).expanduser(), map_location=device)
    model_args = checkpoint_args(state.get("args", {}))
    model = build_model(model_args).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    # Image-aware checkpoints need a real image path in addition to semantic fields.
    image_path = args.image or item.get("image_path") or item.get("img_path") or item.get("path")
    if getattr(model_args, "encoder_backend", None) == "st_image" and not image_path:
        raise ValueError("This checkpoint uses st_image. Pass --image, or provide image_path in the example JSON.")

    outputs = model([fields], image_paths=[image_path] if image_path else None)
    logit = float(outputs["logits"][0].detach().cpu())
    probability = float(torch.sigmoid(outputs["logits"])[0].detach().cpu())
    field_weights = outputs["field_weights"][0].detach().cpu().tolist()

    result = {
        "id": str(item.get("id", "unknown")),
        "label": item.get("label"),
        "logit": logit,
        "probability": probability,
        "prediction": int(probability >= args.threshold),
        "threshold": args.threshold,
        "field_weights": dict(zip(FIELD_NAMES, field_weights)),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
