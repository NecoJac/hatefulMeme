#!/usr/bin/env python3
"""Extract neutral semantic cues from Facebook hateful-meme samples with Qwen3-VL."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from qwen_hatememe.config import QWEN3_VL_MODEL


SEMANTIC_CUE_PROMPT = """You are assisting a hateful meme detection system.

Analyze the meme image together with the visible meme text in the image.
Do NOT decide whether the meme is hateful or not.
Only extract a small set of neutral, structured semantic cues for later classification.

Important rules:
- Only use information clearly visible in the image or explicit meme text.
- Do NOT guess sensitive identity attributes from appearance alone.
- A hate target must be a person or a group.
- Do NOT treat a country, flag, national symbol, public office, public official, religious leader, or belief doctrine as a hate target by itself.
- If the meme refers only to an object, place, flag, office, official, leader, or doctrine, do not mark it as a protected target.
- If a visible object, animal, place, or symbol is used to insult, compare, or imply a person/group target, set TargetCandidate to the implied person/group target rather than the literal object.
- Do not mark a person or group as a protected target unless the meme explicitly indicates a protected identity factor.
- If ProtectedTargetPossible is "no", then ProtectedTargetType must be "none".
- If ProtectedTargetPossible is "yes", then ProtectedTargetType must not be "none".
- If the meme text refers to multiple people using plural forms such as "they", "them", or "their", prefer TargetCandidateType = group unless there is clear evidence of a single person.
- If a field is unclear, output "unknown".
- If a field does not apply, output "none".
- Keep the output short, factual, and structured.

Return exactly in this format:

GlobalDescription: <brief neutral summary of the meme image and visible text>
TargetCandidateType: <person / group / none / unknown>
TargetCandidate: <main person/group referred to by the meme, especially the likely target of the text-image message, or none/unknown>
ProtectedTargetPossible: <yes / no / unknown>
ProtectedTargetType: <religion / ethnicity / nationality / race / skin_color / descent / sex / language / socioeconomic_origin / disability / health_condition / sexual_orientation / none / unknown>
RelationOrAction: <main visible relation, action, or interaction, or none>
SafetyReasonCode: <explicit_person_group / protected_attribute_explicit / only_flag_or_country / only_public_office_or_official / only_religious_leader_or_doctrine / only_animal_object_place / unclear>"""


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run Qwen3-VL over Facebook hateful-meme dataset samples and save textified semantic cues."
    )
    parser.add_argument("--model", default=QWEN3_VL_MODEL)
    parser.add_argument("--dataset-name", default="cs5242-hateful-memes/hateful-memes-data")
    parser.add_argument("--cache-dir", default=str(repo_root / "data" / "hf_cache"))
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        help='Dataset split names, or "all" to process every available split.',
    )
    parser.add_argument("--output", default=str(repo_root / "data" / "facebook_semantic_cues.jsonl"))
    parser.add_argument("--limit", type=int, default=None, help="Process at most N samples per split.")
    parser.add_argument("--resume", action="store_true", help="Skip split/id pairs already present in --output.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--include-text-in-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append the dataset text field after the fixed prompt as explicit visible meme text.",
    )
    return parser


def import_hf_datasets() -> Any:
    """Import Hugging Face datasets without being shadowed by local data directories."""
    script_dir = Path(__file__).resolve().parent
    original_path = list(sys.path)
    filtered_path = []

    for entry in original_path:
        entry_path = Path(entry or os.getcwd()).resolve()
        if entry_path == script_dir:
            continue
        filtered_path.append(entry)

    previous_module = sys.modules.pop("datasets", None)
    try:
        sys.path = filtered_path
        module = importlib.import_module("datasets")
        if not hasattr(module, "load_dataset"):
            raise ImportError("imported module does not expose load_dataset")
        return module
    except Exception:
        if previous_module is not None:
            sys.modules["datasets"] = previous_module
        raise
    finally:
        sys.path = original_path


def load_splits(dataset_name: str, cache_dir: str, requested_splits: list[str]) -> dict[str, Any]:
    load_dataset = import_hf_datasets().load_dataset
    ds = load_dataset(dataset_name, cache_dir=cache_dir)
    if requested_splits == ["all"]:
        return {split: ds[split] for split in ds.keys()}

    missing = [split for split in requested_splits if split not in ds]
    if missing:
        available = ", ".join(ds.keys())
        raise ValueError(f"Unknown split(s): {', '.join(missing)}. Available splits: {available}")

    return {split: ds[split] for split in requested_splits}


def load_done_keys(output_path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not output_path.exists():
        return done

    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            split = str(item.get("split", ""))
            sample_id = str(item.get("id", ""))
            if split and sample_id:
                done.add((split, sample_id))
    return done


def get_sample_id(sample: dict[str, Any], fallback_idx: int) -> str:
    for key in ("id", "idx", "index"):
        if key in sample and sample[key] is not None:
            return str(sample[key])
    return str(fallback_idx)


def get_sample_text(sample: dict[str, Any]) -> str:
    for key in ("text", "ocr_text", "caption"):
        value = sample.get(key)
        if value is not None:
            return str(value)
    return ""


def get_sample_label(sample: dict[str, Any]) -> Any:
    for key in ("label", "labels", "class"):
        if key in sample:
            return sample[key]
    return None


def save_sample_image(sample: dict[str, Any], tmpdir: Path, sample_id: str) -> str:
    image = sample.get("image") or sample.get("img")
    if image is None:
        raise ValueError("sample has no image/img field")

    if isinstance(image, str):
        return image

    safe_id = "".join(ch if ch.isalnum() else "_" for ch in sample_id)
    image_path = tmpdir / f"{safe_id}.png"
    image.save(image_path)
    return str(image_path)


def build_user_text(sample_text: str, include_text: bool) -> str:
    if not include_text:
        return SEMANTIC_CUE_PROMPT
    return f"{SEMANTIC_CUE_PROMPT}\n\nVisible meme text field:\n{sample_text if sample_text else 'unknown'}"


def generate_cues(
    image_path: str,
    sample_text: str,
    model: Any,
    processor: Any,
    max_new_tokens: int,
    temperature: float,
    include_text: bool,
) -> str:
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(Path(image_path).expanduser().resolve())},
                {"type": "text", "text": build_user_text(sample_text, include_text)},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
    )
    generated_ids = generated_ids[:, inputs.input_ids.shape[1] :]
    return processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def append_jsonl(output_path: Path, record: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def main() -> None:
    args = build_parser().parse_args()
    output_path = Path(args.output).expanduser()

    datasets_by_split = load_splits(args.dataset_name, args.cache_dir, args.splits)
    done_keys = load_done_keys(output_path) if args.resume else set()

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    print(f"[model] loading {args.model}", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model.eval()
    print("[model] ready", flush=True)

    total_seen = 0
    total_written = 0
    with tempfile.TemporaryDirectory(prefix="facebook_semantic_cues_") as tmp:
        tmpdir = Path(tmp)
        for split, dataset in datasets_by_split.items():
            split_len = len(dataset) if args.limit is None else min(len(dataset), args.limit)
            print(f"[split] {split}: {split_len} sample(s)", flush=True)
            for idx, sample in enumerate(dataset):
                if args.limit is not None and idx >= args.limit:
                    break

                sample_id = get_sample_id(sample, idx)
                sample_text = get_sample_text(sample)
                sample_label = get_sample_label(sample)
                key = (split, sample_id)
                total_seen += 1

                if key in done_keys:
                    print(f"[skip] {split}/{sample_id} already in {output_path}", flush=True)
                    continue

                print(f"[run] {split} {idx + 1}/{split_len} id={sample_id}", flush=True)
                try:
                    image_path = save_sample_image(sample, tmpdir, sample_id)
                    cues_text = generate_cues(
                        image_path=image_path,
                        sample_text=sample_text,
                        model=model,
                        processor=processor,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        include_text=args.include_text_in_prompt,
                    )
                    record = {
                        "split": split,
                        "id": sample_id,
                        "text": sample_text,
                        "label": sample_label,
                        "semantic_cues": cues_text,
                    }
                except Exception as exc:
                    record = {
                        "split": split,
                        "id": sample_id,
                        "text": sample_text,
                        "label": sample_label,
                        "error": type(exc).__name__,
                        "error_message": str(exc),
                    }
                    print(f"[error] {split}/{sample_id}: {exc}", file=sys.stderr, flush=True)

                append_jsonl(output_path, record)
                done_keys.add(key)
                total_written += 1

    print(f"[done] seen={total_seen} written={total_written} output={output_path}", flush=True)


if __name__ == "__main__":
    main()
