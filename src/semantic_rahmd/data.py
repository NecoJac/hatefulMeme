from __future__ import annotations

import json
import importlib
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset


SEMANTIC_FIELDS = [
    "GlobalDescription",
    "TargetCandidateType",
    "TargetCandidate",
    "ProtectedTargetPossible",
    "ProtectedTargetType",
    "RelationOrAction",
    "SafetyReasonCode",
]


@dataclass(frozen=True)
class MemeRecord:
    sample_id: str
    split: str
    label: int
    fields: list[str]
    image_path: str | None = None
    image: Any | None = None


def parse_semantic_cues(cues: str) -> dict[str, str]:
    parsed = {field: "unknown" for field in SEMANTIC_FIELDS}
    field_pattern = re.compile(rf"(?<!\w)({'|'.join(re.escape(field) for field in SEMANTIC_FIELDS)}):")
    matches = list(field_pattern.finditer(cues))
    for idx, match in enumerate(matches):
        field = match.group(1)
        value_start = match.end()
        value_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(cues)
        value = cues[value_start:value_end].strip()
        value = re.sub(r"\s+", " ", value)
        parsed[field] = value if value else "unknown"
    return parsed


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
    return records


def records_from_jsonl(
    path: str | Path,
    skip_errors: bool = True,
    image_dir: str | Path | None = None,
    dataset_name: str | None = None,
    cache_dir: str | Path | None = None,
    use_hf_images: bool = False,
) -> list[MemeRecord]:
    records: list[MemeRecord] = []
    image_root = Path(image_dir).expanduser() if image_dir is not None else None
    image_index = build_image_index(image_root)
    hf_images = load_hf_image_lookup(dataset_name, cache_dir) if use_hf_images and dataset_name else {}
    for item in load_jsonl(path):
        if skip_errors and ("error" in item or "semantic_cues" not in item):
            continue
        label = item.get("label")
        if label is None:
            continue

        parsed = parse_semantic_cues(str(item.get("semantic_cues", "")))
        fields = [str(item.get("text", ""))]
        fields.extend(parsed[field] for field in SEMANTIC_FIELDS)
        sample_id = str(item.get("id", len(records)))
        split = str(item.get("split", "train"))
        image_path = resolve_image_path(item, sample_id, image_root, image_index)
        image = hf_images.get((split, sample_id))
        records.append(
            MemeRecord(
                sample_id=sample_id,
                split=split,
                label=int(label),
                fields=fields,
                image_path=image_path,
                image=image,
            )
        )
    if not records:
        raise ValueError(f"No usable records found in {path}")
    return records


def import_hf_datasets() -> Any:
    original_path = list(sys.path)
    filtered_path = []
    for entry in original_path:
        entry_path = Path(entry or os.getcwd()).resolve()
        if entry_path.name == "hateMM":
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


def load_hf_image_lookup(dataset_name: str | None, cache_dir: str | Path | None) -> dict[tuple[str, str], Any]:
    if dataset_name is None:
        return {}
    load_dataset = import_hf_datasets().load_dataset
    kwargs = {"cache_dir": str(Path(cache_dir).expanduser())} if cache_dir is not None else {}
    dataset_dict = load_dataset(dataset_name, **kwargs)
    lookup: dict[tuple[str, str], Any] = {}
    for split, dataset in dataset_dict.items():
        for idx, sample in enumerate(dataset):
            sample_id = get_sample_id(sample, idx)
            image = sample.get("image") or sample.get("img")
            if image is not None:
                lookup[(str(split), sample_id)] = image
    return lookup


def get_sample_id(sample: dict[str, Any], fallback_idx: int) -> str:
    for key in ("id", "idx", "index"):
        if key in sample and sample[key] is not None:
            return str(sample[key])
    return str(fallback_idx)


def build_image_index(image_root: Path | None) -> dict[str, str]:
    if image_root is None or not image_root.exists():
        return {}
    return {
        path.stem.lower(): str(path)
        for path in image_root.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    }


def resolve_image_path(
    item: dict,
    sample_id: str,
    image_root: Path | None,
    image_index: dict[str, str] | None = None,
) -> str | None:
    for key in ("image_path", "img_path", "path", "image"):
        value = item.get(key)
        if isinstance(value, str) and value:
            path = Path(value).expanduser()
            if path.exists():
                return str(path)
            if image_root is not None:
                candidate = image_root / path.name
                if candidate.exists():
                    return str(candidate)

    if image_root is None:
        return None
    if image_index:
        indexed = image_index.get(sample_id.lower())
        if indexed is not None:
            return indexed
    for suffix in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = image_root / f"{sample_id}{suffix}"
        if candidate.exists():
            return str(candidate)
    return str(image_root / f"{sample_id}.png")


def normalize_split_names(value: str | Iterable[str]) -> set[str]:
    if isinstance(value, str):
        return {value}
    return {str(item) for item in value}


def split_records(
    records: list[MemeRecord],
    train_split: str = "train",
    dev_split: str | Iterable[str] = "dev_seen",
    test_split: str | Iterable[str] = ("test_seen", "test_unseen"),
    val_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[list[MemeRecord], list[MemeRecord], list[MemeRecord] | None]:
    dev_splits = normalize_split_names(dev_split)
    test_splits = normalize_split_names(test_split)
    train = [r for r in records if r.split == train_split]
    dev = [r for r in records if r.split in dev_splits]
    test = [r for r in records if r.split in test_splits]

    if not train:
        train = [r for r in records if r.split not in dev_splits | test_splits]
    if not dev:
        shuffled = list(train)
        rng = random.Random(seed)
        rng.shuffle(shuffled)
        n_dev = max(1, int(len(shuffled) * val_fraction))
        dev = shuffled[:n_dev]
        dev_ids = {r.sample_id for r in dev}
        train = [r for r in shuffled[n_dev:] if r.sample_id not in dev_ids]
    return train, dev, test or None


def records_for_split(records: list[MemeRecord], split: str) -> list[MemeRecord]:
    return [record for record in records if record.split == split]


class SemanticCueDataset(Dataset):
    def __init__(self, records: Iterable[MemeRecord]) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> MemeRecord:
        return self.records[idx]


def collate_semantic_batch(batch: list[MemeRecord]) -> dict[str, object]:
    return {
        "ids": [item.sample_id for item in batch],
        "fields": [item.fields for item in batch],
        "image_paths": [item.image_path for item in batch],
        "images": [item.image for item in batch],
        "labels": torch.tensor([item.label for item in batch], dtype=torch.float32),
    }
