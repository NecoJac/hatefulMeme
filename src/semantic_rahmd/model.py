from __future__ import annotations

import hashlib
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class HashFieldEncoder(nn.Module):
    """Deterministic non-LLM encoder for smoke tests and CPU-only debugging."""

    def __init__(self, hidden_size: int = 384, num_buckets: int = 4096) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_buckets = num_buckets
        self.embedding = nn.EmbeddingBag(num_buckets, hidden_size, mode="mean")

    def _token_to_bucket(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "little") % self.num_buckets

    def forward(self, field_texts: list[list[str]]) -> torch.Tensor:
        flat_fields = [text for item in field_texts for text in item]
        indices: list[int] = []
        offsets: list[int] = []
        for text in flat_fields:
            offsets.append(len(indices))
            tokens = text.lower().split() or ["<empty>"]
            indices.extend(self._token_to_bucket(token) for token in tokens)
        device = self.embedding.weight.device
        idx = torch.tensor(indices, dtype=torch.long, device=device)
        off = torch.tensor(offsets, dtype=torch.long, device=device)
        encoded = self.embedding(idx, off)
        return encoded.view(len(field_texts), len(field_texts[0]), self.hidden_size)


class HfFieldEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        freeze: bool = True,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: list[str] | None = None,
        trust_remote_code: bool = True,
        max_length: int = 128,
        device_map: str | None = None,
        torch_dtype: str | None = None,
        fallback_hidden_size: int | None = None,
    ) -> None:
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if device_map:
            model_kwargs["device_map"] = device_map
        if torch_dtype:
            model_kwargs["torch_dtype"] = _resolve_torch_dtype(torch_dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.backbone = AutoModel.from_pretrained(model_name, **model_kwargs)
        self.max_length = max_length
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if use_lora:
            from peft import LoraConfig, get_peft_model

            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="FEATURE_EXTRACTION",
                target_modules=lora_target_modules,
            )
            self.backbone = get_peft_model(self.backbone, config)

        config = getattr(self.backbone, "config", None)
        hidden_size = _infer_hidden_size(config)
        if hidden_size is None and hasattr(self.backbone, "base_model"):
            hidden_size = _infer_hidden_size(getattr(self.backbone.base_model, "config", None))
        if hidden_size is None:
            hidden_size = fallback_hidden_size
        if hidden_size is None:
            raise ValueError("Could not infer HF hidden size from model config. Set --embedding-output-dim.")
        self.hidden_size = int(hidden_size)

    def forward(self, field_texts: list[list[str]]) -> torch.Tensor:
        flat_fields = [text for item in field_texts for text in item]
        device = next(self.backbone.parameters()).device
        encoded = self.tokenizer(
            flat_fields,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = self.backbone(**encoded)
        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        mask = encoded["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return pooled.view(len(field_texts), len(field_texts[0]), self.hidden_size)


class SentenceTransformerFieldEncoder(nn.Module):
    """Frozen SentenceTransformer encoder for embedding models such as Qwen3-VL-Embedding."""

    def __init__(
        self,
        model_name: str,
        instruction: str | None = None,
        output_dim: int | None = None,
        batch_size: int = 16,
        trust_remote_code: bool = True,
    ) -> None:
        super().__init__()
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        st_model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=trust_remote_code,
            model_kwargs={"torch_dtype": "auto"},
            tokenizer_kwargs={"padding_side": "left"},
        )
        object.__setattr__(self, "model", st_model)
        self.instruction = instruction
        self.output_dim = output_dim
        self.batch_size = batch_size

        dim = self.model.get_sentence_embedding_dimension()
        if dim is None:
            dim = output_dim
        if dim is None:
            raise ValueError("Could not infer sentence embedding dimension; set --embedding-output-dim.")
        self.hidden_size = int(min(dim, output_dim) if output_dim else dim)

        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, field_texts: list[list[str]]) -> torch.Tensor:
        flat_fields = [text for item in field_texts for text in item]
        embeddings = self.model.encode(
            flat_fields,
            prompt=self.instruction,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if self.output_dim:
            embeddings = embeddings[:, : self.output_dim]
        embeddings = embeddings.to(dtype=torch.float32)
        return embeddings.view(len(field_texts), len(field_texts[0]), self.hidden_size)


class SentenceTransformerFieldImageEncoder(SentenceTransformerFieldEncoder):
    """Frozen Qwen-VL embedding encoder for eight semantic fields plus the raw image."""

    def forward(
        self,
        field_texts: list[list[str]],
        image_paths: list[str | None] | None = None,
        images: list[Any | None] | None = None,
    ) -> torch.Tensor:
        if image_paths is None and images is None:
            raise ValueError("st_image backend requires image_paths or images in the batch.")

        flat_fields = [text for item in field_texts for text in item]
        text_embeddings = self._encode_padded(
            flat_fields,
            prompt=self.instruction,
        )

        from PIL import Image, ImageFile

        ImageFile.LOAD_TRUNCATED_IMAGES = True

        pil_images = []
        try:
            for idx in range(len(field_texts)):
                image_obj = images[idx] if images is not None else None
                path = image_paths[idx] if image_paths is not None else None
                if image_obj is not None:
                    image = image_obj.convert("RGB") if hasattr(image_obj, "convert") else Image.open(image_obj).convert("RGB")
                else:
                    try:
                        image = Image.open(str(path)).convert("RGB")
                    except Exception as exc:
                        raise OSError(f"Failed to load image for st_image backend: {path}") from exc
                pil_images.append(image)
            image_embeddings = self._encode_padded(
                pil_images,
            )
        finally:
            for image in pil_images:
                image.close()

        if self.output_dim:
            text_embeddings = text_embeddings[:, : self.output_dim]
            image_embeddings = image_embeddings[:, : self.output_dim]
        text_embeddings = text_embeddings.to(dtype=torch.float32)
        image_embeddings = image_embeddings.to(dtype=torch.float32)
        text_embeddings = text_embeddings.view(len(field_texts), len(field_texts[0]), self.hidden_size)
        return torch.cat([text_embeddings, image_embeddings.unsqueeze(1)], dim=1)

    def _encode_padded(self, inputs: list[Any], prompt: str | None = None) -> torch.Tensor:
        original_len = len(inputs)
        if original_len == 0:
            return torch.empty(0, self.hidden_size)
        padded_inputs = list(inputs)
        remainder = original_len % self.batch_size
        if remainder:
            padded_inputs.extend([inputs[-1]] * (self.batch_size - remainder))
        embeddings = self.model.encode(
            padded_inputs,
            prompt=prompt,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings[:original_len]


class FieldAttentionAggregator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        proj_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.1,
        use_residual_projection: bool = False,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, proj_dim)
        self.use_residual_projection = use_residual_projection
        if use_residual_projection:
            self.input_residual = nn.Sequential(
                nn.Linear(input_dim, proj_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(proj_dim, proj_dim),
            )
            self.input_norm = nn.LayerNorm(proj_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=num_heads,
            dim_feedforward=proj_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn_pool = nn.Linear(proj_dim, 1)
        self.norm = nn.LayerNorm(proj_dim)

    def forward(self, fields: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(fields)
        if self.use_residual_projection:
            x = self.input_norm(x + self.input_residual(fields))
        x = self.encoder(x)
        weights = torch.softmax(self.attn_pool(x).squeeze(-1), dim=-1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return self.norm(pooled), weights


class SemanticRAHMD(nn.Module):
    def __init__(
        self,
        field_encoder: nn.Module,
        field_dim: int,
        proj_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.1,
        use_residual_projection: bool = False,
    ) -> None:
        super().__init__()
        self.field_encoder = field_encoder
        self.aggregator = FieldAttentionAggregator(
            field_dim,
            proj_dim,
            num_heads,
            num_layers,
            dropout,
            use_residual_projection=use_residual_projection,
        )
        self.classifier = nn.Linear(proj_dim, 1)

    def forward(
        self,
        field_texts: list[list[str]],
        image_paths: list[str | None] | None = None,
        images: list[Any | None] | None = None,
    ) -> dict[str, Any]:
        if image_paths is None and images is None:
            field_vectors = self.field_encoder(field_texts)
        else:
            try:
                field_vectors = self.field_encoder(field_texts, image_paths=image_paths, images=images)
            except TypeError:
                field_vectors = self.field_encoder(field_texts)
        aggregator_param = next(self.aggregator.parameters())
        field_vectors = field_vectors.to(device=aggregator_param.device, dtype=aggregator_param.dtype)
        embedding, field_weights = self.aggregator(field_vectors)
        logits = self.classifier(embedding).squeeze(-1)
        return {
            "logits": logits,
            "embedding": F.normalize(embedding, p=2, dim=-1),
            "field_weights": field_weights,
        }


def build_model(args: Any) -> SemanticRAHMD:
    if args.encoder_backend == "hash":
        field_encoder = HashFieldEncoder(hidden_size=args.hash_dim)
        field_dim = args.hash_dim
    elif args.encoder_backend == "hf":
        field_encoder = HfFieldEncoder(
            model_name=args.llm_model,
            freeze=not args.train_llm,
            use_lora=args.use_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=_split_csv(args.lora_target_modules),
            trust_remote_code=args.trust_remote_code,
            device_map=args.hf_device_map,
            torch_dtype=args.hf_torch_dtype,
            fallback_hidden_size=args.embedding_output_dim,
        )
        field_dim = field_encoder.hidden_size
    elif args.encoder_backend == "hf_lora":
        field_encoder = HfFieldEncoder(
            model_name=args.llm_model,
            freeze=True,
            use_lora=True,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=_split_csv(args.lora_target_modules),
            trust_remote_code=args.trust_remote_code,
            device_map=args.hf_device_map or "auto",
            torch_dtype=args.hf_torch_dtype or "auto",
            fallback_hidden_size=args.embedding_output_dim,
        )
        field_dim = field_encoder.hidden_size
    elif args.encoder_backend == "st":
        field_encoder = SentenceTransformerFieldEncoder(
            model_name=args.llm_model,
            instruction=args.embedding_instruction,
            output_dim=args.embedding_output_dim,
            batch_size=args.encoder_batch_size,
            trust_remote_code=args.trust_remote_code,
        )
        field_dim = field_encoder.hidden_size
    elif args.encoder_backend == "st_image":
        field_encoder = SentenceTransformerFieldImageEncoder(
            model_name=args.llm_model,
            instruction=args.embedding_instruction,
            output_dim=args.embedding_output_dim,
            batch_size=args.encoder_batch_size,
            trust_remote_code=args.trust_remote_code,
        )
        field_dim = field_encoder.hidden_size
    else:
        raise ValueError(f"Unknown encoder backend: {args.encoder_backend}")

    return SemanticRAHMD(
        field_encoder=field_encoder,
        field_dim=field_dim,
        proj_dim=args.proj_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_residual_projection=getattr(args, "use_residual_projection", False),
    )


def _split_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _infer_hidden_size(config: Any) -> int | None:
    if config is None:
        return None
    for attr in ("hidden_size", "d_model", "n_embd", "embed_dim", "embedding_size"):
        value = getattr(config, attr, None)
        if isinstance(value, int):
            return value
    for attr in ("text_config", "language_config", "llm_config", "model_config"):
        nested = getattr(config, attr, None)
        value = _infer_hidden_size(nested)
        if value is not None:
            return value
    return None


def _resolve_torch_dtype(value: str) -> torch.dtype | str:
    normalized = value.strip().lower()
    if normalized == "auto":
        return "auto"
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported --hf-torch-dtype: {value}")
