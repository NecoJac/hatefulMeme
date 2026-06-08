from __future__ import annotations

import torch
import torch.nn.functional as F


def lcl_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    memory_embeddings: torch.Tensor | None = None,
    memory_labels: torch.Tensor | None = None,
    anchor_ids: list[str] | None = None,
    memory_ids: list[str] | None = None,
) -> torch.Tensor:
    """Hard positive/negative contrastive loss over current batch or train memory."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    labels = labels.long()
    if memory_embeddings is None:
        memory_embeddings = embeddings.detach()
    if memory_labels is None:
        memory_labels = labels.detach()

    memory_embeddings = memory_embeddings.to(embeddings.device)
    memory_labels = memory_labels.to(embeddings.device).long()
    # Similarity search is over normalized meme embeddings.
    sim = embeddings @ memory_embeddings.T
    positive_mask = labels[:, None].eq(memory_labels[None, :])
    negative_mask = labels[:, None].ne(memory_labels[None, :])

    if anchor_ids is not None and memory_ids is not None:
        self_mask = torch.tensor(
            [[anchor_id == memory_id for memory_id in memory_ids] for anchor_id in anchor_ids],
            dtype=torch.bool,
            device=embeddings.device,
        )
        positive_mask = positive_mask & ~self_mask
    elif memory_embeddings.shape[0] == embeddings.shape[0]:
        self_mask = torch.eye(embeddings.shape[0], dtype=torch.bool, device=embeddings.device)
        positive_mask = positive_mask & ~self_mask

    valid = positive_mask.any(dim=1) & negative_mask.any(dim=1)
    if not valid.any():
        return embeddings.sum() * 0.0

    # Match the RA-HMD-style objective: nearest same-label positive vs nearest different-label negative.
    masked_pos = sim.masked_fill(~positive_mask, float("-inf"))
    masked_neg = sim.masked_fill(~negative_mask, float("-inf"))
    pos_sim = masked_pos.max(dim=1).values[valid] / temperature
    neg_sim = masked_neg.max(dim=1).values[valid] / temperature
    logits = torch.stack([pos_sim, neg_sim], dim=1)
    targets = torch.zeros(logits.shape[0], dtype=torch.long, device=embeddings.device)
    return F.cross_entropy(logits, targets)


def rahmd_loss(
    logits: torch.Tensor,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    contrastive_weight: float = 1.0,
    temperature: float = 1.0,
    pos_weight: float | None = None,
    memory_embeddings: torch.Tensor | None = None,
    memory_labels: torch.Tensor | None = None,
    anchor_ids: list[str] | None = None,
    memory_ids: list[str] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Joint objective: direct classifier BCE plus retrieval-friendly contrastive loss."""
    weight_tensor = None
    if pos_weight is not None:
        weight_tensor = torch.tensor([pos_weight], device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, labels.float(), pos_weight=weight_tensor)
    lcl = lcl_loss(
        embeddings,
        labels,
        temperature,
        memory_embeddings=memory_embeddings,
        memory_labels=memory_labels,
        anchor_ids=anchor_ids,
        memory_ids=memory_ids,
    )
    total = bce + contrastive_weight * lcl
    return total, {"bce": float(bce.detach()), "lcl": float(lcl.detach())}
