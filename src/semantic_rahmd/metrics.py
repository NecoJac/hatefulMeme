from __future__ import annotations

import torch


def auroc_score(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    """Compute AUROC without sklearn so evaluation works in minimal environments."""
    scores = scores.detach().float().cpu()
    labels = labels.detach().long().cpu()
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None

    # Rank-based AUROC with average ranks for tied prediction scores.
    order = torch.argsort(scores)
    sorted_scores = scores[order]
    ranks = torch.zeros_like(scores)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end

    pos_rank_sum = ranks[labels == 1].sum()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def binary_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float | None]:
    """Return threshold metrics plus AUROC from raw classifier logits."""
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()
    labels = labels.long()
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    total = max(1, len(labels))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "acc": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc_score(probs, labels),
    }


def knn_predict(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    query_embeddings: torch.Tensor,
    topk: int = 5,
    temperature: float = 0.07,
) -> torch.Tensor:
    # Embeddings are normalized by the model, so dot product is cosine similarity.
    sims = query_embeddings @ train_embeddings.T
    topk = min(topk, train_embeddings.shape[0])
    scores, idx = torch.topk(sims, k=topk, dim=1)
    labels = train_labels[idx].float()
    weights = torch.softmax(scores / temperature, dim=1)
    return (weights * labels).sum(dim=1)


def rkc_logits(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    query_embeddings: torch.Tensor,
    topk: int = 20,
    temperature: float = 1.0,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    # Retrieval-kernel classifier: neighbors vote with signed labels weighted by similarity.
    sims = query_embeddings @ train_embeddings.T
    topk = min(topk, train_embeddings.shape[0])
    scores, idx = torch.topk(sims, k=topk, dim=1)
    signed_labels = train_labels[idx].float().mul(2.0).sub(1.0)
    return (scores / temperature * signed_labels).sum(dim=1)
