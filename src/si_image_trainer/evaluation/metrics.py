from __future__ import annotations

from collections import Counter


def summarize_predictions(records: list[dict[str, object]], top_k: int = 5) -> dict[str, object]:
    total = len(records)
    top1 = 0
    topk = 0
    reciprocal_rank = 0.0
    confidence_counts: Counter[str] = Counter()

    for record in records:
        label = record["label_invader_id"]
        ranked = [row["invader_id"] for row in record["top_k"]]
        confidence_counts[str(record["confidence_label"])] += 1
        if ranked and ranked[0] == label:
            top1 += 1
        if label in ranked[:top_k]:
            topk += 1
        if label in ranked:
            reciprocal_rank += 1.0 / (ranked.index(label) + 1)

    if total == 0:
        return {"count": 0, "top1_accuracy": 0.0, "topk_accuracy": 0.0, "mrr": 0.0, "confidence_counts": {}}
    return {
        "count": total,
        "top1_accuracy": round(top1 / total, 4),
        "topk_accuracy": round(topk / total, 4),
        "mrr": round(reciprocal_rank / total, 4),
        "confidence_counts": dict(confidence_counts),
    }
