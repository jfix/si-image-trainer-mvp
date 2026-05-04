from __future__ import annotations

from collections import defaultdict


def aggregate_hits(
    hits: list[dict[str, object]],
    role_weights: dict[str, float] | None = None,
) -> list[dict[str, object]]:
    weights = {"reference": 1.0, "flash-reference": 0.97, "grosplan": 0.82}
    if role_weights:
        weights.update({str(key): float(value) for key, value in role_weights.items()})

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for hit in hits:
        grouped[str(hit["invader_id"])].append(hit)

    rows: list[dict[str, object]] = []
    for invader_id, group in grouped.items():
        scores = [float(item["score"]) for item in group]
        weighted_scores = [float(item["score"]) * weights.get(str(item.get("role", "reference")), 0.75) for item in group]
        rows.append(
            {
                "invader_id": invader_id,
                "city_code": group[0]["city_code"],
                "best_score": max(scores),
                "weighted_best_score": max(weighted_scores),
                "mean_score": sum(scores) / len(scores),
                "weighted_mean_score": sum(weighted_scores) / len(weighted_scores),
                "support": len(group),
                "image_hits": group,
                "aggregate_score": max(weighted_scores) * 0.75 + (sum(weighted_scores) / len(weighted_scores)) * 0.25,
            }
        )
    return sorted(rows, key=lambda row: (-float(row["aggregate_score"]), -float(row["best_score"])))
