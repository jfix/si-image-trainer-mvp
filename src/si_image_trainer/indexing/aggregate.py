from __future__ import annotations

from collections import defaultdict


def aggregate_hits(hits: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for hit in hits:
        grouped[str(hit["invader_id"])].append(hit)

    rows: list[dict[str, object]] = []
    for invader_id, group in grouped.items():
        scores = [float(item["score"]) for item in group]
        rows.append(
            {
                "invader_id": invader_id,
                "city_code": group[0]["city_code"],
                "best_score": max(scores),
                "mean_score": sum(scores) / len(scores),
                "support": len(group),
                "image_hits": group,
                "aggregate_score": max(scores) * 0.75 + (sum(scores) / len(scores)) * 0.25,
            }
        )
    return sorted(rows, key=lambda row: (-float(row["aggregate_score"]), -float(row["best_score"])))
