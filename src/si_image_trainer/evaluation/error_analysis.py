from __future__ import annotations

from collections import Counter
from pathlib import Path

from si_image_trainer.utils.io import write_json


def build_error_analysis(report: dict[str, object], report_dir: str | Path) -> dict[str, object]:
    records = report.get("records", [])
    confusion: Counter[str] = Counter()
    hard_negatives: list[dict[str, object]] = []

    for record in records:
        predicted = record["prediction"]
        actual = record["label_invader_id"]
        if predicted != actual:
            confusion[f"{actual} -> {predicted}"] += 1
            hard_negatives.append(
                {
                    "query_id": record["query_id"],
                    "actual": actual,
                    "predicted": predicted,
                    "confidence_label": record["confidence_label"],
                    "top_k": record["top_k"][:3],
                }
            )

    payload = {
        "top_confusions": [{"pair": pair, "count": count} for pair, count in confusion.most_common(20)],
        "hard_negatives": hard_negatives[:50],
    }
    write_json(Path(report_dir) / "error_analysis.json", payload)
    return payload
