from __future__ import annotations

from pathlib import Path

from si_image_trainer.utils.io import write_json


def evaluate_calibration(report: dict[str, object], report_dir: str | Path) -> dict[str, object]:
    records = report.get("records", [])
    buckets = {"certainly": {"count": 0, "correct": 0}, "probably": {"count": 0, "correct": 0}, "maybe": {"count": 0, "correct": 0}, "unknown": {"count": 0, "correct": 0}}
    for record in records:
        bucket = str(record["confidence_label"])
        buckets[bucket]["count"] += 1
        if record["prediction"] == record["label_invader_id"]:
            buckets[bucket]["correct"] += 1

    payload = {
        label: {
            "count": row["count"],
            "accuracy": round((row["correct"] / row["count"]) if row["count"] else 0.0, 4),
        }
        for label, row in buckets.items()
    }
    write_json(Path(report_dir) / "calibration.json", payload)
    return payload
