from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from si_image_trainer.utils.io import read_jsonl, write_jsonl


def build_eval_manifest(reference_manifest_path: str | Path, output_path: str | Path) -> list[dict[str, str]]:
    rows = read_jsonl(reference_manifest_path)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["invader_id"]].append(row)

    eval_rows: list[dict[str, str]] = []
    for invader_id, group in sorted(grouped.items()):
        if len(group) < 2:
            continue
        usable = [row for row in group if row["role"] != "grosplan"] or group
        holdout = sorted(usable, key=lambda item: item["image_path"])[-1]
        eval_rows.append(
            {
                "query_id": Path(holdout["image_path"]).stem,
                "image_path": holdout["image_path"],
                "city_code": holdout["city_code"],
                "city_name": None,
                "flash_id": None,
                "player": None,
                "observed_at": None,
                "label_invader_id": invader_id,
                "split": "eval",
            }
        )

    write_jsonl(output_path, eval_rows)
    return eval_rows
