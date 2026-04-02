from __future__ import annotations

from pathlib import Path

from si_image_trainer.utils.io import read_jsonl


def duplicate_image_paths(manifest_path: str | Path) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for row in read_jsonl(manifest_path):
        image_path = row["image_path"]
        if image_path in seen:
            duplicates.append(image_path)
        seen.add(image_path)
    return duplicates
