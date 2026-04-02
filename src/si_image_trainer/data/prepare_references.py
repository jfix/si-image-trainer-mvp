from __future__ import annotations

import json
from pathlib import Path

from si_image_trainer.data.schemas import ReferenceRecord
from si_image_trainer.utils.io import write_jsonl


def build_reference_manifest(reference_root: str | Path, output_path: str | Path) -> list[dict[str, str]]:
    root = Path(reference_root)
    rows: list[dict[str, str]] = []

    for metadata_path in sorted(root.glob("*/*/metadata.json")):
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        city_code = payload["place_id"]
        invader_id = payload["invader_id"]
        status = payload.get("status", "unknown")
        for image in payload.get("images", []):
            local_path = image.get("local_path")
            if not local_path:
                continue
            image_path = root.parent / local_path
            if not image_path.exists():
                continue
            rows.append(
                ReferenceRecord(
                    city_code=city_code,
                    invader_id=invader_id,
                    image_path=str(image_path),
                    role=image.get("role", "reference"),
                    status=status,
                    source_type=image.get("type", "unknown"),
                ).to_dict()
            )

    write_jsonl(output_path, rows)
    return rows
