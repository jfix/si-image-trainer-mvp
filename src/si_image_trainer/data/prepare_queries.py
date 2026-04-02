from __future__ import annotations

import csv
import json
from pathlib import Path

from si_image_trainer.data.schemas import QueryRecord
from si_image_trainer.utils.io import write_jsonl


def load_label_manifest(path: str | Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    resolved = Path(path)
    if not resolved.exists():
        return {}
    if resolved.suffix.lower() == ".json":
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        return {row["query_id"]: row for row in payload}
    if resolved.suffix.lower() == ".jsonl":
        return {
            row["query_id"]: row
            for row in (json.loads(line) for line in resolved.read_text(encoding="utf-8").splitlines() if line.strip())
        }
    if resolved.suffix.lower() == ".csv":
        with resolved.open("r", encoding="utf-8") as handle:
            return {row["query_id"]: row for row in csv.DictReader(handle)}
    raise ValueError(f"Unsupported label manifest: {resolved}")


def infer_observed_at(image_path: Path, live_root: Path) -> str | None:
    try:
        relative = image_path.relative_to(live_root)
    except ValueError:
        return None
    parts = relative.parts
    if len(parts) < 4:
        return None
    return "-".join(parts[:3])


def load_place_mapping(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    resolved = Path(path)
    if not resolved.exists():
        return {}
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for city_code, row in payload.items():
        name = row.get("name")
        variation = row.get("variation")
        if name:
            mapping[str(name)] = city_code
        if variation:
            mapping[str(variation)] = city_code
    return mapping


def load_live_events(data_root: str | Path | None) -> dict[str, dict[str, str]]:
    if data_root is None:
        return {}
    root = Path(data_root)
    if not root.exists():
        return {}
    rows: dict[str, dict[str, str]] = {}
    for ndjson_path in sorted(root.glob("*/*.ndjson")):
        for line in ndjson_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            relative_image = event.get("img")
            if not relative_image:
                continue
            rows[str(relative_image)] = event
    return rows


def build_query_manifest(
    live_root: str | Path,
    output_path: str | Path,
    label_manifest_path: str | Path | None = None,
    data_root: str | Path | None = None,
    place_mapping_path: str | Path | None = None,
) -> list[dict[str, str | None]]:
    root = Path(live_root)
    labels = load_label_manifest(label_manifest_path)
    events = load_live_events(data_root)
    place_mapping = load_place_mapping(place_mapping_path)
    rows: list[dict[str, str | None]] = []
    for image_path in sorted(root.rglob("*")):
        if not image_path.is_file():
            continue
        if image_path.name.startswith("."):
            continue
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        query_id = image_path.stem
        label = labels.get(query_id, {})
        relative_image = str(image_path.relative_to(root)).replace("\\", "/")
        event = events.get(relative_image, {})
        city_name = event.get("city") or label.get("city_name")
        city_code = label.get("city_code") or (place_mapping.get(city_name) if city_name else None)
        rows.append(
            QueryRecord(
                query_id=str(event.get("flash_id", query_id)),
                image_path=str(image_path),
                city_code=city_code,
                city_name=city_name,
                flash_id=str(event["flash_id"]) if "flash_id" in event else None,
                player=event.get("player"),
                observed_at=infer_observed_at(image_path, root),
                label_invader_id=label.get("label_invader_id"),
                split=label.get("split", "inference"),
            ).to_dict()
        )
    write_jsonl(output_path, rows)
    return rows
