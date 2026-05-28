#!/usr/bin/env python3
"""Phase B corpus refresh orchestration (local/CI runner).

This script keeps training untouched and automates daily corpus refresh steps:

1) Pull crowd-confirmed labels from image-wall export API with a persistent cursor.
2) Download flash images for new labels.
3) Apply quality filters (sharpness) before embedding.
4) Deduplicate within the same mosaic_id using embedding cosine similarity.
5) Build a merged reference manifest (base refs + accepted flash references).
6) Optionally build versioned city indexes and atomically flip a local pointer file.

Usage:
  python scripts/phase_b_refresh.py --config configs/base.yaml
  python scripts/phase_b_refresh.py --config configs/base.yaml --build-index
  python scripts/phase_b_refresh.py --config configs/base.yaml --since 0 --max-labels 200
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

from si_image_trainer.indexing.build_index import build_indexes
from si_image_trainer.models.embedder import make_embedder
from si_image_trainer.utils.io import ensure_parent, load_yaml, read_jsonl, write_json, write_jsonl


EXPORT_URL = "https://si-image-wall.pages.dev/api/export/confirmed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase B corpus refresh")
    parser.add_argument("--config", required=True, help="Path to YAML config (e.g. configs/base.yaml)")
    parser.add_argument("--since", type=int, default=-1, help="Override cursor value (default: use saved cursor)")
    parser.add_argument("--max-labels", type=int, default=1000, help="Max labels to process per run")
    parser.add_argument("--sharpness-threshold", type=float, default=100.0, help="Min sharpness to accept a flash")
    parser.add_argument("--dedup-threshold", type=float, default=0.95, help="Cosine threshold for same-mosaic dedup")
    parser.add_argument("--build-index", action="store_true", help="Build versioned city indexes after manifest merge")
    parser.add_argument("--allow-insecure-export", action="store_true", help="Allow running without LABELLING_SECRET (for local mock testing)")
    parser.add_argument("--labels-file", default="", help="Optional JSON file with {labels:[...], cursor:int}; bypasses export API")
    return parser.parse_args()


@dataclass
class RunPaths:
    root: Path
    cursor_json: Path
    reports_dir: Path
    downloads_dir: Path
    merged_manifest: Path
    rejects_jsonl: Path
    accepted_jsonl: Path
    indexes_root: Path
    indexes_pointer: Path


@dataclass
class LabelItem:
    flash_id: str
    mosaic_id: str
    flash_url: str
    confirmed_at: int


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def run_id() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ")


def build_paths() -> RunPaths:
    root = Path("data/automation/phase_b")
    return RunPaths(
        root=root,
        cursor_json=root / "cursor.json",
        reports_dir=root / "reports",
        downloads_dir=Path("data/interim/confirmed_flash_images"),
        merged_manifest=Path("data/processed/reference_manifest_phase_b.jsonl"),
        rejects_jsonl=root / "last_rejects.jsonl",
        accepted_jsonl=root / "last_accepted.jsonl",
        indexes_root=Path("outputs/indexes_versions"),
        indexes_pointer=Path("outputs/indexes_current.json"),
    )


def read_cursor(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = int(payload.get("since", 0))
        return max(0, value)
    except Exception:
        return 0


def write_cursor(path: Path, since: int, last_run_id: str) -> None:
    ensure_parent(path)
    write_json(path, {"since": int(since), "updated_at_utc": utc_now().isoformat(), "last_run_id": last_run_id})


def fetch_confirmed(*, since: int, secret: str) -> tuple[list[LabelItem], int]:
    headers = {"X-Export-Secret": secret}
    resp = requests.get(EXPORT_URL, params={"since": since}, headers=headers, timeout=30)
    if resp.status_code == 401:
        raise RuntimeError("Export unauthorized: LABELLING_SECRET does not match image-wall")
    resp.raise_for_status()
    payload = resp.json()

    raw_labels = payload.get("labels", [])
    next_cursor = int(payload.get("cursor", since))

    labels: list[LabelItem] = []
    for row in raw_labels:
        flash_id = str(row.get("flash_id", "")).strip()
        mosaic_id = str(row.get("mosaic_id", "")).strip()
        flash_url = str(row.get("flash_url", "")).strip()
        confirmed_at = row.get("confirmed_at")
        if not flash_id or not mosaic_id or not flash_url or not isinstance(confirmed_at, int):
            continue
        labels.append(LabelItem(flash_id=flash_id, mosaic_id=mosaic_id, flash_url=flash_url, confirmed_at=confirmed_at))

    return labels, next_cursor


def download_flash(label: LabelItem, downloads_dir: Path) -> Path:
    ensure_parent(downloads_dir / "placeholder")
    target = downloads_dir / f"{label.flash_id}.jpg"
    if target.exists():
        return target

    resp = requests.get(label.flash_url, timeout=30)
    resp.raise_for_status()
    target.write_bytes(resp.content)
    return target


def sharpness_score(path: Path) -> float:
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0

    # Discrete Laplacian approximation; variance is a robust sharpness proxy.
    lap = (
        -4.0 * arr
        + np.roll(arr, 1, axis=0)
        + np.roll(arr, -1, axis=0)
        + np.roll(arr, 1, axis=1)
        + np.roll(arr, -1, axis=1)
    )
    return float(np.var(lap)) * 100000.0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def build_mosaic_image_map(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for row in rows:
        invader_id = str(row.get("invader_id", ""))
        path = str(row.get("image_path", ""))
        if invader_id and path:
            result.setdefault(invader_id, []).append(path)
    return result


def accept_row(label: LabelItem, local_image: Path) -> dict[str, Any]:
    city_code = label.mosaic_id.split("_", 1)[0]
    return {
        "city_code": city_code,
        "invader_id": label.mosaic_id,
        "image_path": str(local_image),
        "role": "flash-reference",
        "status": "confirmed",
        "source_type": "crowd-confirmed-flash",
        "flash_id": label.flash_id,
        "confirmed_at": label.confirmed_at,
    }


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    paths = build_paths()

    current_since = args.since if args.since >= 0 else read_cursor(paths.cursor_json)
    labels_file = Path(args.labels_file) if args.labels_file else None

    if labels_file:
        payload = json.loads(labels_file.read_text(encoding="utf-8"))
        raw_labels = payload.get("labels", [])
        labels = []
        for row in raw_labels:
            flash_id = str(row.get("flash_id", "")).strip()
            mosaic_id = str(row.get("mosaic_id", "")).strip()
            flash_url = str(row.get("flash_url", "")).strip()
            confirmed_at = row.get("confirmed_at")
            if not flash_id or not mosaic_id or not flash_url or not isinstance(confirmed_at, int):
                continue
            labels.append(LabelItem(flash_id=flash_id, mosaic_id=mosaic_id, flash_url=flash_url, confirmed_at=confirmed_at))
        next_cursor = int(payload.get("cursor", current_since))
    else:
        secret = os.environ.get("LABELLING_SECRET", "")
        if not secret and not args.allow_insecure_export:
            raise RuntimeError("LABELLING_SECRET is required (or use --allow-insecure-export for local mocks)")
        labels, next_cursor = fetch_confirmed(since=current_since, secret=secret)

    labels = labels[: max(0, args.max_labels)]

    base_manifest_path = Path(cfg["paths"]["reference_manifest"])
    base_rows = read_jsonl(base_manifest_path)
    base_map = build_mosaic_image_map(base_rows)

    embedder = make_embedder(cfg["embedding"])
    embed_cache: dict[str, np.ndarray] = {}

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_by_mosaic_sha: set[tuple[str, str]] = set()

    def get_embed(path_str: str) -> np.ndarray:
        if path_str not in embed_cache:
            embed_cache[path_str] = embedder.embed_path(path_str)
        return embed_cache[path_str]

    for label in labels:
        try:
            local_path = download_flash(label, paths.downloads_dir)
        except Exception as exc:
            rejected.append({"flash_id": label.flash_id, "mosaic_id": label.mosaic_id, "reason": f"download_failed:{exc}"})
            continue

        try:
            sharpness = sharpness_score(local_path)
        except Exception as exc:
            rejected.append({"flash_id": label.flash_id, "mosaic_id": label.mosaic_id, "reason": f"sharpness_failed:{exc}"})
            continue

        if sharpness < args.sharpness_threshold:
            rejected.append({
                "flash_id": label.flash_id,
                "mosaic_id": label.mosaic_id,
                "reason": "low_sharpness",
                "sharpness": round(sharpness, 2),
            })
            continue

        digest = file_sha256(local_path)
        key = (label.mosaic_id, digest)
        if key in seen_by_mosaic_sha:
            rejected.append({"flash_id": label.flash_id, "mosaic_id": label.mosaic_id, "reason": "duplicate_digest_same_run"})
            continue

        try:
            candidate_vec = get_embed(str(local_path))
            compare_paths = list(base_map.get(label.mosaic_id, [])) + [str(r["image_path"]) for r in accepted if r["invader_id"] == label.mosaic_id]
            max_sim = -1.0
            for ref_path in compare_paths:
                ref = Path(ref_path)
                if not ref.exists():
                    continue
                sim = cosine(candidate_vec, get_embed(ref_path))
                if sim > max_sim:
                    max_sim = sim
            if max_sim > args.dedup_threshold:
                rejected.append({
                    "flash_id": label.flash_id,
                    "mosaic_id": label.mosaic_id,
                    "reason": "near_duplicate_same_mosaic",
                    "max_cosine": round(max_sim, 4),
                })
                continue
        except Exception as exc:
            rejected.append({"flash_id": label.flash_id, "mosaic_id": label.mosaic_id, "reason": f"dedup_failed:{exc}"})
            continue

        seen_by_mosaic_sha.add(key)
        row = accept_row(label, local_path)
        accepted.append(row)

    merged_rows = [*base_rows, *accepted]
    write_jsonl(paths.merged_manifest, merged_rows)
    write_jsonl(paths.accepted_jsonl, accepted)
    write_jsonl(paths.rejects_jsonl, rejected)

    run = run_id()
    summary: dict[str, Any] = {
        "run_id": run,
        "started_since": current_since,
        "next_cursor": next_cursor,
        "fetched": len(labels),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "sharpness_threshold": args.sharpness_threshold,
        "dedup_threshold": args.dedup_threshold,
        "merged_manifest": str(paths.merged_manifest),
        "accepted_jsonl": str(paths.accepted_jsonl),
        "rejects_jsonl": str(paths.rejects_jsonl),
        "index_build": bool(args.build_index),
        "built_indexes": None,
        "index_version_dir": None,
        "index_pointer": str(paths.indexes_pointer),
    }

    if args.build_index:
        version_dir = paths.indexes_root / run
        ensure_parent(version_dir / "placeholder")

        rows = build_indexes(
            reference_manifest_path=paths.merged_manifest,
            output_dir=version_dir,
            embedding_config=cfg["embedding"],
            retrieval_config=cfg.get("retrieval"),
            exclude_manifest_path=cfg["paths"].get("eval_manifest"),
            detector_config=cfg.get("detector"),
        )
        summary["built_indexes"] = rows
        summary["index_version_dir"] = str(version_dir)

        pointer_payload = {
            "active": str(version_dir),
            "updated_at_utc": utc_now().isoformat(),
            "run_id": run,
        }
        write_json(paths.indexes_pointer, pointer_payload)

    ensure_parent(paths.reports_dir / "placeholder")
    write_json(paths.reports_dir / f"{run}.json", summary)

    # Commit cursor only after outputs were persisted.
    write_cursor(paths.cursor_json, next_cursor, run)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
