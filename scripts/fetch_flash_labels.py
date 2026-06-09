#!/usr/bin/env python3
"""Fetch confirmed flash labels from the export API for local training.

Mirrors the Modal `fetch_flash_images_remote` function exactly:
sharpness filter only, no embedding-based dedup.

Usage:
  LABELLING_SECRET=xxx python scripts/fetch_flash_labels.py
  LABELLING_SECRET=xxx python scripts/fetch_flash_labels.py --since 0 --max-labels 5000

Outputs:
  data/interim/confirmed_flash_images/<flash_id>.jpg   — downloaded images
  data/automation/phase_b/last_accepted.jsonl          — updated label list
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import requests
from PIL import Image

EXPORT_URL = "https://si-image-wall.pages.dev/api/export/confirmed"
IMAGES_DIR = Path("data/interim/confirmed_flash_images")
OUTPUT_JSONL = Path("data/automation/phase_b/last_accepted.jsonl")
CURSOR_FILE = Path("data/automation/phase_b/cursor.json")


def sharpness_score(path: Path) -> float:
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    lap = (
        -4.0 * arr
        + np.roll(arr, 1, axis=0)
        + np.roll(arr, -1, axis=0)
        + np.roll(arr, 1, axis=1)
        + np.roll(arr, -1, axis=1)
    )
    return float(np.var(lap)) * 100_000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", type=int, default=-1)
    parser.add_argument("--max-labels", type=int, default=5000)
    parser.add_argument("--sharpness-threshold", type=float, default=100.0)
    args = parser.parse_args()

    secret = os.environ.get("LABELLING_SECRET", "")
    if not secret:
        sys.exit("LABELLING_SECRET env var is required")

    if args.since >= 0:
        since = args.since
    elif CURSOR_FILE.exists():
        since = json.loads(CURSOR_FILE.read_text()).get("since", 0)
    else:
        since = 0

    print(f"Fetching labels since cursor={since} (max {args.max_labels})...")

    # Paginated fetch
    labels: list[dict] = []
    cursor = since
    while len(labels) < args.max_labels:
        resp = requests.get(
            EXPORT_URL,
            params={"since": cursor},
            headers={"X-Export-Secret": secret},
            timeout=30,
        )
        if resp.status_code == 401:
            sys.exit("Export unauthorized — check LABELLING_SECRET")
        resp.raise_for_status()
        payload = resp.json()
        page = payload.get("labels", [])
        labels.extend(page)
        cursor = int(payload.get("cursor", cursor))
        print(f"  page: {len(page)} labels (cursor={cursor}, total={len(labels)})")
        if len(page) < 1000:
            break
    labels = labels[: args.max_labels]
    print(f"Fetched {len(labels)} labels total")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    accepted: list[dict] = []
    rejected = 0

    for i, item in enumerate(labels, 1):
        flash_id = str(item.get("flash_id", "")).strip()
        mosaic_id = str(item.get("mosaic_id", "")).strip()
        flash_url = str(item.get("flash_url", "")).strip()
        if not flash_id or not mosaic_id or mosaic_id == "none" or not flash_url:
            rejected += 1
            continue

        # Rejected candidates = everything the model proposed that wasn't confirmed
        raw_candidates = item.get("candidates", [])
        rejected_candidates = [
            c["mosaic_id"] for c in raw_candidates
            if isinstance(c, dict) and c.get("mosaic_id") and c["mosaic_id"] != mosaic_id
        ]

        target = IMAGES_DIR / f"{flash_id}.jpg"
        if not target.exists():
            try:
                r = requests.get(flash_url, timeout=30)
                r.raise_for_status()
                target.write_bytes(r.content)
            except Exception as exc:
                print(f"  [{i}] download failed for {flash_id}: {exc}")
                rejected += 1
                continue

        try:
            score = sharpness_score(target)
            if score < args.sharpness_threshold:
                rejected += 1
                continue
        except Exception:
            rejected += 1
            continue

        city_code = mosaic_id.split("_", 1)[0]
        accepted.append({
            "city_code": city_code,
            "invader_id": mosaic_id,
            "image_path": str(target),
            "role": "flash-reference",
            "flash_id": flash_id,
            "rejected_candidates": rejected_candidates,
        })

        if i % 100 == 0:
            print(f"  {i}/{len(labels)} processed — {len(accepted)} accepted so far")

    # Merge with existing entries so incremental runs accumulate rather than overwrite.
    # Full rebuild (--since 0) still works correctly — existing entries get replaced by
    # the freshly validated ones fetched from the API.
    if OUTPUT_JSONL.exists() and since > 0:
        existing = [json.loads(l) for l in OUTPUT_JSONL.read_text().splitlines() if l.strip()]
        new_ids = {r["flash_id"] for r in accepted}
        merged = [r for r in existing if r["flash_id"] not in new_ids] + accepted
    else:
        merged = accepted

    OUTPUT_JSONL.write_text(
        "\n".join(json.dumps(r) for r in merged) + "\n", encoding="utf-8"
    )
    CURSOR_FILE.write_text(
        json.dumps({"since": cursor, "updated_at": "auto"}) + "\n", encoding="utf-8"
    )

    print(f"\nDone: {len(accepted)} accepted, {rejected} rejected ({len(merged)} total in jsonl)")
    print(f"Images: {IMAGES_DIR}")
    print(f"Labels: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
