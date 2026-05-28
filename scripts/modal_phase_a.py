#!/usr/bin/env python3
"""Phase A automation on Modal: fetch confirmed labels with cursor + stage artifacts.

This is the first production slice of corpus-refresh automation. It intentionally
avoids touching training code paths and focuses on reliable ingestion plumbing:

- authenticated pull from /api/export/confirmed using a persistent cursor
- idempotent staging of returned labels into a Modal Volume
- optional download of flash images for downstream processing
- run summaries persisted for observability/debugging

Run once:
    modal run scripts/modal_phase_a.py::run_once

Force full backfill from since=0:
    modal run scripts/modal_phase_a.py::run_once --since 0

Show current state:
    modal run scripts/modal_phase_a.py::show_state

Deploy scheduled daily job:
    modal deploy scripts/modal_phase_a.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

APP_NAME = "si-corpus-refresh"
VOLUME_NAME = "si-corpus-refresh-state"
CURSOR_STORE_NAME = "si-corpus-refresh-cursor"

BASE_URL = os.environ.get("SI_IMAGE_WALL_BASE_URL", "https://si-image-wall.pages.dev")
EXPORT_PATH = "/api/export/confirmed"
EXPORT_URL = f"{BASE_URL}{EXPORT_PATH}"

# Keep this small and cheap for daily ingestion plumbing.
MAX_DOWNLOADS_PER_RUN = int(os.environ.get("MAX_DOWNLOADS_PER_RUN", "200"))
EXPORT_PAGE_LIMIT = int(os.environ.get("EXPORT_PAGE_LIMIT", "1000"))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("requests")
)

app = modal.App(APP_NAME, image=image)
state_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
cursor_store = modal.Dict.from_name(CURSOR_STORE_NAME, create_if_missing=True)

STATE_ROOT = Path("/state")
RUNS_DIR = STATE_ROOT / "runs"
LABELS_DIR = STATE_ROOT / "labels"
FLASH_DIR = STATE_ROOT / "flash_images"


class IngestionError(RuntimeError):
    pass


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _run_id() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _ensure_dirs() -> None:
    for path in (RUNS_DIR, LABELS_DIR, FLASH_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _fetch_confirmed(*, since: int, secret: str) -> dict[str, Any]:
    import requests

    params = {"since": since}
    headers = {"X-Export-Secret": secret}
    resp = requests.get(EXPORT_URL, params=params, headers=headers, timeout=30)
    if resp.status_code == 401:
        raise IngestionError("Unauthorized export request (X-Export-Secret invalid)")
    resp.raise_for_status()

    payload = resp.json()
    labels = payload.get("labels", [])
    cursor = payload.get("cursor", since)

    if not isinstance(labels, list):
        raise IngestionError("Invalid export payload: labels is not a list")
    if not isinstance(cursor, int):
        raise IngestionError("Invalid export payload: cursor is not an integer")

    # Upstream endpoint already limits at 1000; this is a defensive bound.
    if len(labels) > EXPORT_PAGE_LIMIT:
        labels = labels[:EXPORT_PAGE_LIMIT]

    return {"labels": labels, "cursor": cursor}


def _download_flash_images(labels: list[dict[str, Any]], *, max_downloads: int) -> tuple[int, int]:
    import requests

    downloaded = 0
    failed = 0

    for label in labels[:max_downloads]:
        flash_id = str(label.get("flash_id", "")).strip()
        flash_url = str(label.get("flash_url", "")).strip()
        if not flash_id or not flash_url:
            failed += 1
            continue

        target = FLASH_DIR / f"{flash_id}.jpg"
        if target.exists():
            continue

        try:
            resp = requests.get(flash_url, timeout=30)
            resp.raise_for_status()
            target.write_bytes(resp.content)
            downloaded += 1
        except Exception:
            failed += 1

    return downloaded, failed


def _transform_labels(labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in labels:
        flash_id = str(item.get("flash_id", "")).strip()
        mosaic_id = str(item.get("mosaic_id", "")).strip()
        flash_url = str(item.get("flash_url", "")).strip()
        confirmed_at = item.get("confirmed_at")

        if not flash_id or not mosaic_id or not flash_url or not isinstance(confirmed_at, int):
            continue

        city_code = mosaic_id.split("_", 1)[0]
        local_flash_path = str(FLASH_DIR / f"{flash_id}.jpg")
        rows.append(
            {
                "flash_id": flash_id,
                "mosaic_id": mosaic_id,
                "city_code": city_code,
                "flash_url": flash_url,
                "confirmed_at": confirmed_at,
                "local_flash_path": local_flash_path,
                "role": "flash-reference",
                "status": "confirmed",
                "source_type": "crowd-confirmed-flash",
            }
        )
    return rows


@app.function(
    schedule=modal.Period(days=1),
    timeout=60 * 30,
    volumes={str(STATE_ROOT): state_volume},
    secrets=[modal.Secret.from_name("si-archive-admin")],
)
def ingest_confirmed_labels(since: int = -1, download_images: bool = True) -> dict[str, Any]:
    """Daily ingestion entrypoint.

    Args:
      since: override cursor. Use -1 to use stored cursor.
      download_images: if true, fetch flash images into the state volume.
    """
    _ensure_dirs()

    secret = os.environ.get("LABELLING_SECRET", "")
    if not secret:
        raise IngestionError("LABELLING_SECRET not set in Modal secret si-archive-admin")

    start_since = int(since) if since >= 0 else int(cursor_store.get("since", 0))
    run_id = _run_id()
    started_at = _utc_now().isoformat()

    payload = _fetch_confirmed(since=start_since, secret=secret)
    labels = payload["labels"]
    new_cursor = int(payload["cursor"])

    downloaded = 0
    download_failed = 0
    if download_images and labels:
        downloaded, download_failed = _download_flash_images(labels, max_downloads=MAX_DOWNLOADS_PER_RUN)

    transformed = _transform_labels(labels)
    labels_jsonl = LABELS_DIR / "confirmed_labels.jsonl"
    _append_jsonl(labels_jsonl, transformed)

    summary = {
        "run_id": run_id,
        "started_at_utc": started_at,
        "export_url": EXPORT_URL,
        "since": start_since,
        "cursor_before": start_since,
        "cursor_after": new_cursor,
        "fetched_labels": len(labels),
        "staged_rows": len(transformed),
        "download_images": bool(download_images),
        "downloaded_images": downloaded,
        "download_failed": download_failed,
        "labels_jsonl": str(labels_jsonl),
    }

    _write_json(RUNS_DIR / f"{run_id}.json", summary)

    # Advance cursor only after staging succeeded.
    cursor_store["since"] = new_cursor
    cursor_store["last_run"] = summary

    # Persist volume changes for future runs.
    state_volume.commit()

    return summary


@app.local_entrypoint()
def run_once(since: int = -1, download_images: bool = True):
    summary = ingest_confirmed_labels.remote(since=since, download_images=download_images)
    print(json.dumps(summary, indent=2))


@app.local_entrypoint()
def show_state():
    print(
        json.dumps(
            {
                "since": int(cursor_store.get("since", 0)),
                "last_run": cursor_store.get("last_run", None),
                "volume": VOLUME_NAME,
                "cursor_store": CURSOR_STORE_NAME,
                "export_url": EXPORT_URL,
            },
            indent=2,
        )
    )
