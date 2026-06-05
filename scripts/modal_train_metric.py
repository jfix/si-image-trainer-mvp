#!/usr/bin/env python3
"""Modal GPU training entrypoint for metric-learning model training.

This mirrors the existing Colab-style flow:
1) package training data into outputs/colab_training_data.zip
2) upload dataset archive to a persistent Modal Volume
3) run training on Modal GPU and save model artifacts back to the Volume

Examples:
  # Build zip locally and upload to Modal volume
  modal run scripts/modal_train_metric.py::stage_data

  # Train a new experiment on Modal GPU
  modal run scripts/modal_train_metric.py::run_train --exp-name exp18-modal --epochs 20

  # List recent training runs from volume metadata
  modal run scripts/modal_train_metric.py::show_runs --limit 5
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

APP_NAME = "si-train-metric"
VOLUME_NAME = "si-training-artifacts"

VOLUME_ROOT = Path("/training")
DATASETS_DIR = VOLUME_ROOT / "datasets"
RUNS_DIR = VOLUME_ROOT / "runs"
OUTPUT_MODELS_DIR = VOLUME_ROOT / "outputs" / "models"

DATASET_ARCHIVE_NAME = "colab_training_data.zip"
DATASET_ARCHIVE_PATH = DATASETS_DIR / DATASET_ARCHIVE_NAME
DATASET_INFO_PATH = DATASETS_DIR / "dataset_info.json"
FLASH_IMAGES_DIR = VOLUME_ROOT / "flash_images"
FLASH_LABELS_PATH = DATASETS_DIR / "flash_labels.jsonl"

EXPORT_URL = "https://si-image-wall.pages.dev/api/export/confirmed"

LOCAL_DATASET_ARCHIVE = Path("outputs") / DATASET_ARCHIVE_NAME
LOCAL_PREP_SCRIPT = Path("scripts") / "prepare_colab_data.py"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "transformers==4.44.2",
        "numpy>=1.26",
        "Pillow>=10.0",
        "PyYAML>=6.0",
        "ultralytics>=8.4",
        "albumentations>=1.3",
        "requests>=2.28",
    )
    .env({"PYTHONPATH": "/app/src"})
    .add_local_dir("src/", "/app/src", copy=True)
)

app = modal.App(APP_NAME, image=image)
training_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _run_id() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _ensure_local_dataset_archive(rebuild: bool) -> None:
    if rebuild:
        if not LOCAL_PREP_SCRIPT.exists():
            raise SystemExit(f"Missing script: {LOCAL_PREP_SCRIPT}")
        python_exec = os.environ.get("PYTHON", "python")
        subprocess.run([python_exec, str(LOCAL_PREP_SCRIPT)], check=True)

    if not LOCAL_DATASET_ARCHIVE.exists():
        raise SystemExit(
            f"Dataset archive not found at {LOCAL_DATASET_ARCHIVE}. "
            "Run stage_data with --rebuild true or generate outputs/colab_training_data.zip first."
        )


@app.function(volumes={str(VOLUME_ROOT): training_volume}, timeout=60)
def list_runs(limit: int = 10) -> dict[str, Any]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_files = sorted(RUNS_DIR.glob("*.json"), reverse=True)
    selected = run_files[: max(0, limit)]

    runs: list[dict[str, Any]] = []
    for run_file in selected:
        try:
            runs.append(_read_json(run_file))
        except Exception:
            runs.append({"run_id": run_file.stem, "error": "failed to parse run summary"})

    dataset_info = _read_json(DATASET_INFO_PATH) if DATASET_INFO_PATH.exists() else None
    return {
        "app": APP_NAME,
        "volume": VOLUME_NAME,
        "dataset_info": dataset_info,
        "runs": runs,
    }


@app.function(volumes={str(VOLUME_ROOT): training_volume}, timeout=60)
def cleanup_workdirs_remote(keep_last: int = 3) -> dict[str, Any]:
    if keep_last < 0:
        raise RuntimeError("keep_last must be >= 0")

    work_root = VOLUME_ROOT / "work"
    if not work_root.exists():
        return {"deleted": []}

    run_dirs = sorted([p for p in work_root.iterdir() if p.is_dir()], reverse=True)
    to_delete = run_dirs[keep_last:]
    deleted: list[str] = []
    for path in to_delete:
        shutil.rmtree(path, ignore_errors=True)
        deleted.append(str(path))

    training_volume.commit()
    return {"deleted": deleted}


@app.function(
    gpu="T4",
    volumes={str(VOLUME_ROOT): training_volume},
    timeout=60 * 60 * 8,
)
def train_metric_remote(
    exp_name: str,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-5,
    margin: float = 0.3,
    unfreeze_last_n_blocks: int = 2,
    augment: bool = True,
    model_name: str = "facebook/dinov2-small",
) -> dict[str, Any]:
    from si_image_trainer.training.train_metric import train_metric

    if not DATASET_ARCHIVE_PATH.exists():
        raise RuntimeError(
            f"Dataset archive missing at {DATASET_ARCHIVE_PATH}. "
            "Run stage_data first."
        )

    run_id = _run_id()
    run_root = VOLUME_ROOT / "work" / run_id
    extract_root = run_root / "data"
    extract_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(DATASET_ARCHIVE_PATH, "r") as zf:
        zf.extractall(extract_root)

    source_manifest = extract_root / "reference_manifest.jsonl"
    if not source_manifest.exists():
        raise RuntimeError("reference_manifest.jsonl not found inside dataset archive")

    # Remap image paths from zip-relative layout to absolute paths inside the Modal container.
    remapped_manifest = run_root / "reference_manifest.abs.jsonl"
    image_root = extract_root / "images"
    ref_count = 0
    with source_manifest.open("r", encoding="utf-8") as src, remapped_manifest.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            rel_path = str(row.get("image_path", "")).strip()
            if not rel_path:
                continue
            abs_image_path = image_root / rel_path
            if not abs_image_path.exists():
                continue
            row["image_path"] = str(abs_image_path)
            dst.write(json.dumps(row, ensure_ascii=True) + "\n")
            ref_count += 1

    # Merge flash labels from volume if available (staged via stage_flash entrypoint).
    flash_count = 0
    if FLASH_LABELS_PATH.exists():
        with FLASH_LABELS_PATH.open("r", encoding="utf-8") as fsrc, remapped_manifest.open("a", encoding="utf-8") as dst:
            for line in fsrc:
                if not line.strip():
                    continue
                row = json.loads(line)
                img_path = Path(row.get("image_path", ""))
                if img_path.exists():
                    dst.write(json.dumps(row, ensure_ascii=True) + "\n")
                    flash_count += 1
        print(f"Merged {flash_count} flash images into manifest (ref rows: {ref_count})")
    else:
        print(f"No flash_labels.jsonl on volume — training on reference images only ({ref_count} rows)")

    output_dir = OUTPUT_MODELS_DIR / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "paths": {
            "reference_manifest": str(remapped_manifest),
        },
        "training": {
            "model_name": model_name,
            "output_dir": str(output_dir),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "margin": float(margin),
            "unfreeze_last_n_blocks": int(unfreeze_last_n_blocks),
            "augment": bool(augment),
        },
    }

    train_result = train_metric(config)

    summary: dict[str, Any] = {
        "run_id": run_id,
        "started_at_utc": _utc_now().isoformat(),
        "exp_name": exp_name,
        "dataset_archive": str(DATASET_ARCHIVE_PATH),
        "dataset_sha256": _sha256_file(DATASET_ARCHIVE_PATH),
        "model_name": model_name,
        "train_params": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "margin": float(margin),
            "unfreeze_last_n_blocks": int(unfreeze_last_n_blocks),
            "augment": bool(augment),
        },
        "output_dir": str(output_dir),
        "result": train_result,
    }

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(RUNS_DIR / f"{run_id}.json", summary)

    training_volume.commit()
    return summary


@app.function(volumes={str(VOLUME_ROOT): training_volume}, timeout=60 * 30)
def fetch_flash_images_remote(
    secret: str,
    since: int = 0,
    max_labels: int = 5000,
    sharpness_threshold: float = 100.0,
) -> dict[str, Any]:
    """Fetch crowd-confirmed flash images from the export API into the Modal volume."""
    import numpy as np
    import requests
    from PIL import Image as PILImage

    headers = {"X-Export-Secret": secret}
    labels: list[dict[str, Any]] = []
    cursor = since
    while len(labels) < max_labels:
        resp = requests.get(EXPORT_URL, params={"since": cursor}, headers=headers, timeout=30)
        if resp.status_code == 401:
            raise RuntimeError("Export unauthorized — check LABELLING_SECRET")
        resp.raise_for_status()
        payload = resp.json()
        page = payload.get("labels", [])
        labels.extend(page)
        cursor = int(payload.get("cursor", cursor))
        print(f"  Fetched page: {len(page)} labels (cursor={cursor}, total so far={len(labels)})")
        if len(page) < 1000:
            break  # last page
    labels = labels[:max_labels]
    next_cursor = cursor

    FLASH_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    FLASH_LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)

    accepted: list[dict[str, Any]] = []
    rejected = 0

    for item in labels:
        flash_id = str(item.get("flash_id", "")).strip()
        mosaic_id = str(item.get("mosaic_id", "")).strip()
        flash_url = str(item.get("flash_url", "")).strip()
        if not flash_id or not mosaic_id or mosaic_id == "none" or not flash_url:
            rejected += 1
            continue

        target = FLASH_IMAGES_DIR / f"{flash_id}.jpg"
        if not target.exists():
            try:
                r = requests.get(flash_url, timeout=30)
                r.raise_for_status()
                target.write_bytes(r.content)
            except Exception:
                rejected += 1
                continue

        try:
            img = PILImage.open(target).convert("L")
            arr = np.asarray(img, dtype=np.float32) / 255.0
            lap = (
                -4.0 * arr
                + np.roll(arr, 1, axis=0)
                + np.roll(arr, -1, axis=0)
                + np.roll(arr, 1, axis=1)
                + np.roll(arr, -1, axis=1)
            )
            sharpness = float(np.var(lap)) * 100_000.0
            if sharpness < sharpness_threshold:
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
        })

    with FLASH_LABELS_PATH.open("w", encoding="utf-8") as fout:
        for row in accepted:
            fout.write(json.dumps(row) + "\n")

    training_volume.commit()

    result = {
        "fetched": len(labels),
        "accepted": len(accepted),
        "rejected": rejected,
        "next_cursor": next_cursor,
        "flash_labels_path": str(FLASH_LABELS_PATH),
    }
    print(json.dumps(result, indent=2))
    return result


@app.local_entrypoint()
def stage_flash(since: int = 0, max_labels: int = 5000) -> None:
    """Download crowd-confirmed flash images to Modal volume.

    Requires LABELLING_SECRET env var.
    Run before stage_data + run_train to include crowd labels in training.

    Example:
      LABELLING_SECRET=xxx modal run scripts/modal_train_metric.py::stage_flash
    """
    import os
    secret = os.environ.get("LABELLING_SECRET", "")
    if not secret:
        raise SystemExit("LABELLING_SECRET env var is required")
    result = fetch_flash_images_remote.remote(secret=secret, since=since, max_labels=max_labels)
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def stage_data(rebuild: bool = True) -> None:
    """Build and upload Colab-style training dataset zip to Modal volume."""
    _ensure_local_dataset_archive(rebuild=rebuild)

    archive_sha = _sha256_file(LOCAL_DATASET_ARCHIVE)
    archive_size = LOCAL_DATASET_ARCHIVE.stat().st_size

    info_payload = {
        "uploaded_at_utc": _utc_now().isoformat(),
        "local_archive": str(LOCAL_DATASET_ARCHIVE),
        "archive_name": DATASET_ARCHIVE_NAME,
        "archive_sha256": archive_sha,
        "archive_size_bytes": archive_size,
    }

    tmp_info = Path("tmp") / "modal" / "dataset_info.json"
    tmp_info.parent.mkdir(parents=True, exist_ok=True)
    tmp_info.write_text(json.dumps(info_payload, indent=2) + "\n", encoding="utf-8")

    with training_volume.batch_upload(force=True) as upload:
        upload.put_file(str(LOCAL_DATASET_ARCHIVE), str(DATASET_ARCHIVE_PATH.relative_to(VOLUME_ROOT)))
        upload.put_file(str(tmp_info), str(DATASET_INFO_PATH.relative_to(VOLUME_ROOT)))

    print(
        json.dumps(
            {
                "app": APP_NAME,
                "volume": VOLUME_NAME,
                "uploaded": str(DATASET_ARCHIVE_PATH),
                "archive_sha256": archive_sha,
                "archive_size_bytes": archive_size,
            },
            indent=2,
        )
    )


@app.local_entrypoint()
def run_train(
    exp_name: str = "exp-modal-candidate",
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-5,
    margin: float = 0.3,
    unfreeze_last_n_blocks: int = 2,
    augment: bool = True,
    model_name: str = "facebook/dinov2-small",
) -> None:
    """Launch remote metric training on Modal GPU."""
    summary = train_metric_remote.remote(
        exp_name=exp_name,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        margin=margin,
        unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        augment=augment,
        model_name=model_name,
    )
    print(json.dumps(summary, indent=2))


@app.local_entrypoint()
def show_runs(limit: int = 10) -> None:
    """Show most recent training runs and current dataset metadata."""
    print(json.dumps(list_runs.remote(limit=limit), indent=2))


@app.local_entrypoint()
def clean_workdirs(keep_last: int = 3) -> None:
    """Remove old extracted working directories from the training volume."""
    if keep_last < 0:
        raise SystemExit("keep_last must be >= 0")

    print(json.dumps(cleanup_workdirs_remote.remote(keep_last), indent=2))
