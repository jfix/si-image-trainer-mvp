#!/usr/bin/env python3
"""Modal inference service — top-K mosaic retrieval from a flash image URL.

Deploy:
    modal deploy scripts/modal_inference.py

Upload model + indexes to the Modal Volume (run once, then after each new model):
    modal run scripts/modal_inference.py::upload_data

Invoke locally for testing:
    modal run scripts/modal_inference.py::test_predict --city PA --image-url "https://..."
"""
import io
import hashlib
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone
from urllib import error as urlerror
from urllib import request as urlrequest
from typing import Optional

import modal

# ── Volume & image ────────────────────────────────────────────────────────────

VOLUME_NAME = "si-inference-data"
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

DATA_DIR   = Path("/data")
MODEL_DIR  = DATA_DIR / "model"
INDEX_DIR  = DATA_DIR / "indexes"
DETECTOR_PATH = DATA_DIR / "detector" / "mosaic_detector_v3.pt"
MANIFEST_PATH = DATA_DIR / "metadata" / "deployment-manifest.json"

LOCAL_MODEL_DIR    = Path("outputs/models/exp20")
LOCAL_INDEX_DIR    = Path("outputs/indexes")
LOCAL_DETECTOR_PATH = Path("outputs/models/mosaic_detector_v3.pt")
LOCAL_MANIFEST_PATH = Path("tmp/modal/deployment-manifest.json")
IMAGE_WALL_META_ENDPOINT = os.getenv("IMAGE_WALL_META_ENDPOINT", "https://si-image-wall.pages.dev/api/model-meta")
IMAGE_WALL_META_SECRET = os.getenv("IMAGE_WALL_META_SECRET", "")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "transformers==4.44.2",
        "numpy>=1.26",
        "Pillow>=10.0",
        "ultralytics>=8.4",
        "requests",
        "fastapi[standard]",
    )
    .env({"PYTHONPATH": "/app/src"})
    .add_local_dir("src/", "/app/src", copy=True)
)

app = modal.App("si-inference", image=image)

# ── Inference service ─────────────────────────────────────────────────────────

@app.cls(
    gpu="T4",
    volumes={str(DATA_DIR): vol},
    min_containers=0,
    max_containers=3,
    timeout=60,
)
class InferenceService:
    @modal.enter()
    def load(self):
        from si_image_trainer.inference.pipeline import RetrievalPipeline
        self.deployment_manifest = {}
        if MANIFEST_PATH.exists():
            self.deployment_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

        self.pipeline = RetrievalPipeline(
            index_dir=str(INDEX_DIR),
            embedding_config={"type": "pretrained", "model_name": str(MODEL_DIR)},
            retrieval_config={
                "top_n": 20,
                "rerank_m": 5,
                "aggregate_method": "max_mean",
                "role_weights": {"reference": 1.0, "flash-reference": 0.92, "grosplan": 0.82},
            },
            confidence_config={
                "certainly_score": 0.59,
                "probably_score": 0.53,
                "maybe_score": 0.48,
                "unknown_score": 0.44,
                "margin_threshold": 0.15,
            },
            detector_config={"model_path": str(DETECTOR_PATH), "conf": 0.25} if DETECTOR_PATH.exists() else None,
        )

    @modal.fastapi_endpoint(method="POST", label="si-predict")
    def predict_endpoint(self, body: dict) -> dict:
        return self._predict(body["city"], body["image_url"], int(body.get("top_k", 5)))

    @modal.fastapi_endpoint(method="GET", label="si-meta")
    def deployment_info_endpoint(self) -> dict:
        return self._deployment_info()

    @modal.method()
    def predict(self, city: str, image_url: str, top_k: int = 5) -> dict:
        return self._predict(city, image_url, top_k)

    @modal.method()
    def deployment_info(self) -> dict:
        return self._deployment_info()

    def _deployment_info(self) -> dict:
        return {
            "app": "si-inference",
            "manifest_path": str(MANIFEST_PATH),
            "manifest": self.deployment_manifest,
        }

    def _predict(self, city: str, image_url: str, top_k: int) -> dict:
        import requests
        import tempfile

        resp = requests.get(image_url, timeout=10)
        resp.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(resp.content)
            tmp_path = f.name

        result = self.pipeline.predict(tmp_path, city)
        Path(tmp_path).unlink(missing_ok=True)

        return {
            "city": city,
            "image_url": image_url,
            "prediction": result["prediction"],
            "confidence": result["confidence_label"],
            "candidates": [
                {"mosaic_id": c["invader_id"], "score": c["score"]}
                for c in result["top_k"][:top_k]
            ],
        }

# ── Upload local artifacts to Volume ─────────────────────────────────────────

@app.local_entrypoint()
def upload_data():
    """Upload model weights, FAISS indexes, and detector to the Modal Volume."""
    if not LOCAL_MODEL_DIR.exists():
        raise SystemExit(f"Model not found at {LOCAL_MODEL_DIR} — download from Drive first")
    if not LOCAL_INDEX_DIR.exists():
        raise SystemExit(f"Indexes not found at {LOCAL_INDEX_DIR} — run `siit build-index` first")

    with vol.batch_upload(force=True) as upload:
        print(f"Uploading model weights from {LOCAL_MODEL_DIR}...")
        upload.put_directory(str(LOCAL_MODEL_DIR), "/model")

        print(f"Uploading indexes from {LOCAL_INDEX_DIR}...")
        upload.put_directory(str(LOCAL_INDEX_DIR), "/indexes")

        if LOCAL_DETECTOR_PATH.exists():
            print("Uploading detector weights...")
            upload.put_file(str(LOCAL_DETECTOR_PATH), "/detector/mosaic_detector_v3.pt")

        manifest = build_deployment_manifest()
        ensure_parent(LOCAL_MANIFEST_PATH)
        LOCAL_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Uploading deployment manifest from {LOCAL_MANIFEST_PATH}...")
        upload.put_file(str(LOCAL_MANIFEST_PATH), "/metadata/deployment-manifest.json")

    publish_manifest_to_image_wall(manifest)

    print("Upload complete.")


def publish_manifest_to_image_wall(manifest: dict) -> None:
    if not IMAGE_WALL_META_SECRET:
        print("Skipping image-wall model-meta publish (IMAGE_WALL_META_SECRET not set).")
        return

    payload = json.dumps({"manifest": manifest}).encode("utf-8")
    req = urlrequest.Request(
        IMAGE_WALL_META_ENDPOINT,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Model-Meta-Secret": IMAGE_WALL_META_SECRET,
            "User-Agent": "si-image-trainer/1.0 (+https://si-image-wall.pages.dev)",
        },
    )

    try:
        with urlrequest.urlopen(req, timeout=20) as response:
            status = response.getcode()
            if status < 200 or status >= 300:
                raise RuntimeError(f"image-wall model-meta publish failed with status {status}")
        verify_published_manifest(manifest)
        print(f"Published and verified deployment manifest on image-wall: {IMAGE_WALL_META_ENDPOINT}")
    except urlerror.HTTPError as exc:
        raise RuntimeError(f"image-wall model-meta publish failed: HTTP {exc.code}") from exc
    except urlerror.URLError as exc:
        raise RuntimeError(f"image-wall model-meta publish failed: {exc}") from exc


def verify_published_manifest(expected_manifest: dict) -> None:
    verify_url = IMAGE_WALL_META_ENDPOINT
    req = urlrequest.Request(
        verify_url,
        method="GET",
        headers={
            "User-Agent": "si-image-trainer/1.0 (+https://si-image-wall.pages.dev)",
        },
    )

    with urlrequest.urlopen(req, timeout=20) as response:
        status = response.getcode()
        if status < 200 or status >= 300:
            raise RuntimeError(f"image-wall model-meta verify failed with status {status}")
        body = response.read().decode("utf-8")

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("image-wall model-meta verify failed: invalid JSON response") from exc

    stored_manifest = payload.get("manifest")
    if not isinstance(stored_manifest, dict):
        raise RuntimeError("image-wall model-meta verify failed: missing manifest in response")

    # Verify a minimal set of lineage keys round-tripped correctly.
    keys_to_check = [
        "generated_at_utc",
        "model_label",
        "model_sha256",
        "index_sha256",
        "trainer_repo_commit",
        "reference_repo_commit",
    ]
    mismatches = [
        key for key in keys_to_check
        if stored_manifest.get(key) != expected_manifest.get(key)
    ]
    if mismatches:
        raise RuntimeError(
            "image-wall model-meta verify failed: mismatch for keys "
            + ", ".join(mismatches)
        )


def build_deployment_manifest() -> dict:
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_label": LOCAL_MODEL_DIR.name,
        "model_dir": str(LOCAL_MODEL_DIR),
        "index_dir": str(LOCAL_INDEX_DIR),
        "model_sha256": hash_directory(LOCAL_MODEL_DIR),
        "index_sha256": hash_directory(LOCAL_INDEX_DIR),
        "detector_sha256": hash_file(LOCAL_DETECTOR_PATH) if LOCAL_DETECTOR_PATH.exists() else None,
        "trainer_repo_commit": git_commit(Path.cwd()),
        "reference_repo_commit": git_commit(Path("/Users/jakob/Projects/si-reference-library")),
    }

    local_training_state = LOCAL_MODEL_DIR / "training_state.json"
    if local_training_state.exists():
        manifest["training_state"] = json.loads(local_training_state.read_text(encoding="utf-8"))

    return manifest


def git_commit(repo_path: Path) -> Optional[str]:
    if not (repo_path / ".git").exists():
        return None

    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def hash_directory(root: Path) -> str:
    if not root.exists():
        return "missing"

    digest = hashlib.sha256()
    for file_path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = file_path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hash_file(file_path).encode("utf-8"))
        digest.update(b"\0")

    return digest.hexdigest()


def hash_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

# ── Local test entrypoint ─────────────────────────────────────────────────────

@app.local_entrypoint()
def test_predict(city: str = "PA", image_url: str = ""):
    if not image_url:
        raise SystemExit("Provide --image-url")
    svc = InferenceService()
    result = svc.predict.remote(city, image_url)
    import json
    print(json.dumps(result, indent=2))
