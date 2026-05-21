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
from pathlib import Path

import modal

# ── Volume & image ────────────────────────────────────────────────────────────

VOLUME_NAME = "si-inference-data"
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

DATA_DIR   = Path("/data")
MODEL_DIR  = DATA_DIR / "model"
INDEX_DIR  = DATA_DIR / "indexes"
DETECTOR_PATH = DATA_DIR / "detector" / "mosaic_detector_v3.pt"

LOCAL_MODEL_DIR    = Path("outputs/models/exp17")
LOCAL_INDEX_DIR    = Path("outputs/indexes")
LOCAL_DETECTOR_PATH = Path("outputs/models/mosaic_detector_v3.pt")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.0",
        "torchvision==0.18.0",
        "transformers>=4.40",
        "numpy>=1.26",
        "Pillow>=10.0",
        "ultralytics>=8.4",
        "requests",
    )
    .add_local_dir("src/", "/app/src")
    .env({"PYTHONPATH": "/app/src"})
)

app = modal.App("si-inference", image=image)

# ── Inference service ─────────────────────────────────────────────────────────

@app.cls(
    gpu="T4",
    volumes={str(DATA_DIR): vol},
    keep_warm=1,
    timeout=60,
)
class InferenceService:
    @modal.enter()
    def load(self):
        from si_image_trainer.inference.pipeline import RetrievalPipeline
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

    @modal.web_endpoint(method="POST", label="si-predict")
    def predict_endpoint(self, body: dict) -> dict:
        return self._predict(body["city"], body["image_url"], int(body.get("top_k", 5)))

    @modal.method()
    def predict(self, city: str, image_url: str, top_k: int = 5) -> dict:
        return self._predict(city, image_url, top_k)

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
        upload.put_directory(str(LOCAL_MODEL_DIR), "/data/model")

        print(f"Uploading indexes from {LOCAL_INDEX_DIR}...")
        upload.put_directory(str(LOCAL_INDEX_DIR), "/data/indexes")

        if LOCAL_DETECTOR_PATH.exists():
            print("Uploading detector weights...")
            upload.put_file(str(LOCAL_DETECTOR_PATH), "/data/detector/mosaic_detector_v3.pt")

    print("Upload complete.")

# ── Local test entrypoint ─────────────────────────────────────────────────────

@app.local_entrypoint()
def test_predict(city: str = "PA", image_url: str = ""):
    if not image_url:
        raise SystemExit("Provide --image-url")
    svc = InferenceService()
    result = svc.predict.remote(city, image_url)
    import json
    print(json.dumps(result, indent=2))
