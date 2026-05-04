from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from si_image_trainer.indexing.aggregate import aggregate_hits
from si_image_trainer.indexing.search import top_k_search
from si_image_trainer.models.calibrator import label_confidence
from si_image_trainer.models.detector import MosaicDetector
from si_image_trainer.models.embedder import BaselineEmbedder


class RetrievalPipeline:
    def __init__(self, index_dir: str | Path, embedding_config: dict[str, int], retrieval_config: dict[str, Any], confidence_config: dict[str, float], detector_config: dict[str, Any] | None = None) -> None:
        self.index_dir = Path(index_dir)
        self.embedder = BaselineEmbedder(**embedding_config)
        self.retrieval_config = retrieval_config
        self.confidence_config = confidence_config
        self.detector = MosaicDetector(**detector_config) if detector_config else None
        self._city_cache: dict[str, tuple[np.ndarray, list[dict[str, Any]]]] = {}

    def _load_city_index(self, city_code: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
        cached = self._city_cache.get(city_code)
        if cached is not None:
            return cached

        city_dir = self.index_dir / city_code
        embeddings_path = city_dir / "embeddings.npy"
        legacy_path = city_dir / "index.npz"
        metadata_path = city_dir / "metadata.json"

        if not embeddings_path.exists() and not legacy_path.exists():
            raise FileNotFoundError(f"No index available for city_code={city_code}")
        if embeddings_path.exists():
            matrix = np.load(embeddings_path, mmap_mode="r")
        else:
            payload = np.load(legacy_path)
            matrix = payload["embeddings"]

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        loaded = (matrix, metadata)
        self._city_cache[city_code] = loaded
        return loaded

    def predict(self, image_path: str, city_code: str) -> dict[str, Any]:
        try:
            matrix, metadata = self._load_city_index(city_code)
        except FileNotFoundError as exc:
            return {
                "query_image": image_path,
                "city_code": city_code,
                "prediction": None,
                "top_k": [],
                "confidence_label": "unknown",
                "diagnostics": {"reason": str(exc), "top_score": 0.0, "margin_to_second": 0.0, "raw_hits": []},
            }
        crop = self.detector.crop(image_path) if self.detector else None
        if crop is not None:
            query = self.embedder.embed_image(crop)
        else:
            query = self.embedder.embed_path(image_path)
        indices, scores = top_k_search(query, matrix, int(self.retrieval_config["top_n"]))
        hits: list[dict[str, Any]] = []
        for index, score in zip(indices.tolist(), scores.tolist()):
            hit = dict(metadata[index])
            hit["score"] = round(float(score), 6)
            hits.append(hit)

        candidates = aggregate_hits(hits, self.retrieval_config.get("role_weights"))[: int(self.retrieval_config["rerank_m"])]
        top_candidate = candidates[0] if candidates else None
        next_score = float(candidates[1]["aggregate_score"]) if len(candidates) > 1 else 0.0
        top_score = float(top_candidate["aggregate_score"]) if top_candidate else 0.0
        margin = top_score - next_score
        confidence = label_confidence(top_score, margin, self.confidence_config)

        return {
            "query_image": image_path,
            "city_code": city_code,
            "prediction": top_candidate["invader_id"] if top_candidate else None,
            "top_k": [
                {
                    "invader_id": row["invader_id"],
                    "score": round(float(row["aggregate_score"]), 6),
                    "support": row["support"],
                }
                for row in candidates
            ],
            "confidence_label": confidence,
            "diagnostics": {
                "top_score": round(top_score, 6),
                "margin_to_second": round(margin, 6),
                "used_crop": crop is not None,
                "raw_hits": hits[:5],
            },
        }
