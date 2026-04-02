from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from si_image_trainer.indexing.aggregate import aggregate_hits
from si_image_trainer.indexing.search import top_k_search
from si_image_trainer.models.calibrator import label_confidence
from si_image_trainer.models.embedder import BaselineEmbedder


class RetrievalPipeline:
    def __init__(self, index_dir: str | Path, embedding_config: dict[str, int], retrieval_config: dict[str, Any], confidence_config: dict[str, float]) -> None:
        self.index_dir = Path(index_dir)
        self.embedder = BaselineEmbedder(**embedding_config)
        self.retrieval_config = retrieval_config
        self.confidence_config = confidence_config

    def _load_city_index(self, city_code: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
        city_dir = self.index_dir / city_code
        if not (city_dir / "index.npz").exists():
            raise FileNotFoundError(f"No index available for city_code={city_code}")
        payload = np.load(city_dir / "index.npz")
        metadata = json.loads((city_dir / "metadata.json").read_text(encoding="utf-8"))
        return payload["embeddings"], metadata

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
        query = self.embedder.embed_path(image_path)
        indices, scores = top_k_search(query, matrix, int(self.retrieval_config["top_n"]))
        hits: list[dict[str, Any]] = []
        for index, score in zip(indices.tolist(), scores.tolist()):
            hit = dict(metadata[index])
            hit["score"] = round(float(score), 6)
            hits.append(hit)

        candidates = aggregate_hits(hits)[: int(self.retrieval_config["rerank_m"])]
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
                "raw_hits": hits[:5],
            },
        }
