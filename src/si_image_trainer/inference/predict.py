from __future__ import annotations

import json
from typing import Any

from si_image_trainer.inference.pipeline import RetrievalPipeline

_PIPELINE_CACHE: dict[str, RetrievalPipeline] = {}


def predict_one(config: dict[str, Any], image_path: str, city_code: str) -> dict[str, Any]:
    cache_key = json.dumps(
        {
            "index_dir": config["paths"]["index_dir"],
            "embedding": config["embedding"],
            "retrieval": config["retrieval"],
            "confidence": config["confidence"],
        },
        sort_keys=True,
    )
    pipeline = _PIPELINE_CACHE.get(cache_key)
    if pipeline is None:
        pipeline = RetrievalPipeline(
            index_dir=config["paths"]["index_dir"],
            embedding_config=config["embedding"],
            retrieval_config=config["retrieval"],
            confidence_config=config["confidence"],
            detector_config=config.get("detector"),
        )
        _PIPELINE_CACHE[cache_key] = pipeline
    return pipeline.predict(image_path=image_path, city_code=city_code)
