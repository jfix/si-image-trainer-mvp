from __future__ import annotations

from typing import Any

from si_image_trainer.inference.pipeline import RetrievalPipeline


def predict_one(config: dict[str, Any], image_path: str, city_code: str) -> dict[str, Any]:
    pipeline = RetrievalPipeline(
        index_dir=config["paths"]["index_dir"],
        embedding_config=config["embedding"],
        retrieval_config=config["retrieval"],
        confidence_config=config["confidence"],
    )
    return pipeline.predict(image_path=image_path, city_code=city_code)
