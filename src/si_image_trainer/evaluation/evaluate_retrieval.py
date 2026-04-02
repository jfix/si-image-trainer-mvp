from __future__ import annotations

from pathlib import Path
from typing import Any

from si_image_trainer.evaluation.metrics import summarize_predictions
from si_image_trainer.inference.pipeline import RetrievalPipeline
from si_image_trainer.utils.io import read_jsonl, write_json


def evaluate(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(config["paths"]["eval_manifest"])
    queries = read_jsonl(manifest_path)
    pipeline = RetrievalPipeline(
        index_dir=config["paths"]["index_dir"],
        embedding_config=config["embedding"],
        retrieval_config=config["retrieval"],
        confidence_config=config["confidence"],
    )

    records: list[dict[str, Any]] = []
    for query in queries:
        prediction = pipeline.predict(query["image_path"], query["city_code"])
        prediction["query_id"] = query["query_id"]
        prediction["label_invader_id"] = query["label_invader_id"]
        records.append(prediction)

    summary = summarize_predictions(records, top_k=min(5, int(config["retrieval"]["top_n"])))
    report = {"summary": summary, "records": records}
    write_json(Path(config["paths"]["report_dir"]) / "evaluation.json", report)
    return report
