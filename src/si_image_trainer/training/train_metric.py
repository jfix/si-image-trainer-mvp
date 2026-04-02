from __future__ import annotations


def train_metric(config: dict[str, object]) -> dict[str, object]:
    return {
        "status": "not_implemented",
        "message": "Metric learning is deferred until labeled query data exists. The baseline retrieval pipeline is ready for measurement now.",
        "config": config.get("training", {}),
    }
