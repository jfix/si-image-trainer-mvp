from __future__ import annotations

import argparse
import json

from si_image_trainer.data.prepare_queries import build_query_manifest
from si_image_trainer.data.prepare_references import build_reference_manifest
from si_image_trainer.data.split_dataset import build_eval_manifest
from si_image_trainer.evaluation.error_analysis import build_error_analysis
from si_image_trainer.evaluation.evaluate_calibration import evaluate_calibration
from si_image_trainer.evaluation.evaluate_retrieval import evaluate
from si_image_trainer.indexing.build_index import build_indexes
from si_image_trainer.inference.predict import predict_one
from si_image_trainer.training.train_metric import train_metric
from si_image_trainer.utils.io import load_yaml
from si_image_trainer.utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="siit")
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ["prepare-references", "prepare-queries", "split-dataset", "build-index", "evaluate", "evaluate-calibration", "error-analysis", "train-metric"]:
        sub = subparsers.add_parser(command)
        sub.add_argument("--config", required=True)

    predict = subparsers.add_parser("predict")
    predict.add_argument("--config", required=True)
    predict.add_argument("--city", required=True)
    predict.add_argument("--image", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    config = load_yaml(args.config)

    if args.command == "prepare-references":
        rows = build_reference_manifest(config["paths"]["reference_root"], config["paths"]["reference_manifest"])
        print(json.dumps({"written": len(rows), "path": config["paths"]["reference_manifest"]}, indent=2))
        return
    if args.command == "prepare-queries":
        rows = build_query_manifest(
            config["paths"]["live_root"],
            config["paths"]["query_manifest"],
            data_root=config["paths"].get("live_data_root"),
            place_mapping_path=config["paths"].get("place_mappings"),
        )
        print(json.dumps({"written": len(rows), "path": config["paths"]["query_manifest"]}, indent=2))
        return
    if args.command == "split-dataset":
        rows = build_eval_manifest(config["paths"]["reference_manifest"], config["paths"]["eval_manifest"])
        print(json.dumps({"written": len(rows), "path": config["paths"]["eval_manifest"]}, indent=2))
        return
    if args.command == "build-index":
        rows = build_indexes(
            config["paths"]["reference_manifest"],
            config["paths"]["index_dir"],
            config["embedding"],
            config.get("retrieval"),
            exclude_manifest_path=config["paths"].get("eval_manifest"),
            detector_config=config.get("detector"),
        )
        print(json.dumps(rows, indent=2))
        return
    if args.command == "predict":
        print(json.dumps(predict_one(config, args.image, args.city), indent=2))
        return
    if args.command == "evaluate":
        print(json.dumps(evaluate(config)["summary"], indent=2))
        return
    if args.command == "evaluate-calibration":
        report = evaluate(config)
        print(json.dumps(evaluate_calibration(report, config["paths"]["report_dir"]), indent=2))
        return
    if args.command == "error-analysis":
        report = evaluate(config)
        print(json.dumps(build_error_analysis(report, config["paths"]["report_dir"]), indent=2))
        return
    if args.command == "train-metric":
        print(json.dumps(train_metric(config), indent=2))
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
