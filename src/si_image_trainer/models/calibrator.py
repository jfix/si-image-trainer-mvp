from __future__ import annotations


def label_confidence(score: float, margin: float, thresholds: dict[str, float]) -> str:
    if score >= thresholds["certainly_score"] and margin >= thresholds["margin_threshold"]:
        return "certainly"
    if score >= thresholds["probably_score"]:
        return "probably"
    if score >= thresholds["maybe_score"]:
        return "maybe"
    if score >= thresholds["unknown_score"] and margin >= thresholds["margin_threshold"] / 2:
        return "maybe"
    return "unknown"
