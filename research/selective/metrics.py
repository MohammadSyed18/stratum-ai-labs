"""Probability quality, confidence abstention, and risk/coverage summaries."""

from math import isfinite
from .model import log_loss


def evaluate(labels, probabilities, threshold=0.5):
    if not isfinite(threshold) or not 0.5 <= threshold <= 1:
        raise ValueError("confidence threshold must be .5..1")
    logloss = log_loss(labels, probabilities)
    answered = [
        index
        for index, probability in enumerate(probabilities)
        if max(probability, 1 - probability) >= threshold
    ]
    errors = sum(
        (probabilities[index] >= 0.5) != bool(labels[index]) for index in answered
    )
    return {
        "rows": len(labels),
        "brier_score": sum(
            (probability - label) ** 2
            for label, probability in zip(labels, probabilities)
        )
        / len(labels),
        "log_loss": logloss,
        "coverage": len(answered) / len(labels),
        "answered": len(answered),
        "deferred": len(labels) - len(answered),
        "answered_error_rate": errors / len(answered) if answered else None,
    }


def choose_threshold(
    labels, probabilities, *, maximum_error=0.15, minimum_coverage=0.2
):
    if not 0 <= maximum_error <= 1 or not 0 < minimum_coverage <= 1:
        raise ValueError("invalid selection constraints")
    candidates = sorted({0.5, 1.0, *[max(value, 1 - value) for value in probabilities]})
    eligible = []
    for threshold in candidates:
        result = evaluate(labels, probabilities, threshold)
        if (
            result["coverage"] >= minimum_coverage
            and result["answered_error_rate"] <= maximum_error
        ):
            eligible.append((threshold, result))
    if not eligible:
        return {
            "threshold": 1.0,
            "feasible": False,
            "validation": evaluate(labels, probabilities, 1),
        }
    threshold, result = min(eligible, key=lambda item: (-item[1]["coverage"], item[0]))
    return {"threshold": threshold, "feasible": True, "validation": result}


def risk_coverage(labels, probabilities):
    return [
        {"threshold": threshold, **evaluate(labels, probabilities, threshold)}
        for threshold in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
    ]
