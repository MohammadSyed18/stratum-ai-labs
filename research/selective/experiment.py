"""Leakage-aware split, model fitting, calibration, and untouched test reporting."""

from .data import Preprocessor, stratified_split
from .model import LogisticRegression
from .calibration import TemperatureCalibrator
from .metrics import evaluate, choose_threshold, risk_coverage


def run_experiment(dataset, *, seed=19, maximum_error=0.15):
    train, validation, test = stratified_split(dataset.labels, seed)
    rows = lambda indices: [dataset.rows[index] for index in indices]
    labels = lambda indices: [dataset.labels[index] for index in indices]
    preprocessor = Preprocessor().fit(rows(train))
    training = preprocessor.transform(rows(train))
    val = preprocessor.transform(rows(validation))
    heldout = preprocessor.transform(rows(test))
    model = LogisticRegression().fit(training, labels(train))
    calibrator = TemperatureCalibrator().fit(model.logits(val), labels(validation))
    validation_probs = calibrator.transform(model.logits(val))
    selection = choose_threshold(
        labels(validation), validation_probs, maximum_error=maximum_error
    )
    raw = model.predict_proba(heldout)
    calibrated = calibrator.transform(model.logits(heldout))
    stability = []
    for row in heldout:
        original = model.predict_proba([row])[0]
        largest = 0
        for index in range(len(row)):
            shifted = row[:]
            shifted[index] += 0.01
            largest = max(largest, abs(model.predict_proba([shifted])[0] - original))
        stability.append(largest)
    return {
        "seed": seed,
        "features": list(dataset.features),
        "split_rows": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "split_indices": {
            "train": list(train),
            "validation": list(validation),
            "test": list(test),
        },
        "preprocessing": preprocessor.parameters(),
        "weights": model.weights,
        "bias": model.bias,
        "training_trace": model.losses,
        "temperature": calibrator.temperature,
        "threshold_selection": selection,
        "test": {
            "raw": evaluate(labels(test), raw),
            "calibrated": evaluate(labels(test), calibrated),
            "selective": evaluate(labels(test), calibrated, selection["threshold"]),
            "risk_coverage": risk_coverage(labels(test), calibrated),
            "mean_probability_change_for_0_01_standardized_perturbation": sum(stability)
            / len(stability),
        },
        "interpretation": "Local benchmark evaluation. Threshold feasibility is measured on validation data and is not a guarantee of test or deployment error. No original study result is implied.",
    }
