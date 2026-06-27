"""Inspectable full-batch logistic regression without third-party dependencies."""

from math import exp, isfinite, log


def sigmoid(value):
    if value >= 0:
        return 1 / (1 + exp(-value))
    exponential = exp(value)
    return exponential / (1 + exponential)


def log_loss(labels, probabilities):
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("aligned nonempty predictions required")
    total = 0
    for label, probability in zip(labels, probabilities):
        if (
            type(label) is not int
            or label not in (0, 1)
            or not isfinite(probability)
            or not 0 <= probability <= 1
        ):
            raise ValueError("invalid prediction")
        clipped = max(1e-12, min(1 - 1e-12, probability))
        total -= label * log(clipped) + (1 - label) * log(1 - clipped)
    return total / len(labels)


class LogisticRegression:
    def __init__(self, *, learning_rate=0.1, epochs=600, regularization=0.01):
        if (
            not isfinite(learning_rate)
            or learning_rate <= 0
            or type(epochs) is not int
            or not 1 <= epochs <= 100000
            or not isfinite(regularization)
            or regularization < 0
        ):
            raise ValueError("invalid optimizer settings")
        self.learning_rate, self.epochs, self.regularization = (
            learning_rate,
            epochs,
            regularization,
        )

    def fit(self, rows, labels):
        if not rows or len(rows) != len(labels) or not rows[0]:
            raise ValueError("aligned training rows required")
        width = len(rows[0])
        if any(
            len(row) != width or any(not isfinite(value) for value in row)
            for row in rows
        ):
            raise ValueError("invalid training rows")
        if any(type(label) is not int or label not in (0, 1) for label in labels):
            raise ValueError("binary labels required")
        self.weights, self.bias, self.losses = [0.0] * width, 0.0, []
        for epoch in range(self.epochs):
            predictions = self.predict_proba(rows)
            errors = [
                probability - label for probability, label in zip(predictions, labels)
            ]
            for feature in range(width):
                gradient = sum(
                    error * row[feature] for row, error in zip(rows, errors)
                ) / len(rows)
                self.weights[feature] -= self.learning_rate * (
                    gradient + self.regularization * self.weights[feature]
                )
            self.bias -= self.learning_rate * sum(errors) / len(rows)
            if epoch % 25 == 0 or epoch == self.epochs - 1:
                self.losses.append(
                    {
                        "epoch": epoch + 1,
                        "log_loss": log_loss(labels, self.predict_proba(rows)),
                    }
                )
        return self

    def logits(self, rows):
        if not hasattr(self, "weights"):
            raise ValueError("model must be fit first")
        result = []
        for row in rows:
            if len(row) != len(self.weights) or any(
                not isfinite(value) for value in row
            ):
                raise ValueError("invalid prediction row")
            result.append(
                self.bias
                + sum(weight * value for weight, value in zip(self.weights, row))
            )
        return result

    def predict_proba(self, rows):
        return [sigmoid(value) for value in self.logits(rows)]

    def explain(self, row):
        if len(row) != len(self.weights):
            raise ValueError("invalid explanation row")
        return {
            "bias": self.bias,
            "logit_contributions": [
                weight * value for weight, value in zip(self.weights, row)
            ],
        }
