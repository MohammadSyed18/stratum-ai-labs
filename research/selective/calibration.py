"""Temperature selection on validation logits only."""

from math import isfinite
from .model import sigmoid, log_loss


class TemperatureCalibrator:
    def fit(self, logits, labels, candidates=None):
        if (
            not logits
            or len(logits) != len(labels)
            or any(not isfinite(value) for value in logits)
        ):
            raise ValueError("validation logits and labels required")
        candidates = (
            list(candidates)
            if candidates is not None
            else [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8]
        )
        if not candidates or any(
            not isfinite(value) or value <= 0 for value in candidates
        ):
            raise ValueError("positive temperatures required")
        scored = [
            (
                log_loss(labels, [sigmoid(value / temperature) for value in logits]),
                temperature,
            )
            for temperature in candidates
        ]
        self.validation_loss, self.temperature = min(
            scored, key=lambda item: (item[0], abs(item[1] - 1))
        )
        return self

    def transform(self, logits):
        if not hasattr(self, "temperature"):
            raise ValueError("calibrator must be fit first")
        if any(not isfinite(value) for value in logits):
            raise ValueError("nonfinite logit")
        return [sigmoid(value / self.temperature) for value in logits]
