"""CSV import, stratified splitting, and train-only imputation/standardization."""

import csv
from dataclasses import dataclass
from math import isfinite, sqrt
from pathlib import Path
import random
from statistics import mean, median


@dataclass(frozen=True)
class Dataset:
    features: tuple[str, ...]
    rows: tuple[tuple[float | None, ...], ...]
    labels: tuple[int, ...]


def read_csv(path, label="label"):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        if not columns or label not in columns or len(columns) != len(set(columns)):
            raise ValueError("unique column names and a label column required")
        features = tuple(column for column in columns if column != label)
        if not features:
            raise ValueError("feature columns required")
        rows, labels = [], []
        for number, raw in enumerate(reader, 2):
            if None in raw or any(value is None for value in raw.values()):
                raise ValueError(f"line {number}: malformed row")
            if raw[label] not in ("0", "1"):
                raise ValueError(f"line {number}: label must be 0 or 1")
            row = []
            for feature in features:
                value = raw[feature].strip()
                parsed = None if value in ("", "?", "NA") else float(value)
                if parsed is not None and (not isfinite(parsed) or abs(parsed) > 1e100):
                    raise ValueError(f"line {number}: invalid feature")
                row.append(parsed)
            rows.append(tuple(row))
            labels.append(int(raw[label]))
    if not rows:
        raise ValueError("dataset is empty")
    return Dataset(features, tuple(rows), tuple(labels))


def stratified_split(labels, seed=19):
    groups = {0: [], 1: []}
    for index, label in enumerate(labels):
        if type(label) is not int or label not in groups:
            raise ValueError("binary integer labels required")
        groups[label].append(index)
    rng = random.Random(seed)
    splits = [[], [], []]
    for label in (0, 1):
        indices = groups[label][:]
        if len(indices) < 5:
            raise ValueError(
                "each class needs at least five rows for train/validation/test splitting"
            )
        rng.shuffle(indices)
        train = max(1, int(len(indices) * 0.6))
        validation = max(1, int(len(indices) * 0.2))
        splits[0].extend(indices[:train])
        splits[1].extend(indices[train : train + validation])
        splits[2].extend(indices[train + validation :])
    return tuple(tuple(sorted(indices)) for indices in splits)


class Preprocessor:
    def fit(self, rows):
        if not rows or not rows[0]:
            raise ValueError("training rows required")
        width = len(rows[0])
        if any(len(row) != width for row in rows):
            raise ValueError("inconsistent feature width")
        self.medians, self.means, self.scales = [], [], []
        for index in range(width):
            observed = [row[index] for row in rows if row[index] is not None]
            if not observed:
                raise ValueError(f"training feature {index} is entirely missing")
            if any(not isfinite(value) for value in observed):
                raise ValueError("nonfinite feature")
            center = median(observed)
            values = [center if row[index] is None else row[index] for row in rows]
            average = mean(values)
            scale = sqrt(mean((value - average) ** 2 for value in values)) or 1
            self.medians.append(center)
            self.means.append(average)
            self.scales.append(scale)
        return self

    def transform(self, rows):
        if not hasattr(self, "medians"):
            raise ValueError("preprocessor must be fit first")
        result = []
        for row in rows:
            if len(row) != len(self.medians):
                raise ValueError("feature width changed")
            converted = [
                ((self.medians[index] if value is None else value) - self.means[index])
                / self.scales[index]
                for index, value in enumerate(row)
            ]
            if not all(isfinite(value) for value in converted):
                raise ValueError("nonfinite transformed feature")
            result.append(converted)
        return result

    def parameters(self):
        return {
            "medians": self.medians[:],
            "means": self.means[:],
            "scales": self.scales[:],
        }
