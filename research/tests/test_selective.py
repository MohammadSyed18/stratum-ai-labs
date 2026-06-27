import unittest
from research.selective.data import Dataset, Preprocessor, stratified_split
from research.selective.model import LogisticRegression
from research.selective.calibration import TemperatureCalibrator
from research.selective.metrics import evaluate
from research.selective.experiment import run_experiment


class SelectiveTests(unittest.TestCase):
    def test_training_statistics_do_not_use_test_rows(self):
        processor = Preprocessor().fit([[1, None], [3, 2], [5, 4]])
        before = processor.parameters()
        transformed = processor.transform([[10000, None]])
        self.assertEqual(processor.parameters(), before)
        self.assertEqual(before["medians"], [3, 3])
        self.assertGreater(transformed[0][0], 100)

    def test_optimizer_and_calibrator_are_inspectable(self):
        model = LogisticRegression(epochs=200).fit([[-2], [-1], [1], [2]], [0, 0, 1, 1])
        self.assertLess(model.losses[-1]["log_loss"], model.losses[0]["log_loss"])
        calibrated = TemperatureCalibrator().fit([-10, 10, -10, 10], [0, 1, 1, 0])
        self.assertGreater(calibrated.temperature, 1)
        self.assertEqual(evaluate([0, 1], [0.5, 0.5], 0.9)["answered_error_rate"], None)

    def test_splits_are_disjoint_and_full_experiment_reproduces(self):
        labels = (0,) * 15 + (1,) * 15
        train, validation, test = stratified_split(labels)
        self.assertEqual(len(set(train) | set(validation) | set(test)), 30)
        self.assertFalse(set(train) & set(validation))
        self.assertFalse(set(test) & set(validation))
        rows = tuple(
            (
                float(index % 15) / 10 + 2 * label,
                None if index % 4 == 0 else float(label),
            )
            for index, label in enumerate(labels)
        )
        dataset = Dataset(("a", "b"), rows, labels)
        first = run_experiment(dataset)
        self.assertEqual(first, run_experiment(dataset))
        self.assertEqual(first["split_rows"]["test"], 6)
        self.assertIn("risk_coverage", first["test"])
