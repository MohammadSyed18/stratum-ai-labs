# Selective prediction experiments

A standard-library Python pipeline for binary classification, probability calibration, and confidence-based abstention. The implementation is inspectable and runs without downloading data or models.

```sh
python -m unittest discover -s research/tests -v
python -m research.selective.cli research/examples/synthetic.csv --seed 19 --out research/runs/example.json
```

Run these commands from the repository root. The included CSV is a small synthetic fixture for checking the pipeline; it contains no patient records and does not reproduce a published study result.

## Modules

| Module | Purpose |
| --- | --- |
| [data.py](selective/data.py) | Numeric CSV parsing, stratified splits, train-only median imputation and standardization |
| [model.py](selective/model.py) | Full-batch regularized logistic regression, stable sigmoid, learning trace, logit contributions |
| [calibration.py](selective/calibration.py) | Temperature selection using validation labels and logits |
| [metrics.py](selective/metrics.py) | Brier score, log loss, abstention, answered-case error, risk/coverage curves |
| [experiment.py](selective/experiment.py) | Fit/calibrate/select/evaluate orchestration and small feature-perturbation checks |
| [cli.py](selective/cli.py) | Reproducible CSV experiment command |

## Data and leakage boundaries

Input requires a binary `label` column and numeric feature columns. Empty fields, `?`, and `NA` represent missing features. Each class needs at least five rows for the three-way split. Duplicate CSV column names, malformed rows, nonfinite features, and unsupported labels are rejected.

The seeded split assigns 60% of each class to training, 20% to validation, and the remainder to testing. Imputation and scale statistics come only from training rows. The model trains on that transformed training set. Temperature and confidence thresholds are selected only on validation data. Test rows are then evaluated once with those settings.

Validation feasibility is not a guarantee of future error. If no candidate threshold meets the requested validation risk/coverage constraints, `feasible` is false. Output preserves that status rather than presenting the threshold as a successful guarantee.

The perturbation metric measures the largest probability change for a +0.01 shift in a standardized feature, averaged across test rows. It does not establish clinical robustness or causal feature importance. Logit contributions describe this linear model's score.

## Output

The JSON includes split indices, preprocessing parameters, weights, bias, training loss, temperature, threshold-selection status, raw/calibrated/selective test metrics, and a risk/coverage table. It contains no external benchmark claims. Use properly prepared, authorized data and evaluate its sampling assumptions separately before interpreting results.
