# Stratum AI Labs

Website and research-planning resources for Stratum AI Labs, a student-led AI research group.

## Website

- `index.html` — lab homepage
- `fellows.html` — fellows program and application section
- `projects.html` — project portfolio
- `styles.css` — shared layout and styling

Open `index.html` in a browser to preview the static site. The HTML pages share the same stylesheet and navigation.

## Research resources

- [Research overview](docs/research-overview.md): uncertainty in prediction, AI in education, and reasoning under changing evidence.
- [Experiment plan](templates/experiment-plan.md): define a question, baseline, data, metrics, and reproduction steps.
- [Experiment log](templates/experiment-log.md): record runs, results, limitations, and research decisions.
- [ARISE experiment](https://github.com/MohammadSyed18/ARISE-experiment): a separate repository for claims, sources, confidence, and evidence revision.

The project portfolio records project status. Research directions and planning templates are separate from implemented experiments and published results.

## Implementation details

The project portfolio includes accessible text search, category filters, deterministic ranking, and facet counts. Run `npm test` to check the search module.

## Research implementation

[research/](research/README.md) contains a runnable selective-prediction experiment in Python: train-only preprocessing, regularized logistic regression, validation-only probability calibration, abstention threshold selection, and test-set metrics. The website's project-search module and its JavaScript tests remain independent of this experiment package.

```sh
npm test
python -m unittest discover -s research/tests -v
python -m research.selective.cli research/examples/synthetic.csv --out research/runs/example.json
```

The example dataset is synthetic and demonstrates the code path. It does not establish or reproduce a paper's reported benchmark values.
