# Research overview

Stratum AI Labs brings student researchers together to develop questions, compare methods, and communicate results clearly.

## Research areas

### Clinical prediction and uncertainty

Research on probability calibration, confidence-based abstention, and the stability of explanations under small input changes. Evaluation should distinguish prediction quality from coverage: how often a model answers and how often it defers.

### AI in education

Literature-based work examining personalized tutoring, productivity, educational expression, equity, privacy, academic integrity, misinformation, and social interaction.

### Reasoning and changing evidence

ARISE explores claims, supporting sources, confidence, dependency relationships, and revisions when evidence changes. The public [ARISE experiment](https://github.com/MohammadSyed18/ARISE-experiment) currently includes a deterministic adapter, example scenarios, and evaluation utilities.

## How this repository supports research

The website presents the lab and its project portfolio. The templates in this repository support project scoping, reproducibility, meeting discussion, and evaluation. Experiment implementations, datasets, and publications should be linked from the relevant project when available.

- [Experiment plan](../templates/experiment-plan.md)
- [Experiment log](../templates/experiment-log.md)

## Weekly research discussion

1. State the question and the current evidence.
2. Review what changed since the last meeting.
3. Inspect results and failure cases together.
4. Identify the next experiment and who owns it.
5. Record decisions, limitations, and links to artifacts.

## Experiment code

The [selective prediction package](../research/README.md) contains CSV preprocessing, logistic regression, validation-only temperature calibration, confidence-based abstention, risk/coverage evaluation, and feature-perturbation summaries. It includes a synthetic runnable fixture and tests for split isolation, training statistics, calibration, and reproducibility.
