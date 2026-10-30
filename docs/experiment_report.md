# Project report template

## Data and protocol
- Permitted data location and CSV schema:
- Training/validation counts and split fingerprint:
- Training seed(s), hardware, package versions:

## Architecture
- Stage widths, block counts, kernel size, skip projections:
- Trainable parameter count (maximum 5,000,000):
- Motivation and changes from baseline:

## Training
- Optimizer, learning-rate schedule, epochs, batch size:
- Augmentation and regularization:
- Reproduction command and resolved config path:

## Experiments
Copy measured results from README and include learning curves from metrics.csv. Separate architectural changes from training-recipe changes; compare using the same validation split. Do not report synthetic smoke-test metrics as competition results.

## Selected model
- Highest validation accuracy, training/validation loss, selected epoch:
- Checkpoint path and selection rationale:
- Final full-data retraining epoch budget, if used:
- Prediction command and submission row count:

## Limitations
Discuss seed variability and validation overfitting. Hidden evaluation loss cannot be measured without labels. Private leaderboard accuracy cannot be guaranteed from validation results.
