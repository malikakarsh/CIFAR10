# CIFAR-10 ResNet Architecture Challenge

Configurable PyTorch experiments for CIFAR-10 classification under a **5-million-trainable-parameter limit**. The project supports residual CNN architecture comparisons, SGD and AdamW, resumable experiment plans, and staged training on Apple MPS or NVIDIA CUDA.

## Results

**Best completed validation accuracy: 95.82%. Best reported public Kaggle accuracy: 85.30%.**

<!-- RESULTS:START -->
| Architecture | Optimizer | Parameters | Epochs completed | Best epoch | Training loss | Validation loss | Validation accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline | SGD | 1,084,586 | 200/200 | 194 | 0.0032 | 0.1824 | 95.26% |
| Baseline | AdamW | 1,084,586 | 200/200 | 199 | 0.0005 | 0.4123 | 93.24% |
| Deeper | SGD | 1,860,522 | 200/200 | 199 | 0.0019 | 0.1826 | 95.38% |
| Deeper | AdamW | 1,860,522 | 200/200 | 191 | 0.0005 | 0.4550 | 93.56% |
| Wider | SGD | 2,436,346 | 200/200 | 200 | 0.0022 | 0.1574 | 95.64% |
| Wider | AdamW | 2,436,346 | 200/200 | 198 | 0.0002 | 0.3893 | 94.10% |
| Deeper + wider | SGD | 4,180,858 | 200/200 | 195 | 0.0016 | 0.1761 | **95.82%** |
| Deeper + wider | AdamW | 4,180,858 | 95/200 | 89 | 0.0182 | 0.3324 | 93.66% |

Losses and accuracy are reported at each run's highest-validation-accuracy epoch. Seven runs are complete; the final AdamW experiment is in progress, with 95/200 epochs saved in this snapshot. All runs use the same stratified 45,000/5,000 split. Training loss uses augmented images; validation loss uses unaugmented images.
<!-- RESULTS:END -->

| Public Kaggle submission | Accuracy |
|---|---:|
| Initial baseline submission during training | 79.50% |
| Completed baseline submission | 84.00% |
| Deeper + wider SGD submission | **85.30%** |

Public scores are reported from leaderboard screenshots. Hidden evaluation labels are unavailable; evaluation loss and private accuracy cannot be calculated. Details: [submission history](docs/leaderboard.md).

The completed SGD recipes outperform their matched AdamW recipes. Increasing SGD model capacity improves validation accuracy from 95.26% to 95.82%; the 2.44M wider model has the lowest loss at its selected checkpoint, **0.1574**, with **95.64%** accuracy. These single-seed results motivate testing augmentation and repeatability before another broad architecture sweep. See the [model comparison](docs/model_comparison.md).

## Setup

For **Python 3.12 on Apple Silicon**:

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
python scripts/check_env.py --require-mps
```

[requirements.txt](requirements.txt) pins the Mac environment. For NYU HPC, use a site-compatible Linux CUDA PyTorch environment and install the project with `python -m pip install -e .`; do not copy the Mac virtual environment. See the [HPC setup guide](docs/nyu_hpc.md). Optional dependencies are in `requirements-plot.txt` and `requirements-download.txt`.

## Dataset

Place the extracted competition files in `dataset/`, or override `data.root`:

```text
dataset/
├── train_labels.csv       # id,filename,label
├── test.csv               # id,filename
├── sample_submission.csv  # id,label
├── train_images/
└── test_images/
```

The default split uses **45,000 training and 5,000 validation images**, with split seed `42`. The evaluation set contains **2,000 unlabeled images**. Models train from scratch using only permitted training data.

```bash
python -m cifar10 check-data --config configs/base.yaml
```

## Next experiment plan

The next round starts with the **2.44M wider ResNet**, SGD with Nesterov momentum 0.9, batch size 128, initial LR 0.1, weight decay 0.0005, five warmup epochs, and cosine decay.

| Stage | Experiment | Epoch budget |
|---|---|---:|
| 1 | Same-hardware control, RandAugment, RandAugment + mixup | 200 per run |
| 2 | Selected recipe and control with a second training seed | 200 per run |
| 3 | Confirmed recipe with a longer cosine schedule | 300 |
| 4 | Selected recipe on the 4.18M model | Selected budget |
| 5 | Selected model trained on all 50,000 permitted images | Selected budget |

Later stages depend on validation results and are generated from the selected run's resolved config. The new control separates augmentation effects from the move to CUDA and mixed precision. Longer training and the larger architecture are retained only if their results justify the cost. The staged experiments are **prepared; no training from this new plan has been launched**.

From a **Burst Code Server terminal with an allocated GPU and prepared environment**:

```bash
python -m cifar10 plan --config configs/stages/01_augmentation.yaml --dry-run
python -m cifar10 plan --config configs/stages/01_augmentation.yaml \
  --readme runs_hpc/report.md
```

This executes the three first-stage trials sequentially with CUDA, writing checkpoints and metrics to `runs_hpc/`. Run Python directly inside the allocated Code Server session. Batch jobs are also supported through `scripts/hpc_train.sbatch`, using the account and partition assigned to your cluster.

For Mac training, override the device, worker count, and output directory:

```bash
python -m cifar10 plan --config configs/stages/01_augmentation.yaml \
  --readme docs/local_stage_report.md \
  --set training.device=mps data.num_workers=0 experiment.output_dir=runs_next
```

Generate a later stage after selecting a recipe:

```bash
python -m cifar10 prepare-stage \
  --source runs_hpc/SELECTED_STAGE1_RUN/config.yaml \
  --stage confirm --output-dir configs/generated/02_confirm
```

`prepare-stage` supports `confirm`, `budget`, `scale`, and `final`. It creates configurations without starting training. Generated configs may contain machine-specific paths and are ignored by Git. Full commands, decision criteria, and seed controls are in the [experiment plan](docs/experiment_plan.md).

## Configuration and checkpoints

Architecture, optimizer, augmentation, and scheduling settings are defined in [configs/base.yaml](configs/base.yaml) and inherited by experiment presets. Override them without editing code:

```bash
python -m cifar10 train --config configs/adamw.yaml \
  --set 'model.blocks=[5,5,5]' optimizer.lr=0.0003
```

The original four-architecture SGD/AdamW comparison remains available through `python -m cifar10 plan --config configs/all_experiments.yaml`. All configurations are checked against the parameter limit before training.

Each run saves its config, split, environment, per-epoch metrics, summary, and checkpoints:

- **best.pt:** highest validation-accuracy checkpoint.
- **last.pt:** latest saved epoch, including optimizer and random-number states.

The `plan` command automatically resumes matching interrupted experiments and skips completed ones. Use **Ctrl+C** to stop a terminal training process, then rerun the same command; work after the last saved checkpoint is repeated. Keep config and path settings consistent. The standalone `train` command requires explicit resume:

```bash
python -m cifar10 train --config runs/RUN/config.yaml --resume runs/RUN/last.pt
```

Final full-data training has no validation split and produces `last.pt` without `best.pt`.

## Reporting and verification

```bash
python -m cifar10 report --runs runs --readme README.md
python -m unittest discover -s tests -v
```

**13 tests passed at this milestone**, covering data loading, parameter limits, checkpoint resume, submission formatting, HPC launcher arguments, and stage generation. NYU cluster execution and the new staged training results remain unverified.

To generate learning curves:

```bash
python -m pip install -r requirements-plot.txt
python scripts/plot_training.py --run runs/RUN
```
