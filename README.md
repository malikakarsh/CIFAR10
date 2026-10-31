# CIFAR-10 ResNet Architecture Challenge

Configurable PyTorch experiments for CIFAR-10 classification under a **5-million-trainable-parameter limit**. The project supports residual CNN architecture comparisons, SGD and AdamW, resumable experiment plans, and staged training on Apple MPS or NVIDIA CUDA.

## Results

**Best completed validation accuracy: 97.26%. Best reported public Kaggle accuracy: 88.80%.**

<!-- RESULTS:START -->
| Architecture | Optimizer | Parameters | Epochs completed | Best epoch | Training loss | Validation loss | Validation accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline | SGD | 1,084,586 | 200/200 | 194 | 0.0032 | 0.1824 | 95.26% |
| Baseline | AdamW | 1,084,586 | 200/200 | 199 | 0.0005 | 0.4123 | 93.24% |
| Deeper | SGD | 1,860,522 | 200/200 | 199 | 0.0019 | 0.1826 | 95.38% |
| Deeper | AdamW | 1,860,522 | 200/200 | 191 | 0.0005 | 0.4550 | 93.56% |
| Wider | SGD | 2,436,346 | 200/200 | 200 | 0.0022 | 0.1574 | 95.64% |
| Wider | AdamW | 2,436,346 | 200/200 | 198 | 0.0002 | 0.3893 | 94.10% |
| Deeper + wider | SGD | 4,180,858 | 200/200 | 195 | 0.0016 | 0.1761 | 95.82% |
| Deeper + wider | AdamW | 4,180,858 | 143/200 | 136 | 0.0027 | 0.3771 | 94.34% |
| Attention ResNet control | SGD | 4,534,066 | 300/300 | 300 | 0.4666 | **0.1324** | 97.14% |
| Pre-activation control | SGD | 4,696,394 | 400/400 | 389 | 0.6549 | 0.1689 | **97.26%** |

Losses and accuracy are reported at each run's highest-validation-accuracy epoch. Nine runs are complete; the larger AdamW trial has 143/200 epochs saved and remains incomplete. All runs use the same stratified 45,000/5,000 split. Training loss uses augmented images and the configured label smoothing; validation loss is ordinary cross-entropy on unaugmented images. Training losses across different objectives are not directly comparable.
<!-- RESULTS:END -->

| Public Kaggle submission | Accuracy |
|---|---:|
| Initial baseline submission during training | 79.50% |
| Completed baseline submission | 84.00% |
| Deeper + wider SGD submission | 85.30% |
| Attention ResNet control, intermediate epoch 156 | 81.10% |
| Attention ResNet control, completed epoch 300 | 87.80% |
| Pre-activation control, best epoch 389 of 400 | **88.80%** |

Public scores are reported from leaderboard screenshots. Hidden evaluation labels are unavailable; evaluation loss and private accuracy cannot be calculated. Details: [submission history](docs/leaderboard.md).

The completed pre-activation control achieved **97.26% validation accuracy** and **88.80% reported public accuracy**, improving on the attention model by 0.12 and 1.00 percentage points respectively. Its best checkpoint is epoch 389 of 400. This submission uses the configured mirror/translation augmentation, whereas the attention submission used standard inference, so the public-score difference includes changes in inference as well as training.

The completed attention control achieved **97.14% validation accuracy**, improving on the previous 95.82% best by **1.32 percentage points**, with validation loss falling from 0.1761 to **0.1324**. Its checkpoint is `runs_attention/attention_resnet_sweep000_20261031T025638Z_303819/best.pt`. This result is from the ten-class attention model without an auxiliary rotation task or inference augmentation; its reported public score is **87.80%**, improving on 85.30% by **2.50 percentage points**. Matched completed SGD recipes also outperform their AdamW counterparts. These are single-seed validation results; see the earlier [model comparison](docs/model_comparison.md).


### Pre-activation control learning curves

![Pre-activation ResNet control: loss, accuracy, and learning rate over 400 epochs](docs/figures/preact_control_sweep000_20261031T071758Z_ed2534/learning_curves.png)

Thin lines show per-epoch metrics; bold lines show trailing 10-epoch means. Best validation accuracy is **97.26% at epoch 389**, with training loss **0.6549** and validation loss **0.1689**. Training uses augmentation and label smoothing; validation uses unaugmented images and ordinary cross-entropy. The **88.80% public score** is a separate submission result, not a curve measured during training. [Metric snapshot](docs/figures/preact_control_sweep000_20261031T071758Z_ed2534/metrics_snapshot.csv).

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

## Current experiment plan: Pre-activation ResNet + SLA

The plain pre-activation control has completed 400 epochs, with **97.26% validation accuracy** and **88.80% reported public accuracy**. The SLA+SD comparison is in progress; no completed SLA result is available yet.

| Trial | Blocks | Channels | Trainable parameters | Epochs |
|---|---|---|---:|---:|
| Pre-activation control | `[4,4,3]` | `[64,128,256]` | 4,696,394 | 400 |
| Pre-activation SLA+SD | `[4,4,3]` | `[64,128,256]` | 4,706,674 | 400 |

Both use 400 epochs, batch size 128, Nesterov SGD, five warmup epochs, cosine decay per optimizer step, TrivialAugmentWide, crop/flip, 12-pixel Cutout, and label smoothing 0.1. Weight decay excludes bias and normalization parameters. SLA learns 40 joint class/rotation labels and distills their aggregated predictions into a ten-class head. Its four rotated views increase computation per batch. Prediction supports mirror/translation augmentation and combined head probabilities.

```bash
source .venv/bin/activate
python -m cifar10 plan --config configs/sla/plan.yaml --dry-run
caffeinate -i python -m cifar10 plan --config configs/sla/plan.yaml \
  --readme runs_sla/report.md
```

The two models run sequentially on MPS and save checkpoints and measured losses under `runs_sla/`. Rerun the same command to resume from the last completed epoch. Stop any other local training process before starting. Existing attention checkpoints remain usable with their original architecture but cannot resume this new recipe.

Use `configs/sla/continue_current.yaml` to resume and finish the existing attention control before these two trials; stop the old process with Ctrl+C before switching plans. Use `configs/sla/only_sla.yaml` to run only SLA+SD. After validation selects the recipe, final training can use all 50,000 permitted images. No 90% hidden-set accuracy is claimed. See [objective, inference, and research references](docs/sla.md). Previous [attention experiments](docs/attention_resnet.md) and the [ten-experiment local queue](docs/local300_plan.md) remain available.

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

**22 tests passed at the latest code verification**, covering data loading, parameter limits, checkpoint resume, submission formatting, HPC launcher arguments, stage generation, attention models, RAM caching, SLA joint-label alignment, detached self-distillation targets, head gradients, prediction augmentation, and checkpoint inference. NYU cluster execution and SLA training results remain unverified.

To generate learning curves:

```bash
python -m pip install -r requirements-plot.txt
python scripts/plot_training.py --run runs/RUN
```
