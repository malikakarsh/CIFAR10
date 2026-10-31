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
| Attention ResNet control | SGD | 4,534,066 | 300/300 | 300 | 0.4666 | 0.1324 | 97.14% |
| Pre-activation control | SGD | 4,696,394 | 400/400 | 389 | 0.6549 | 0.1689 | **97.26%** |
| Attention ResNet, A100 | SGD | 4,534,066 | 600/600 | 583 | 0.4427 | **0.1267** | 97.10% |

Losses and accuracy are reported at each run's highest-validation-accuracy epoch. Ten runs are complete; the larger AdamW trial has 143/200 epochs saved and remains incomplete. All runs use the same stratified 45,000/5,000 split. Training loss uses augmented images and the configured label smoothing; validation loss is ordinary cross-entropy on unaugmented images. Training losses across different objectives are not directly comparable.
<!-- RESULTS:END -->

| Public Kaggle submission | Accuracy |
|---|---:|
| Initial baseline submission during training | 79.50% |
| Completed baseline submission | 84.00% |
| Deeper + wider SGD submission | 85.30% |
| Attention ResNet control, intermediate epoch 156 | 81.10% |
| Attention ResNet control, completed epoch 300 | 87.80% |
| Pre-activation control, best epoch 389 of 400 | **88.80%** |
| Attention600, best epoch 583, mirror/translation TTA | **88.80%** |

Public scores are reported from leaderboard screenshots. Hidden evaluation labels are unavailable; evaluation loss and private accuracy cannot be calculated. Details: [submission history](docs/leaderboard.md).

The completed A100 attention600 run reached **97.10% best validation accuracy** (epoch 583), with **0.1267 validation loss**, and **88.80% reported public accuracy**. It ties the 400-epoch plain control on the public leaderboard. Its lower validation loss does not translate into a higher public score, and its validation accuracy does not exceed the 300-epoch attention baseline. The 300-epoch and 600-epoch submissions also differ in TTA and training platform/precision, so this is not an isolated epoch-budget comparison. Further epoch increases are not supported by these results. [600-epoch curves](docs/figures/attention600/learning_curves.png) ([PDF](docs/figures/attention600/learning_curves.pdf)). Published plots and metric snapshots are in `docs/figures/attention600/`; the imported raw run is under `runs_hpc/attention600/`, and its submission CSV is in `submissions/`. Raw runs and submission files remain Git-ignored.

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


## Attention model: prediction augmentation and 600-epoch trial

Generate a separate submission from the completed 300-epoch attention checkpoint with mirror and reflected one-pixel translation augmentation:

```bash
python -m cifar10 predict \
  --checkpoint runs_attention/attention_resnet_sweep000_20261031T025638Z_303819/best.pt \
  --device mps --tta --translation-tta \
  --output submissions/submission_attention300_tta.csv
```

This keeps the trained weights fixed, allowing comparison with the existing 87.80% standard-inference submission. Its augmented public score has not been reported; the separate 600-epoch TTA submission scored 88.80%. Prediction also saves a `.metadata.json` sidecar containing the checkpoint epoch, inference flags, and CSV hash.

A separate **600-epoch attention trial** uses the same 4,534,066-parameter model, augmentation, optimizer, split, label smoothing, and epoch-based cosine schedule as the completed attention control. The schedule spans 600 epochs; training starts from scratch. Mirror/translation augmentation is enabled for prediction, while checkpoint selection still uses ordinary unaugmented validation accuracy. The 600-epoch trial has completed: best validation accuracy 97.10%, validation loss 0.1267, and reported public accuracy 88.80%.

```bash
caffeinate -i python -m cifar10 plan \
  --config configs/attention/600_plan.yaml \
  --readme runs_attention600/report.md
```

Run this after the current training queue finishes. To resume the SLA comparison and then run attention600 automatically, stop the old queue with Ctrl+C and use `configs/attention/after_sla600.yaml` instead. It skips completed trials and resumes matching saved checkpoints. Only an unfinished epoch repeats. The new attention trial writes to `runs_attention600/`.


## A100 experiment queue

A resumable queue continues after the existing attention600 run, then tests **eight more 600-epoch trials**: a matched float32 SGD control and SAM (rho 0.05), CUDA SLA+SD600, a full-50k control, channel attention, mixup, RandAugment, and a second training seed. Every model stays below 5 million parameters. The queue can span multiple allocations; it does not assume all nine runs will fit within 20 hours. The ongoing local SLA400 run is unchanged.

```bash
nohup python -u scripts/queue_a100.py --project "$PWD" \
  > runs_hpc/a100_queue.log 2>&1 &
```

On HPC, first upload/extract the small code update and activate the existing environment as described in the [A100 queue guide](docs/a100_queue.md). The launcher waits for the current attention run without stopping it, resumes saved trials, and skips completed matching trials. New results go to `runs_hpc/a100_experiments/`. SAM and its control use float32; SLA uses CUDA AMP. The initial attention600 result is recorded above; no completed follow-up queue results have been imported yet.

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

**29 tests passed at the latest code verification**, covering data loading, parameter limits, checkpoint resume, submission formatting, HPC launcher arguments, stage generation, attention models, RAM caching, SLA joint-label alignment, detached self-distillation targets, head gradients, prediction augmentation, checkpoint inference, SAM analytical updates, BatchNorm handling, exact SAM resume, and queue locking. SAM execution on CUDA and completed SLA600 results remain unverified.

To generate learning curves:

```bash
python -m pip install -r requirements-plot.txt
python scripts/plot_training.py --run runs/RUN
```
