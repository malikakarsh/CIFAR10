# Resumable single-A100 experiment queue

The queue keeps the existing attention600 training intact, then runs eight more experiments sequentially. Every new trial trains for **600 epochs**, from scratch, with at most 5 million trainable parameters. It is a backlog across allocations, not a promise that all trials fit in 20 hours or reach a particular leaderboard score.

| Order | Trial | Parameters | Precision | Data |
|---|---|---:|---|---|
| Already running | Original attention600 | 4,534,066 | CUDA AMP | Fixed 45k/5k split |
| 1 | Pre-activation SGD control | 4,696,394 | Float32 | Fixed 45k/5k split |
| 2 | Matched SAM, rho 0.05, underlying SGD | 4,696,394 | Float32 | Fixed 45k/5k split |
| 3 | SLA+SD600, both heads included | 4,706,674 | CUDA AMP | Fixed 45k/5k split |
| 4 | Pre-activation control600, full permitted dataset | 4,696,394 | Float32 | All 50k training images |
| 5 | Matched control backbone with channel attention | 4,734,906 | Float32 | Fixed 45k/5k split |
| 6 | Control with mixup alpha 0.2 | 4,696,394 | Float32 | Fixed 45k/5k split |
| 7 | Control with RandAugment (2 operations, magnitude 9) replacing TrivialAugment | 4,696,394 | Float32 | Fixed 45k/5k split |
| 8 | Repeat control with training seed 43 | 4,696,394 | Float32 | Same fixed split seed 42 |

The original attention600 uses its original epoch-based schedule. The new pre-activation trials and the matched channel-attention trial use step-based cosine schedules spanning 600 epochs, five warmup epochs, LR 0.1, Nesterov SGD momentum 0.9, and weight decay 0.0005 excluding normalization/bias parameters. They keep batch size 128. Most use TrivialAugment before crop/flip, 12-pixel Cutout, and label smoothing 0.1. Mixup changes only mixup; RandAugment replaces TrivialAugment; the second seed changes training randomness without changing the split. Prediction settings use mirror/translation TTA, with a joint-head vote for SLA.

The full-data experiment is a scheduled baseline, not a result-selected winner. It has no validation loss or accuracy, and produces `last.pt`; its results must not be interpreted as a validation comparison. All other trials select `best.pt` using ordinary ten-class validation accuracy without TTA. No held-out evaluation images or labels enter training or checkpoint selection. Public scores are recorded only after the user submits a CSV.

## SAM implementation and validation

[Sharpness-Aware Minimization](https://arxiv.org/abs/2010.01412) uses a first forward/backward pass to construct a global L2-gradient perturbation of radius rho, then a second pass at perturbed weights. The implementation restores the original weights exactly before applying the second-pass SGD gradient. The second pass uses batch statistics without updating BatchNorm running statistics a second time. Weights and BN tracking settings are restored even when a second-pass exception occurs. Checkpoints retain the underlying SGD momentum and random-number states; no perturbation remains in saved weights.

SAM currently requires float32, SGD, and no SLA/auxiliary rotation objective; invalid combinations are rejected. The matched SGD control is also float32. Ordinary first-pass training loss/accuracy are logged for comparison with SGD; validation always uses the unperturbed model. Both SAM passes reuse the same augmented/mixed input. Tests verify a scalar analytical SAM update, exact restoration after exceptions, BN behavior, and exact CPU checkpoint/resume agreement. CUDA SAM execution remains to be validated on HPC; published SAM improvements do not establish improvements for this dataset's hidden evaluation set.

SLA600 is a new CUDA run, not a migration of the ongoing local 400-epoch SLA experiment. The local run's configuration and checkpoints are unchanged. The four rotated views make SLA more expensive than an ordinary epoch; SAM uses two optimization passes. Budget the remaining allocation using measured wall times rather than epoch count alone.

## Install the small update on the existing HPC project

Upload `cifar10-a100-queue.tar.gz` to the home folder through the existing browser VS Code Explorer. The update contains code, configs, tests and documentation, not another copy of the dataset. In the working PyTorch container:

```bash
cd "$HOME/cifar10-work/CIFAR-10"
tar -xzf "$HOME/cifar10-a100-queue.tar.gz" -C "$PWD"
source "$HOME/cifar10-hpc-env/bin/activate"
mkdir -p runs_hpc
nohup python -u scripts/queue_a100.py --project "$PWD" \
  > runs_hpc/a100_queue.log 2>&1 &
tail -f runs_hpc/a100_queue.log
```

Do not stop the already-running attention job. The launcher waits for its `.pipeline.lock`, checks/resumes the exact original attention600 configuration if necessary, and then starts the follow-up queue. A separate launcher lock prevents duplicate waiting queues. Original attention outputs stay in `runs_hpc/attention600/`; new experiments go to `runs_hpc/a100_experiments/`, including a report after each completed trial. Existing checkpoints are not removed. Optional CLI path overrides support other locations; keep the same lexical absolute data/output paths on subsequent allocations for exact config matching.

CPU preparation while waiting does not interrupt the current GPU training. Ctrl+C on `tail` only stops the log viewer. `nohup` handles ordinary terminal hangups, not Slurm timeout, cancellation, or an enforced container teardown. The allocation remains subject to NYU GPU-utilization rules. Preserve home/scratch outputs and back up checkpoints; do not assume a time-limited allocation backs up data.

On the next allocation, enter the same site PyTorch environment, activate the same venv, and rerun the same launcher command from the same project directory. Completed matching trials are skipped; the interrupted trial resumes at the last completed epoch, with a partial epoch repeated. A changed experiment config intentionally starts a separate run. Completed summary files and CSV metrics are retained; migrated storage paths require deliberate handling rather than silently bypassing resume checks.

For a preview without allocating a GPU or starting training:

```bash
python scripts/queue_a100.py --project "$PWD" --dry-run
```
