# Optional full-data follow-up for SLA+SD and SAM

The user subsequently chose 800-epoch experiments with a 49k/1k training/validation split instead. That active plan is implemented by `scripts/run_800_val1k.py`; this full-50k plan remains a separate alternative. Do not launch both plans as competing GPU processes.

The first A100 comparison reports SAM at 97.78% validation accuracy and 89.60% public accuracy, and SLA+SD at 97.86% validation accuracy and 90.10% public accuracy. Both methods’ exported checkpoints, configs, metric histories and submission records are now imported and verified locally. SLA is the leading candidate on both reported metrics; its 0.08-point validation advantage over SAM is only four images, which also motivates a second training seed.

The first queue's last configured trial is `a100_sgd_control_seed43`. The full-50k plain control is only its fourth trial. Without current remote log output, neither this local repository nor the assistant can establish which trial is actively running or how much allocation time remains. The new plan is prepared separately and does not modify the running queue.

| Order | New trial | Epochs | Parameters | Precision | Data |
|---|---|---:|---:|---|---|
| 1 | SLA+SD full-data | 800 | 4,706,674 | CUDA AMP | All 50,000 permitted training images |
| 2 | SAM rho 0.05 full-data | 800 | 4,696,394 | Float32 | All 50,000 permitted training images |
| 3 | SLA+SD, training seed 43 | 600 | 4,706,674 | CUDA AMP | Original fixed 45k/5k split, split seed 42 |
| 4 | SAM rho 0.05, training seed 43 | 600 | 4,696,394 | Float32 | Original fixed 45k/5k split, split seed 42 |

These inherit the measured methods' architecture, optimizer, augmentation, smoothing, and batch size. Full-data variants use all 50,000 permitted images and extend the per-step cosine schedule to 800 epochs, keeping the original five-epoch warmup and decay to zero. Their names include 800 to distinguish them from any earlier 600-epoch full-data runs; existing checkpoints remain preserved. This tests data size and schedule together, so it does not isolate either change. Repeatability variants retain the measured 600-epoch schedule and change only training seed and output/name settings. All weights initialize from scratch. The local 400-epoch SLA run, original A100 comparison, and their checkpoints remain unchanged. Full-data training has no validation metrics and uses the final `last.pt`; validation-seed repeats use `best.pt`. Mirror/translation TTA is preserved; SLA retains its joint-head vote.

The full-data runs test whether using the extra 5,000 permitted training images helps the selected methods. That is a hypothesis, not a guarantee of a higher public score. No evaluation images, labels, source matching, or manual labeling enter this plan. The two methods are deployed as separate models under the parameter budget, not a multi-model prediction ensemble. No 95% hidden-set accuracy is claimed.

## Start automatically after the current queue

Upload `cifar10-finalists800-update.tar.gz` to the home directory using the existing browser VS Code. This is a small config/launcher update, not another dataset upload. In the working PyTorch terminal:

```bash
cd "$HOME/cifar10-work/CIFAR-10"
tar -xzf "$HOME/cifar10-finalists800-update.tar.gz" -C "$PWD"
source "$HOME/cifar10-hpc-env/bin/activate"
mkdir -p runs_hpc
nohup python -u scripts/queue_finalists.py --project "$PWD" \
  > runs_hpc/finalists_queue.log 2>&1 &
tail -f runs_hpc/finalists_queue.log
```

This launcher first waits for the existing A100 launcher's lock, plus the output locks used by direct plan invocations. It then runs `configs/hpc_finalists/plan.yaml` sequentially. Its own launcher lock prevents duplicate followers, and the standard pipeline lock prevents two new plans from training into the same output directory. It neither stops nor appends to the already-loaded queue in memory. Start only one follower; do not independently launch another GPU training process alongside it.

Outputs go to `runs_hpc/a100_finalists/`, with a report refreshed after each completed trial. The backlog may span allocations, especially because SLA has four views per image and SAM has two optimization passes. A later GPU session can resume the same command, provided data, config and output paths stay consistent. Completed matching trials are skipped; only the unfinished epoch repeats. `nohup` does not override Slurm timeouts, cancellation, or container teardown. The GPU allocation is still subject to site utilization policy.

To inspect without launching training:

```bash
python scripts/queue_finalists.py --project "$PWD" --dry-run
```

If allocation time is limited, preserving a valid checkpoint and resuming later is preferable to changing the promised epoch budget mid-run. Use measured per-epoch times from the completed SLA/SAM runs and the portal's remaining allocation time to assess whether a full trial can finish.

## Updating a waiting launcher

The current launcher waits for the original A100 queue before it calls `run_plan` and loads the finalist YAML files. If a finalist launcher is already waiting, extract the new config update and leave that same launcher running; it will read the 800-epoch full-data configs when the wait ends. Do not start a duplicate launcher. A finalist model already training has loaded its config and is not changed by editing YAML; do not assume an active 600-epoch model has been extended. Any change to such an active run needs a deliberate restart as a separate 800-epoch experiment. Always confirm the preview shows `0/800 epochs` for the two full-data trials. The original queue and local runs remain unchanged.
