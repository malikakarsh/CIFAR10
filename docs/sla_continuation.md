# Continue the proven SLA600 model, rather than restart a different experiment

**Completed outcome:** the user reports 800 epochs completed, selected epoch 748 with 97.98% validation accuracy and 0.107566 loss, but **89.40% public accuracy**, below original SLA600's 90.10%. This extension did not improve the public submission. The procedure below documents the completed experiment; it is not a recommendation to repeat it. [Reported results](sla600_extension_reported_results.json).

SLA600 achieved 90.10% public accuracy. A fresh SLA800 run changed both the training budget and split (45k/5k to 49k/1k) and scored 89.60%. It is not an extension of the original weights. Continuing the original model is a separate, cheaper hypothesis worth testing; it does not guarantee a higher score.

## Completed inference audit

`scripts/audit_sla_inference.py` evaluated both models on the intersection of their validation IDs: 1,000 images that were excluded from BOTH training sets. It checks training/validation disjointness and training metadata hashes before evaluation. Normalization and all spatial/joint-head operations match the submission implementation, with a unit test confirming equivalent policy probabilities. The evaluation used CPU float32; tiny differences from CUDA/MPS inference remain possible.

| Policy | SLA600 (epoch 588) | Fresh SLA800 (epoch 704) |
|---|---:|---:|
| Ordinary head, no TTA | 98.50% | 98.70% |
| Ordinary head, mirror/translation TTA | 98.30% | 98.60% |
| Joint head, mirror/translation TTA | 98.90% | 98.60% |
| Combined heads, mirror TTA | 98.90% | 98.50% |
| Combined heads, mirror/translation TTA (submitted policy) | 98.80% | 98.50% |

The models are nearly tied on this small common holdout, and the inference-policy differences are only a handful of images. This does not support projecting a 95% public score or generating many leaderboard submissions to choose a policy. The full audit is in `docs/sla_inference_audit.json`. Neither model trained on this common subset, but both checkpoints were previously selected using validation data that includes it; the audit is a diagnostic, not a new independent test set.

## Controlled 600-to-800 continuation

Use the original SLA600 run's **epoch-600 `last.pt`**, not its epoch-588 `best.pt`. The original terminal learning rate was zero; an explicit restart is required for nonzero updates. The first continuation trial uses peak LR 0.001, five warmup epochs, and cosine decay to zero across the next 200 epochs. It preserves the original 45,000/5,000 split, architecture, objective, augmentation, precision setting, batch size, weight decay, optimizer momentum, scaler and random-number states. Only epoch budget, LR schedule and branch output settings change. An explicitly requested device/data-root change is supported and still requires identical training metadata and split IDs; a different device does not promise identical numerics.

The new `extend` command validates that the source run is complete and rejects architecture, split, or objective changes. Ordinary `--resume` remains strict and cannot silently change configs. The saved branch config records `scheduler.restart_epoch=600`, so a later ordinary resume continues the same restarted schedule correctly. The source checkpoints are not modified. The best validation result is carried forward from the source; if continuation never improves it, the best-checkpoint pointer remains the original best.

The local imported SLA600 export contains `best.pt` only. Use the original `last.pt` at its existing HPC path in a new A100 allocation. The new core code must be uploaded before the command exists on HPC. In the site's working PyTorch environment, from the repository root:

```bash
nohup python -u -m cifar10 extend \
  --checkpoint runs_hpc/a100_experiments/a100_sla600_sweep000_20261101T013008Z_62b447/last.pt \
  --epochs 800 --restart-lr 0.001 --warmup-epochs 5 \
  --device cuda --data-root "$PWD/dataset" \
  --output-dir "$PWD/runs_hpc/sla600_extend800" \
  > runs_hpc/sla600_extend800.log 2>&1 &
```

After interruption, use the new branch's saved config and `last.pt` with `train --resume` (or a matching plan). Do not repeatedly invoke `extend` on the original source to resume a partially trained branch: that starts a separate continuation experiment. Do not run this concurrently with other GPU training.

At the earlier A100 rate of about 20 seconds per epoch, 200 extra epochs are roughly 67 minutes plus loading/checkpoint overhead. This is an estimate; actual runtime and generalization need measurement. Reject the extension as a replacement if it does not improve held-out validation; preserve the existing 90.10% submission regardless.

## Requested exact-checkpoint branch (not yet trained)

The user separately requested continuing the exact epoch-588 `best.pt` submitted for 90.10%. `scripts/run_best588_extension.py` checks its SHA-256 against the verified submission checkpoint, waits for known active training locks, then trains epochs 589–800 using the original recipe and a 212-epoch restarted schedule (peak LR 0.001, five warmup epochs, cosine decay). The source weights and original 45k/5k split are preserved, with new results under `runs_hpc/sla_best588_extend800`. It resumes matching partial branch checkpoints and skips a completed branch. If the allocation expires while waiting or training, rerun it in the next allocation.

```bash
nohup python -u scripts/run_best588_extension.py --project "$PWD" \
  > runs_hpc/best588_extension.log 2>&1 &
tail -f runs_hpc/best588_extension.log
```

The new core code must be uploaded first. Low-level branching from a selected checkpoint requires the explicit `extend --from-selected-checkpoint` option; default extension still requires a completed source, and normal resume still requires an unchanged config. This is a separate hypothesis from continuing epoch 600, not an established remedy for the public accuracy gap.

## A method change prepared separately: SAM + SLA

`configs/robustness/plan.yaml` compares a float32 SLA600 control with an otherwise identical float32 SLA600 + SAM (rho 0.05). Both contain 4,706,674 trainable parameters and use the original 45k/5k split. The SAM implementation evaluates the FULL SLA loss on both passes, including a newly computed detached self-distillation target on each pass, restores original weights before the SGD update, and avoids duplicate BatchNorm buffer updates. This is an independently implemented combination of [SAM](https://arxiv.org/abs/2010.01412) and [SLA](https://arxiv.org/abs/1910.05872), not a claim that either paper guarantees their combined benefit.

CPU tests verify that both passes call the joint objective, that both-head gradients are meaningful, and that interrupted/resumed training exactly matches uninterrupted weights. CUDA performance and accuracy are untested for this new combination. It is slower than ordinary SLA because it performs two four-view optimization passes. It should follow the short continuation test, rather than be another blind increase in epoch count.
