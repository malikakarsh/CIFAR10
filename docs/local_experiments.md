# Complementary local experiments

**Superseded as the current execution plan:** use [the local sequential 300-epoch queue](local300_plan.md). All ten trials run automatically. The instructions below are retained as optional staged/earlier workflows.

## Current priority: local only

Kaggle training has been stopped. All further experiments can run on the Mac. Start with one RandAugment trial against the already-completed local wider-SGD control:

```bash
source .venv/bin/activate
python -m cifar10 plan --config configs/local/priority_plan.yaml \
  --readme docs/local_stage_report.md
```

This schedules one 200-epoch run on MPS, with checkpoints in `runs_next/`. Review its validation results before deciding on mixup or the three independent regularization trials below. Their previous purpose as parallel work alongside Kaggle is now historical. Later confirmation, epoch-budget, scaling, and full-data stages can also be run locally using the selected local run configs as sources; their device and data settings are preserved. No new training was launched by this update.

Run these on the Mac if Kaggle continues the control/RandAugment/mixup stage. They explore different regularization methods rather than repeating the Kaggle trials. No local training has been started by preparing these configs.

All three use the existing local wider-SGD architecture (2,436,346 parameters), 200 epochs, batch size 128, seed 42, and split seed 42. Their reference is the completed local wider SGD model: **95.64% validation accuracy**, validation loss **0.1574** at the selected epoch. Each candidate changes one setting from that reference:

| Candidate | Change | Hypothesis |
|---|---|---|
| Random erasing | probability 0.25 | Reduce reliance on a small visible region |
| Dropout | probability 0.1 inside residual branches | Reduce dependence on particular activations |
| Label smoothing | epsilon 0.05 | Reduce overly confident class predictions |

These settings are starting trials, not established optima. RandAugment and mixup remain disabled here to isolate the individual changes. The baseline's default crop/flip augmentation remains enabled.

```bash
source .venv/bin/activate
python -m cifar10 plan --config configs/local/plan.yaml --dry-run
python -m cifar10 plan --config configs/local/plan.yaml \
  --readme docs/local_experiment_report.md
```

Stop any previous local training process before launching this plan. The experiments run sequentially and support automatic checkpoint resume. They write to `runs_next/`; old checkpoints are preserved.

At the previously measured wider-model speed of about 49 seconds per epoch, three runs would take roughly eight hours before recipe-dependent overhead. This is a local estimate, not a guaranteed duration. Check the first completed epochs for actual timings.

Select by validation accuracy and ordinary validation cross-entropy. Label smoothing changes the training objective, so its training loss should not be compared directly with unsmoothed training loss. Do not pool cross-hardware results as though the execution protocols were identical: use the local control for local candidates and the new CUDA control for Kaggle candidates.

If an individual method improves results, confirm it with a second training seed before combining it with the winning augmentation recipe. These three trials are optional complementary exploration; the staged confirmation/budget/scaling/full-data plan remains in [experiment_plan.md](experiment_plan.md). A 90% public score is not guaranteed by adding these methods.
