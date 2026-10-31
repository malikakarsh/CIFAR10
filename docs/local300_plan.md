# Sequential local experiment queue: 300 epochs

The active plan is **local Mac/MPS only**, with all ten trials running sequentially and **300 epochs per trial**. It supersedes the earlier single-priority and manually gated plans as the current execution choice. No selection or confirmation is required between these ten experiments. Preparing the queue has not launched it.

## Run

Stop any old local training process, then from the repository root:

```bash
source .venv/bin/activate
python -m cifar10 plan --config configs/local300/plan.yaml --dry-run
python -m cifar10 plan --config configs/local300/plan.yaml \
  --readme runs_local300/report.md
```

Use the same command after interruption. The plan resumes matching checkpoints and skips completed trials. Checkpoints, resolved configs, and metrics are stored in `runs_local300/`. A report is refreshed after each model completes. Batch progress is printed during training and validation.

## Queue

| Order | Architecture | Experiment |
|---|---|---|
| 1 | Wider, 2.44M | Crop/flip control |
| 2 | Wider, 2.44M | RandAugment: 2 operations, magnitude 7 |
| 3 | Wider, 2.44M | Mixup: alpha 0.2 |
| 4 | Wider, 2.44M | RandAugment + mixup |
| 5 | Wider, 2.44M | Random erasing: probability 0.25 |
| 6 | Wider, 2.44M | Dropout: probability 0.1 |
| 7 | Wider, 2.44M | Label smoothing: epsilon 0.05 |
| 8 | Wider, 2.44M | Combined regularization |
| 9 | Deeper + wider, 4.18M | Crop/flip control |
| 10 | Deeper + wider, 4.18M | Combined regularization |

Combined regularization uses RandAugment 2/7, mixup 0.2, random erasing 0.1, dropout 0.1, and label smoothing 0.05. It is an exploratory combination and may over-regularize; the earlier single-change trials help interpret its result. All models retain random crops and horizontal flips.

Every trial uses SGD with Nesterov momentum 0.9, LR 0.1, weight decay 0.0005, batch size 128, training seed 42, and split seed 42. Five warmup epochs are followed by cosine decay to 0.0001 over the **300-epoch** budget. The existing 45,000/5,000 training/validation split is retained.

These are fresh 300-epoch schedules. Existing 200-epoch model files are preserved, but cannot be resumed as this queue because changing the budget changes the configuration and LR schedule. The larger model has 4,180,858 parameters, below the 5,000,000 limit.

## Runtime and reporting

The queue contains **3,000 total epochs**. Prior Mac timings of about 49 seconds for wider models and 99 seconds for deeper+wider models imply roughly **50 hours before augmentation overhead**, assuming similar machine load. Actual speed will vary. Keep the Mac connected to power and awake; an optional macOS launch is:

```bash
caffeinate -i python -m cifar10 plan --config configs/local300/plan.yaml \
  --readme runs_local300/report.md
```

Compare validation accuracy and ordinary validation cross-entropy. Mixup and label smoothing change the training objective, so raw training losses across recipes are not directly comparable. Results from one seed remain provisional, and improvements in public accuracy are not guaranteed.

This queue covers all listed recipe experiments automatically. After it finishes, seed confirmation and final training on all 50,000 permitted images can be prepared from the selected config using `prepare-stage`; full-data retraining is not silently included because it depends on the eventual selected model. See the earlier [stage-generation instructions](experiment_plan.md).
