# Pre-activation ResNet with SLA and self-distillation

This experiment implements the joint class/transformation task and self-distillation described by [Lee, Hwang and Shin, ICML 2020](https://arxiv.org/abs/1910.05872), using [pre-activation residual blocks](https://arxiv.org/abs/1603.05027). It is a separate experiment from the earlier attention network and its auxiliary rotation task. All weights are initialized from scratch.

## Architecture and training

The backbone has three stages with `[4,4,3]` two-convolution residual blocks and `[64,128,256]` channels. Blocks use BN–ReLU before each convolution, an identity or projection shortcut, and no channel attention. Final BN–ReLU and global average pooling feed a ten-class head. The SLA variant also has a 40-output joint head. Counts include every trainable parameter in both heads:

| Trial | Parameters | Epochs | Status |
|---|---:|---:|---|
| Plain pre-activation control | 4,696,394 | 300 | Not trained |
| Pre-activation SLA+SD | 4,706,674 | 300 | Not trained |

Both trials use the fixed seed-42 stratified 45,000/5,000 split, batch size 128, Nesterov SGD with peak LR 0.1, momentum 0.9, weight decay 0.0005 on convolution/linear weights only, five warmup epochs, and cosine decay to zero updated each optimizer step. Batch size counts original images; SLA expands each batch to 512 views. The chosen duration retains the project's requested 300 epochs. Changing the batch size requires deliberately retuning LR; there is no implicit LR scaling.

Augmentation applies [TrivialAugmentWide](https://arxiv.org/abs/2103.10158), reflected-padding random crop, horizontal flip, normalization, and a 12-pixel [Cutout](https://arxiv.org/abs/1708.04552) mask. Ordinary classification uses label smoothing 0.1. No dropout, mixup, or separate four-output rotation classifier is enabled. Decoded images are cached in CPU RAM; augmentation runs on the CPU. This is not an accelerator-resident input pipeline.

## Objective

Each augmented image produces four rotations (0°, 90°, 180°, 270°). One shared backbone processes these views. The joint target is `4 * class + rotation`, giving 40 labels. Training minimizes the sum of:

1. Joint cross-entropy over all four views, without label smoothing.
2. KL divergence from a detached teacher distribution into the ordinary head on the upright view. The teacher averages the ten class logits corresponding to each view's known rotation, then applies softmax.
3. Label-smoothed ordinary ten-class cross-entropy on the upright view.

All three terms have weight 1. Metrics include `train_joint_loss`, `train_distillation_loss`, and `train_classification_loss`. `train_loss` is their sum; validation loss is ordinary unaugmented ten-class cross-entropy, so the two totals are not directly comparable. Checkpoints are selected by ordinary validation accuracy, consistently with previous runs.

Prediction averages ordinary logits over mirrored views and two reflected one-pixel diagonal translations. SLA also aggregates the matching joint-head rotation logits. After spatial TTA, the ordinary-head and joint-head softmax distributions receive equal weight. This uses two heads of one model. `--no-tta` disables mirror/translation augmentation, while retaining the configured joint-head vote.

## Run and resume

Stop any other local training process before starting this queue:

```bash
source .venv/bin/activate
python -m cifar10 plan --config configs/sla/plan.yaml --dry-run
caffeinate -i python -m cifar10 plan --config configs/sla/plan.yaml \
  --readme runs_sla/report.md
```

The control runs first, then SLA+SD. The same command resumes matching checkpoints at the last completed epoch after interruption. An interrupted partial epoch is repeated. Old attention checkpoints cannot initialize or resume this different architecture/objective; their runs remain available.

To finish an existing attention control before starting these two models, use `configs/sla/continue_current.yaml` instead. It preserves the attention control's original configuration for exact checkpoint matching, then schedules plain pre-activation control followed by SLA+SD. Stop the currently running plan with Ctrl+C before switching; its queue was loaded at startup and cannot be extended by editing YAML while it runs. Saved epochs are retained, and only an unfinished epoch repeats. Each run retains its metrics and checkpoints in its original output directory; the supplied report file is refreshed from the output directory of the model that just finished.

To run only SLA+SD with automatic resumption:

```bash
caffeinate -i python -m cifar10 plan --config configs/sla/only_sla.yaml \
  --readme runs_sla/report.md
```

After comparing validation results, freeze the selected recipe before final retraining on all 50,000 permitted images. Use `prepare-stage --stage final` with the selected run's saved config; it creates a separate plan and does not train automatically. Full-data runs have no validation loss or validation-based best checkpoint, so use their `last.pt`. Do not tune using hidden evaluation images. No accuracy result is claimed for this new recipe yet.
