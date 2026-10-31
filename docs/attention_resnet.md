# Attention ResNet experiments

This is an experimental architecture implemented within this project's PyTorch model/training interfaces. It combines established methods from the papers cited below; the combination is not a claim that those methods were invented here. No external trained weights are loaded. Accuracy beyond the current measured baseline is unverified.

## Architecture

- Three stages, with **[3,3,4] blocks** and **[64,128,224] channels**.
- Pre-activation residual paths with identity/projection shortcuts.
- Squeeze-and-excitation channel gates, reduction ratio 16.
- Spatial dropout probability 0.05 inside residual branches.
- Global average pooling and a 10-class head.
- Optional separate four-class rotation head, sharing the backbone.

Verified trainable parameter counts are **4,534,066** without the rotation head and **4,534,966** with it. Both include every trainable module and remain below 5,000,000. The shared model forward always returns 10-class logits for standard validation and submission prediction.

## Training and data

Both trials use 300 epochs, batch 128, seed 42, and the existing split seed 42 (45k training / 5k validation). Nesterov SGD uses momentum 0.9, LR 0.1, weight decay 0.0005 on multidimensional weights only, and no decay on biases/normalization parameters. Five warmup epochs precede cosine decay to 0.0001. Only the new configs opt into this parameter grouping; existing recipes retain their optimizer behavior.

Training augmentation is reflected crop, random horizontal flip, TrivialAugmentWide, and a zero-valued mask up to 10x10 in normalized pixel space. Label smoothing is 0.05. Validation uses ordinary cross-entropy on clean normalized training-holdout images. Mixup and erasing remain disabled for this comparison.

The optional RAM cache decodes each dataset's PNGs once per process/run into an RGB uint8 array. Subsequent epochs read those arrays and apply the configured transforms. This is a CPU RAM cache, not an accelerator-resident training pipeline. It uses about 154 MB of pixel storage for all 50k training images plus Python/library overhead, and is rebuilt when a process resumes. There is no persistent cache that can silently become stale. Images for evaluation are never used to train the model or choose its recipe.

## Auxiliary rotation task

The second trial adds one random quarter-turn (0, 90, 180, or 270 degrees) per training image. The regular image supplies the classification loss; a second backbone pass on the rotated image supplies four-way rotation cross-entropy. The combined objective is:

`smoothed_classification_loss + 0.25 * rotation_loss`

There is no joint class/transformation label space or teacher/distillation term. Rotation labels are derived solely from transformations applied to permitted training images. The auxiliary head is not needed for classification inference, but all its parameters remain counted in the checkpoint. This extra pass increases compute; new model speed and peak memory must be measured on the Mac. If the rotation variant does not improve validation results, retain the control.

## Run on the Mac

Stop another local training process before starting this separate queue. Existing run directories are preserved.

```bash
source .venv/bin/activate
python -m cifar10 plan --config configs/attention/plan.yaml --dry-run
caffeinate -i python -m cifar10 plan --config configs/attention/plan.yaml \
  --readme runs_attention/report.md
```

The control runs first, then the rotation variant, **300 epochs each**. Matching checkpoints resume automatically; completed trials are skipped. Files go to `runs_attention/`. These architecture configs start from scratch and are incompatible with old model weights. Old resolved configurations and checkpoint state layouts remain supported by their original architecture path.

For a quick separate smoke run, override `training.epochs=2 scheduler.warmup_epochs=0 experiment.output_dir=runs_attention/smoke`. Do not compare its accuracy to a full 300-epoch run. MPS execution and competition performance have not been benchmarked from the agent session; functional tests use CPU synthetic data.

Select the recipe on the existing validation split. Confirm promising differences with another training seed. Final retraining on all 50k images is a later run with the selected recipe and epoch budget; it produces last.pt without validation. Do not use supplied external weights or hidden labels.

## Method references

- He et al., [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027).
- Hu et al., [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507).
- Gidaris et al., [Unsupervised Representation Learning by Predicting Image Rotations](https://arxiv.org/abs/1803.07728). Our auxiliary supervised-training objective is an experimental use of this task.
- Müller and Hutter, [TrivialAugment](https://arxiv.org/abs/2103.10158), applied through torchvision.
- DeVries and Taylor, [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/abs/1708.04552).

## Longer schedule and spatial prediction augmentation

`configs/attention/600.yaml` inherits the original attention control and changes the training budget to 600 epochs, the output directory to `runs_attention600`, and prediction settings to mirror/translation augmentation with inference batch size 64. It does not add an auxiliary task or change the architecture. All 4,534,066 parameters train from scratch; cosine decay runs over the full 600 epochs with the original five-epoch warmup and minimum LR 0.0001.

The completed 300-epoch checkpoint remains the 97.14% validation / 87.80% reported public baseline. `predict --tta --translation-tta` can apply spatial augmentation to that existing checkpoint without editing its saved config. The prediction rule averages logits over the original crop (weight 0.5) and two reflected one-pixel diagonal translations (weight 0.25 each), with equal mirror averaging within each crop. It then applies softmax. Legacy mirror-only inference remains unchanged unless translation is requested.

Validation still uses the ordinary ten-class head without TTA, so its metrics remain comparable across the 300- and 600-epoch attention trials. Compare their submissions using the same inference settings. Neither the augmented 300-epoch public score nor any 600-epoch result is known yet. Prediction writes the checkpoint epoch, flags and CSV SHA-256 to a `.metadata.json` sidecar; obtain public scores only through normal competition submissions.
