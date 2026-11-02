# Bounded grouped-architecture experiment

## Why this experiment

The verified best remains SLA+SD+SAM600 at **91.20% public accuracy**, with **98.26% validation accuracy** and **0.0986498 validation loss** at epoch 598. Its RandAugment variant scored 90.60%, and continuing the winning checkpoint to epoch 700 did not surpass the source checkpoint. These outcomes do not support another epoch extension or repeating the same augmentation setting.

This trial changes the residual backbone to a **ResNeXt-style grouped bottleneck CNN** while preserving the successful training recipe. The design follows the cardinality/grouped-transformation idea in [Xie et al., Aggregated Residual Transformations for Deep Neural Networks, CVPR 2017](https://arxiv.org/abs/1611.05431). This project's 8x14-width model and its combination with SLA/self-distillation/SAM are experimental adaptations. The paper does not establish a score for this competition or this combination. The implementation is independent and trains from scratch; no pretrained weights are loaded.

## Fixed experiment

| Setting | Value |
|---|---|
| Architecture | 29-layer grouped residual bottleneck CNN |
| Blocks per stage | 3, 3, 3 |
| Output widths | 256, 512, 1024 |
| Groups | 8 |
| Bottleneck widths | 112, 224, 448 (base width 14) |
| Trainable parameters, including 10-class and 40-class heads | **4,942,098**, below 5,000,000 |
| Training budget | **360 epochs** |
| Split and seed | Original **45k/5k**, seed 42; fingerprint checked against verified baseline |
| Objective and optimizer | Original SLA + self-distillation + SAM rho 0.05, SGD/Nesterov |
| Precision and batch size | Float32, 128 |
| Augmentation | Original TrivialAugment, reflected crop, mirror and cutout 12 |
| Schedule | Original LR 0.1, five warmup epochs, per-step cosine to zero |
| Inference | Original mirror/translation TTA and joint-head vote |

The model, 360-epoch schedule and experiment output identity change from the 91.20% recipe. The architecture includes a post-activation grouped bottleneck block, rather than the previous pre-activation basic block. Its parameter allocation and normalization topology therefore differ intentionally. Configuration: `configs/architecture/resnext_sla_sam360.yaml`. The original 600-epoch configuration remains available separately; it is not modified by the restart.

## Allocation and automatic export

The first measured A100 epoch took **140.02 seconds**. At that rate 360 epochs take **14 hours**, before setup and export. The original 600-epoch plan would take approximately 23 hours 20 minutes. The user selected 360 epochs with a full cosine decay to zero. Runtime can vary; the restart defaults to a **14.15-hour training cap**, leaving the rest of the allocation for export. Reduce this cap if less allocation time remains.

Upload `cifar10-resnext360-update.tar.gz` to the HPC home directory using Burst Files. Connect to the allocated host and enter the existing PyTorch container (`bash /share/apps/pytorch/2.9.1/run-pytorch.bash`). Inside the container:

```bash
cd "$HOME/cifar10-work/CIFAR-10"
tar -xzf "$HOME/cifar10-resnext360-update.tar.gz" -C "$PWD"
source "$HOME/cifar10-hpc-env/bin/activate"
python -c 'import torch, matplotlib; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
mkdir -p runs_hpc/resnext_sla_sam360
nohup python -u scripts/restart_resnext360.py --project "$PWD" --max-hours 14.15 \
  >> runs_hpc/resnext_sla_sam360/launch.log 2>&1 &
tail -f runs_hpc/resnext_sla_sam360/launch.log
```

The helper preflights the new recipe and split, stops only the old 600-epoch architecture monitor and its training process group owned by this user in this project, preserves saved files, and launches the fresh 360-epoch schedule in a separate folder. An existing 360-epoch job is left running. It does not run the old model's remaining 600-epoch schedule. Install `requirements-plot.txt` first only if matplotlib is missing.

After this launch no terminal is needed to generate the submission. In **Burst Files → Home Directory**, open:

```text
cifar10-work/CIFAR-10/exports/resnext360_results/
```

Download:

- `submission_resnext360_tta.csv`: prediction from the highest-validation-accuracy checkpoint using mirror/translation TTA and joint-head voting.
- `learning_curves.png` or `learning_curves.pdf`: training and validation curves.
- `results.tar.gz`: full results bundle including checkpoints, metrics, curves and submission.
- `DOWNLOAD_READY.json`: written after every download file is published; completed training has `training_complete: true` and `epochs_completed: 360`.

Refresh the Files page after training finishes. This folder is created during export, not at training start. An orderly time-cap stop exports saved progress separately to `exports/resnext360_partial_results/`, with `training_complete: false`. Allocation expiry can kill export, so retain time for prediction and packaging. No Kaggle upload is automatic.

## Resume and verification

The runner signals only its own training process group at its deadline and retains the last completed epoch. To resume matching 360-epoch checkpoints in a later allocation, use the runner directly:

```bash
python -u scripts/run_resnext_sla_sam.py --project "$PWD" --epochs 360 --max-hours 14.15
```

Training that already completed is skipped and export is retried. Export alone can be retried with:

```bash
python -u scripts/export_resnext_sla_sam.py --project "$PWD" --epochs 360
# For incomplete saved training only:
python -u scripts/export_resnext_sla_sam.py --project "$PWD" --epochs 360 --partial
```

Raw runs live in `runs_hpc/resnext_sla_sam360/`. Detailed per-run exports remain in `exports/resnext29_8x14_sla_sam360_results/`; fixed portal paths are additional copies. Checkpoint epoch, config, split, submission ID order and hashes are checked during export. CPU tests cover architecture/parameter limits, head gradients, TTA, exact SAM/SLA resume, schedule endpoint, deadline handling, restart command targeting and portal publishing. Actual A100 training/export must run on HPC.

Preserve the 91.20% baseline and compare the candidate using the same validation images and inference policy. No competition score improvement is guaranteed.
