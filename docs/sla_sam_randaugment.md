# Controlled augmentation trial after the 91.20% submission

Baseline: the verified SLA+SD+SAM600 epoch-598 checkpoint achieved **98.26% validation accuracy**, **0.0986498 validation loss** and **91.20% reported public Kaggle accuracy**. This new trial changes only the augmentation policy: TrivialAugment is replaced by RandAugment with **two operations and magnitude 5**. It preserves cutout, mirror/crop training augmentation, the original 45k/5k split and seed 42, the 4,706,674-parameter model, the full SLA/self-distillation objective, SAM rho 0.05, float32, batch size 128, optimizer and 600-epoch schedule. It trains from scratch; it does not fine-tune the submitted weights. Accuracy improvement is unproven.

`configs/robustness/sla_sam_randaugment_plan.yaml` contains exactly one experiment. This is distinct from the older `sla_randaugment.yaml`, which uses standalone mixed-precision SLA without SAM.

## HPC launch

Request **one A100, 12 hours, six CPU cores and 16 GB RAM**. The original combination took approximately 10.05 hours for training alone. The new augmentation's runtime is unmeasured; inspect early epoch times and allow time for export. Allocation expiry ends `nohup` jobs too; rerun the launcher in a new allocation to resume saved completed epochs. A partial epoch is repeated.

Connect to the allocated host using the existing account key and enter the verified site PyTorch environment:

```bash
bash /share/apps/pytorch/2.9.1/run-pytorch.bash
```

Upload `cifar10-sla-sam-randaugment-code.tar.gz` to the HPC home directory through Burst Files. In the PyTorch container terminal:

```bash
cd "$HOME/cifar10-work/CIFAR-10"
tar -xzf "$HOME/cifar10-sla-sam-randaugment-code.tar.gz" -C "$PWD"
source "$HOME/cifar10-hpc-env/bin/activate"
python -c 'import torch, matplotlib; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
mkdir -p runs_hpc/sla_sam_randaugment
nohup python -u scripts/run_sla_sam_randaugment.py --project "$PWD" \
  >> runs_hpc/sla_sam_randaugment/launch.log 2>&1 &
tail -f runs_hpc/sla_sam_randaugment/launch.log
```

If matplotlib is missing, install the existing plotting requirements before launching: `python -m pip install -r requirements-plot.txt`. Existing dataset, checkpoints and virtual environment remain in place; the code archive contains no data or weights.

The launcher verifies the baseline config and recorded split fingerprint before training, prints the GPU and parameter count, rejects known competing queue locks, resumes matching saved checkpoints and skips a completed trial. Do not change config or project/data/output paths between resumes. It does not cancel another job. `--dry-run` checks recipe, split and planned resume state without CUDA training. `--skip-export` runs training only.

## Submission, curves and download bundle

After training completes, export runs automatically using the selected best checkpoint with mirror/translation TTA and configured joint-head voting. The log prints exact paths. Outputs are under:

```text
exports/sla_sam_randaugment600_results/<run-name>/
  submission_sla_sam_randaugment600_tta.csv
  submission_sla_sam_randaugment600_tta.metadata.json
  curves/<run-name>/learning_curves.png
  curves/<run-name>/learning_curves.pdf
  best.pt
  last.pt
  config.yaml
  summary.json
  metrics.csv
  split.json
  environment.json
  history_segments.json
  export_manifest.json
```

The sibling `<run-name>_results.tar.gz` is the download bundle. The export follows the summary's best-checkpoint pointer even if the best weights are in an earlier resumed segment. It combines matching metric segments into a complete 600-epoch history for the curves, checks submission ID order and records file hashes. Epoch-time measurements can differ between repeated epochs; later matching segments take precedence when stitching history.

If prediction or plotting is interrupted after training, rerun only the export:

```bash
python -u scripts/export_sla_sam_randaugment.py --project "$PWD"
```

Download the CSV through Burst Files to submit on Kaggle, and the bundle for local verification/README updates. Preserve the existing 91.20% checkpoint and compare both completed models using the same validation images and inference settings. No automatic Kaggle upload occurs.
