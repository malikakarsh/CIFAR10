# Two experiments for the 15-hour A100 allocation

Use **one A100, 15 hours, six CPU cores and 16 GB RAM**. The queue runs these two experiments sequentially:

| Order | Experiment | Starting weights | Epochs | Settings |
|---|---|---|---|---|
| 1 | SLA+SD+SAM with RandAugment | Fresh initialization, seed 42 | 600 | Replace TrivialAugment with RandAugment N=2, M=5; preserve the rest of the 91.20% recipe |
| 2 | Original SLA+SD+SAM continuation | Exact submitted epoch-598 `best.pt` that scored 91.20% | 599–700, 102 additional epochs | Original TrivialAugment, SAM and split preserved; peak LR 0.0001, five warmup epochs and cosine decay |

Both use **4,706,674 trainable parameters**, float32, SAM rho 0.05, batch size 128 and the original 45k/5k split. Trial 2 starts from the **original winning combined-method checkpoint**, not the RandAugment weights, original standalone SLA epoch 588, or an epoch-600 `last.pt`. Checksum and epoch verification occur before either trial trains. The source model is preserved.

The earlier combination took about 10.05 hours; the extra 102 epochs are roughly 1 hour 43 minutes at its measured rate. Fifteen hours provides setup/export buffer, but RandAugment's actual runtime and both experiments' accuracy remain unmeasured. More compute is not a guarantee of improved public accuracy.

## Upload and launch

Connect to the allocated host and enter the working site PyTorch environment using `bash /share/apps/pytorch/2.9.1/run-pytorch.bash`. Upload `cifar10-sla-sam-two-experiments-code.tar.gz` to the HPC home directory via Burst Files. Inside the PyTorch container terminal:

```bash
cd "$HOME/cifar10-work/CIFAR-10"
tar -xzf "$HOME/cifar10-sla-sam-two-experiments-code.tar.gz" -C "$PWD"
source "$HOME/cifar10-hpc-env/bin/activate"
python -c 'import torch, matplotlib; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
mkdir -p runs_hpc
nohup python -u scripts/run_sla_sam_15h_queue.py --project "$PWD" \
  >> runs_hpc/sla_sam_15h_queue.log 2>&1 &
tail -f runs_hpc/sla_sam_15h_queue.log
```

If matplotlib is missing: `python -m pip install -r requirements-plot.txt`. The code package contains no dataset or checkpoints; use their existing HPC copies. No dependency reinstall is needed when the current environment works.

Only start this combined launcher. It holds locks against the earlier known queues, runs trial 1, exports it, then automatically starts trial 2 and exports that. Do not also launch the standalone RandAugment or continuation script. A held lock causes a prompt failure rather than competing GPU training; it does not cancel the other process.

Optional preflight without CUDA training: `python scripts/run_sla_sam_15h_queue.py --project "$PWD" --dry-run`.

## Resume and exports

Rerun the **same combined launch command** after reconnecting in a new allocation. Finished training is skipped; matching partial experiments resume from saved completed epochs. A partial epoch is repeated. If the first training fails or is interrupted, the second does not begin. An export failure is reported and leaves saved training intact; the next training experiment still proceeds so a plotting/packaging error does not waste the allocation. Export retries are listed at the end.

Training outputs:

- `runs_hpc/sla_sam_randaugment/`
- `runs_hpc/sla_sam_best598_extend700/`

Each exporter writes submission CSV/metadata, best/last weights, config, summary, split, complete metric history, curves and file hashes into a separate run folder under:

- `exports/sla_sam_randaugment600_results/`
- `exports/sla_sam_best598_extended700_results/`

The log prints exact `.tar.gz` download paths and CSV paths. Both submissions use mirror/translation TTA and configured joint-head voting. Source and previous submission files are preserved. No Kaggle upload happens automatically.

The continuation curve combines original epochs 1–598 with branch epochs 599–700. Resumed segments are stitched by exact config/split; segment hashes are recorded. If continuation never improves validation accuracy, its selected best checkpoint legitimately remains the source epoch 598, and the exporter supports that case. The source and branch loss curves show a learning-rate restart; they are not a fresh 700-epoch schedule.

To retry only an export after training:

```bash
python -u scripts/export_sla_sam_randaugment.py --project "$PWD"
python -u scripts/export_sla_sam_best598.py --project "$PWD"
```

`scripts/run_sla_sam_best598.py` supports running/resuming only the continuation separately. Do not run it alongside the queue.
