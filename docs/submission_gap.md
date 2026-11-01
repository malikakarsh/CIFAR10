# Validation and public submission gap

The strongest reported public result remains SLA600 at **90.10%**. The fresh 49k/1k SLA800 scored **89.60%**. It changed both the epoch budget and validation split; it did not continue the original SLA600 weights.

## Checks completed locally

- Re-evaluated both checkpoints on the same 1,000 images excluded from both training sets, using the submitted combined-head, rotation, mirror and translation policy: **SLA600 98.80%, SLA800 98.50%**. These images participated in previous checkpoint selection, so this is a diagnostic, not an independent test set. The previous headline comparison used different validation cohorts and inference policies. See [inference audit](sla_inference_audit.json).
- Checked all 52,000 provided images: all are 32x32 RGB PNG files. The existing training split audit found no identical pixel duplicates crossing training/validation. No hidden labels or external data were obtained.
- Verified SLA600, SAM600 and SLA800 CSVs against sample-submission IDs/order, label range and recorded hashes. All contain exactly 2,000 predictions; no export mismatch was found.
- Compared aggregate pixel statistics. Training and validation are close (largest absolute standardized mean difference across the eight measured features: 0.022). Test differences are larger but modest (0.093–0.239): the test set is brighter and has higher within-image contrast and adjacent-pixel differences. This is descriptive evidence of a distribution difference; it does **not** explain the full score gap, identify its cause, or establish a corrective preprocessing rule. Test statistics are not used to modify normalization, select a checkpoint, or tune augmentation.
- SLA600 and SAM600 disagree on 114/2,000 predictions; SLA600 and SLA800 disagree on 95/2,000. Hidden labels are unavailable, so disagreement alone cannot identify the better predictions.

Full measurements and a reproducible command are in [submission gap audit](submission_gap_audit.json) and `scripts/audit_submission_gap.py`:

```bash
python scripts/audit_submission_gap.py --submissions \
  submissions/submission_sla600_tta.csv \
  submissions/submission_sam600_tta.csv \
  submissions/submission_sla800_49k_val1k_tta.csv
```

No demonstrated serialization, ID ordering or inference-policy bug explains the gap. More epochs alone are not a supported solution. Preserve the 90.10% submission and evaluate method changes on the original fixed 5,000-image validation split. Do not compare the 49k-trained model against all 5,000 old validation images: 4,000 of those were in its training set.

## Next HPC queue

1. Continue the original epoch-600 SLA600 `last.pt` to epoch 800 on the original 45k/5k split, with peak LR 0.001, five warmup epochs and cosine decay. This isolates continuation from changing the split; it is not a guaranteed improvement.
2. Train **SLA+SD+SAM600** (rho 0.05, float32) on that same 45k/5k split, **4,706,674 parameters**. This combines the strongest prior method with a changed optimizer. The matched float32 SLA control is available in `configs/robustness/plan.yaml`; the priority queue runs only the new combination after continuation. Comparing it with the existing AMP SLA600 alone cannot isolate SAM from the precision change. CUDA runtime and accuracy for this combination are unmeasured.

Standalone SAM800 and older architecture sweeps are not part of this queue. Request **one A100 for 12 hours** as a planning estimate: roughly 1–1.5 hours for continuation and potentially 6–8 hours for the new combination, plus setup/export buffer. Measure the new method's actual epoch rate and revise this estimate; `nohup` cannot keep training alive beyond the allocation. The launcher resumes a partial continuation branch and the plan resumes the new combination from its saved epoch.

After uploading/extracting the code update into the existing HPC project, enter the previously verified site PyTorch environment and run:

```bash
cd "$HOME/cifar10-work/CIFAR-10"
source "$HOME/cifar10-hpc-env/bin/activate"
mkdir -p runs_hpc
nohup python -u scripts/run_sla_followup.py --project "$PWD" \
  > runs_hpc/sla_followup.log 2>&1 &
tail -f runs_hpc/sla_followup.log
```

The original checkpoint, dataset and environment stay on HPC. The launcher fails promptly if a previous queue holds the known locks, or if the original `last.pt` is missing. It does not terminate existing jobs. Rerun the same launcher after an allocation ends to resume. No HPC process has been started by these local changes.
