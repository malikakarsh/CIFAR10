# Next experiment plan

## Objective and starting point

Improve generalization under the 5,000,000-parameter constraint. Current best results are **95.82% validation accuracy** and **85.30% reported public Kaggle accuracy**. A 90% public score is a target, not a predicted outcome.

Start from the **2.44M wider ResNet**: blocks `[3,3,3]`, channels `[48,96,192]`, batch size 128, SGD with Nesterov momentum 0.9, initial LR 0.1, weight decay 0.0005, five warmup epochs, and cosine decay. Keep the existing stratified 45,000/5,000 split and split seed 42 throughout model selection. All configurations train from scratch.

The next stages are deliberately conditional. Run stage 1 first; inspect its results before preparing later stages. No new training has been started, and later stages are not automatically queued.

| Stage | Experiment | Runs | Epochs per run | Decision |
|---|---|---:|---:|---|
| 1 | Control; RandAugment; RandAugment + mixup | 3 | 200 | Select the recipe to confirm |
| 2 | Selected recipe and control with seed 43 | 2 maximum | 200 | Check whether the improvement repeats |
| 3 | Confirmed recipe with a longer cosine schedule | 1 | 300 | Retain 300 only if it improves over 200 |
| 4 | Selected recipe/budget on the 4.18M model | 1 | 200 or 300 | Select architecture versus the matching wider reference |
| 5 | Selected model and recipe on all permitted training images | 1 | Selected budget | Generate final predictions from last.pt |

Stages 3 and 4 can be skipped if the evidence does not justify their cost. The full path is up to eight new training runs, at most 1,900 epochs. It is not another architecture/optimizer Cartesian sweep. A minimum-cost path runs stage 1, confirms a promising change, and retrains the selected model on all training images. If no new recipe improves, retain the existing SGD recipe and diagnose the validation/public gap before expanding the search.

## Stage 1: augmentation

The three configurations in `configs/stages/01_augmentation.yaml` share architecture, SGD settings, seed 42, hardware/precision settings, and 200 epochs:

| Trial | RandAugment operations / magnitude | Mixup alpha |
|---|---|---:|
| Control | Disabled | 0 |
| RandAugment | 2 / 7 | 0 |
| RandAugment + mixup | 2 / 7 | 0.2 |

Dropout, erasing, and label smoothing stay disabled. The new control is needed when moving from Mac/MPS to HPC/CUDA and mixed precision; use its results as the matched comparison. Do not interpret hardware or precision changes as an augmentation effect.

Inside a **Burst Code Server terminal with an allocated GPU and prepared Python environment**, from the repository root:

```bash
python -m cifar10 plan --config configs/stages/01_augmentation.yaml --dry-run
python -m cifar10 plan --config configs/stages/01_augmentation.yaml \
  --readme runs_hpc/report.md
```

The default data location is `dataset/` and output is `runs_hpc/`. For a different data path, append `--set data.root=/absolute/path/to/dataset` and keep it unchanged on reruns. The plan resumes matching interrupted runs and skips completed ones. Create the report directory before a dry-run-only workflow if needed; actual training creates it.

For an NYU **batch job**, after configuring the appropriate environment, account, and partition for your allocation:

```bash
mkdir -p logs
sbatch --account=YOUR_ACCOUNT scripts/hpc_train.sbatch configs/stages/01_augmentation.yaml
```

On Burst, add the partition assigned to your class if required; do not assume Torch allocation settings apply. The HPC wrapper normalizes data/output paths and sets CUDA plus four workers. Continue using the same launch method and paths for all stages so exact config matching works.

For Mac/MPS instead:

```bash
python -m cifar10 plan --config configs/stages/01_augmentation.yaml \
  --readme docs/local_stage_report.md \
  --set training.device=mps data.num_workers=0 experiment.output_dir=runs_next
```

For the historical float32 protocol on CUDA, explicitly set `training.amp=false` for every stage. Otherwise the new CUDA runs use AMP. Do not change precision, batch size, split seed, or worker settings midway through a comparison.

## Stage 2: confirm the selected recipe

Choose by validation accuracy, inspecting validation loss and per-class behavior too. Keep the control as a candidate if augmentation does not help. A tiny single-seed gain is provisional; it is not evidence of a guaranteed public-score increase.

Use the selected **stage-1 seed-42 run's resolved config**, rather than editing a guessed winning YAML:

```bash
python -m cifar10 prepare-stage \
  --source runs_hpc/SELECTED_STAGE1_RUN/config.yaml \
  --stage confirm --output-dir configs/generated/02_confirm

python -m cifar10 plan --config configs/generated/02_confirm/plan.yaml \
  --readme runs_hpc/report.md
```

This creates a matched control and selected-recipe run with training seed 43, keeping split seed 42. If the selected recipe is already the control, only one new control run is generated. Compare each recipe over seeds 42 and 43 using mean validation accuracy and the individual results; ties favor lower validation loss and lower cost. Two seeds are an initial repeatability check, not a precise uncertainty estimate. If the candidate gain fails to repeat, retain the control or investigate before adding more changes.

## Stage 3: test whether more epochs help

Only after recipe confirmation, prepare a 300-epoch run from that recipe's **200-epoch seed-42 config**:

```bash
python -m cifar10 prepare-stage \
  --source runs_hpc/CONFIRMED_RECIPE_SEED42_RUN/config.yaml \
  --stage budget --output-dir configs/generated/03_budget

python -m cifar10 plan --config configs/generated/03_budget/plan.yaml \
  --readme runs_hpc/report.md
```

It starts fresh and recalculates cosine decay over 300 epochs, preserving the source seed. Compare with the same recipe/seed at 200 epochs. Do not resume a completed 200-epoch run after changing its config. A longer schedule can also change checkpoint selection opportunities; confirm small gains before attributing them to extra optimization. If the longer budget does not justify its cost, retain 200 epochs.

## Stage 4: transfer to the larger architecture

Use the selected 200- or 300-epoch recipe as source:

```bash
python -m cifar10 prepare-stage \
  --source runs_hpc/SELECTED_RECIPE_AND_BUDGET_RUN/config.yaml \
  --stage scale --output-dir configs/generated/04_scale

python -m cifar10 plan --config configs/generated/04_scale/plan.yaml \
  --readme runs_hpc/report.md
```

The generator changes blocks to `[5,5,5]`, retains channels `[48,96,192]`, and preserves the seed and training recipe. The model has **4,180,858 trainable parameters**. Compare with the selected 2.44M reference at the same budget/seed, then choose based on validation accuracy, loss, and cost. The larger model is not automatically the winner.

## Stage 5: final full-data training

Freeze architecture, recipe, inference policy, and epoch budget based on validation results. Then:

```bash
python -m cifar10 prepare-stage \
  --source runs_hpc/SELECTED_FINAL_RECIPE_RUN/config.yaml \
  --stage final --output-dir configs/generated/05_final

python -m cifar10 plan --config configs/generated/05_final/plan.yaml \
  --readme runs_hpc/report.md
```

This changes only the experiment name and `data.val_fraction=0.0`, training from scratch on all **50,000 permitted training images**. It produces **last.pt**, with no validation or best.pt. Use that final checkpoint for inference. The loss and private accuracy of the hidden evaluation set remain unavailable. Training epochs on 50,000 images include more updates than on 45,000; record that protocol change.

For batch execution of generated stages, replace the `python -m cifar10 plan` command with `sbatch --account=YOUR_ACCOUNT scripts/hpc_train.sbatch configs/generated/STAGE/plan.yaml`, including the correct Burst partition if required. In an allocated Code Server session, run Python directly rather than requesting a second GPU job.

## Artifacts and selection rules

- Preserve each run's config, split, metrics, summary, best.pt, and last.pt. Resume keeps original run directories; preserve referenced prior best checkpoints too.
- Generated plans are written under `configs/generated/` and ignored by Git because they may contain machine-specific data/output paths. `prepare-stage` does not choose a winner or launch training. It refuses to overwrite different generated settings; use a new output directory for another selection.
- Refresh reports with `python -m cifar10 report --runs runs_hpc --readme runs_hpc/report.md` and plot selected runs with `scripts/plot_training.py`.
- Compare inference with and without horizontal-flip TTA on validation before freezing the final inference policy. TTA validation and per-class diagnostics are follow-up analysis, not automatically run by this plan.
- Select recipes using permitted validation data. Do not train on hidden evaluation images or recover their labels from external sources. Avoid selecting among a large number of recipes using repeated public submissions.
- This plan targets improved generalization; it does not promise 90% public accuracy.
