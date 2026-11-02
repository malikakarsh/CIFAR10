# Reported Kaggle results

| Reported date | Architecture/run | Submitted checkpoint | Public accuracy | Exact checkpoint epoch |
|---|---|---|---:|---|
| 2026-10-29 | First ResNet; `resnet_baseline_sweep000_20261029T224315Z_18e3b7` | `best.pt`, confirmed by user | 79.50% | Not recorded at prediction time |
| 2026-10-29 | First ResNet, submission reported after training completed | Completed-model submission; checkpoint provenance not independently verified | **84.00%** | Completed best.pt is epoch 194; no prediction-time manifest |
| 2026-10-30 | Deeper + wider SGD; `resnet_baseline_sweep003_20261030T114542Z_9bc762` | Latest submission reported after best-model prediction instructions | 85.30% | Saved best checkpoint is epoch 195; no prediction-time manifest |
| 2026-10-30 | Attention ResNet control, intermediate | `submissions/attention_intermediate.pt` | 81.10% | 156; frozen checkpoint verified |
| 2026-10-31 | Attention ResNet control, completed; `attention_resnet_sweep000_20261031T025638Z_303819` | `best.pt`, prediction command supplied before reported score | **87.80%** | 300; checkpoint verified |
| 2026-10-31 | Pre-activation control; `preact_control_sweep000_20261031T071758Z_ed2534` | `best.pt`, prediction command supplied with configured mirror/translation TTA | **88.80%** | 389 of 400; saved summary verified |
| 2026-10-31 | Attention600 on A100; `attention_resnet600_sweep000_20261031T174911Z_744138` | `submission_attention600_tta.csv`, local hash matches exported metadata; mirror/translation TTA | **88.80%** | 583 of 600; exported summary and metadata verified |
| 2026-11-01 | Pre-activation SAM600 on A100 | `submission_sam600_tta.csv`, screenshot confirms filename and score; mirror/translation TTA | 89.60% | 575 of 600; local checkpoint, config, metrics and CSV hash verified |
| 2026-11-01 | Pre-activation SLA+SD600 on A100 | `submission_sla600_tta.csv`, screenshot confirms filename and score; mirror/translation TTA and configured joint-head vote | **90.10%** | 588 of 600; local checkpoint, config, metrics and CSV hash verified |
| 2026-11-01 | Plain control600, all 50k training images | `submission_control600_full50k_tta.csv`, screenshot confirms filename/score; final checkpoint and TTA per supplied command | 88.70% | 600; CSV hash, inference metadata, final summary and full-data split verified locally |
| 2026-11-01 | SLA800, 49k/1k training/validation | `submission_sla800_49k_val1k_tta.csv`; checkpoint epoch and local CSV hash verified; same configured TTA/joint vote | 89.60% | 704 of 800; does not beat SLA600 |
| 2026-11-01 | SLA+SD+SAM600, original 45k/5k split | `submission_sla_sam600_tta.csv`; mirror/translation TTA and joint-head vote; local artifacts verified | **91.20%** | 598 of 600; 98.26% validation accuracy, 0.0986498 loss |
| 2026-11-02 | SLA+SD+SAM + RandAugment600 | `submission_sla_sam_randaugment600_tta.csv`; score confirmed by screenshot, raw export pending | 90.60% | 586 of 600; 98.04% validation accuracy and 0.104238 loss from supplied summary |

The latest screenshot confirms **90.60%** for the RandAugment trial, **0.60 percentage points below** the preserved 91.20% baseline. The separate continuation completed 700 epochs without surpassing its original epoch-598 best checkpoint; it has no separately reported public score. [Reported outcomes](sla_sam_two_trial_reported_results.json).

The previous screenshot confirms **91.20%** for `submission_sla_sam600_tta.csv`, a **1.10 percentage-point improvement** over original SLA600. The imported export verifies selected epoch **598**, **98.26% validation accuracy**, **0.0986498 validation loss**, 4,706,674 parameters and the original 45k/5k split. The local CSV SHA-256 matches its inference metadata; checkpoint, config, full history and curves have been checked. [Submission record](sla_sam600_submission.json).

The earlier screenshot reports **89.40%** for `submission_sla600_extended800_tta.csv`, the original SLA600 continued to 800 epochs. Its selected epoch 748 achieved 97.98% validation accuracy and 0.107566 loss on the original 5k split, according to the user-pasted summary. Compared with original SLA600, selected validation accuracy increased by 0.12 percentage points but public accuracy decreased by 0.70 points. This continuation did not improve the submission. Weights, history and CSV from this run have not yet been imported locally. [Reported results](sla600_extension_reported_results.json).

The prior screenshot reports **89.60%** for fresh SLA800, 0.50 points below SLA600’s 90.10%. Its selected 98.70% validation accuracy was measured on a different, smaller 1k split and without submission TTA; it was not directly comparable with SLA600’s 5k validation accuracy. A common 1k holdout audit using the submitted policy gives 98.80% for SLA600 versus 98.50% for SLA800.

The full-data control screenshot reports **88.70%** for the full-50k plain control. It does not beat the earlier plain control (88.80%) or the current SLA+SD best (90.10%). The 0.10-point difference from the earlier plain control is small; data split, epoch budget, and execution platform changed, so this does not isolate an effect of training-data size. There is no validation loss/accuracy for this full-data run. The downloaded CSV, inference metadata, final summary and 50k/zero-validation split fingerprint are verified locally; checkpoint weights and per-epoch history were not included.

The SAM/SLA screenshots confirm **89.60% for SAM600** and **90.10% for SLA+SD600**. SLA improves on SAM by **0.50 percentage points** and on the previous 88.80% best by **1.30 points**. Public scores and CSV filenames are verified from screenshots; exported metadata, CSVs, checkpoints, configs and 600-epoch histories have now been imported and cross-checked locally. Private accuracy remains unknown.

The attention600 screenshot confirms **88.80%** for `submission_attention600_tta.csv`, tying the 400-epoch plain control. Its best checkpoint, epoch 583, recorded **97.10% validation accuracy**, **0.4427 training loss**, and **0.1267 validation loss**. This establishes no public-score gain over the plain control. Relative to attention300, both the epoch budget and inference augmentation changed, as did training hardware/precision.

The plain-control screenshot reports **88.80%**, improving on 87.80% by **1.00 percentage point**. The pre-activation control recorded **97.26% validation accuracy**, **0.6549 training loss**, and **0.1689 validation loss** at its best epoch, 389. Validation is unaugmented; submission inference includes mirror/translation augmentation. The public comparison with the attention submission therefore does not isolate architecture or epoch count.

The attention screenshot reports **87.80%**, an improvement of **2.50 percentage points** over 85.30%. The completed attention model achieved **97.14% validation accuracy**, with **0.4666 training loss** and **0.1324 validation loss** at epoch 300.

The third screenshot reports **85.30%**, improving on 84.00% by **1.30 percentage points**. The matching local run has 95.82% best validation accuracy.

The second screenshot reports 0.84000, an improvement of **4.50 percentage points** over 0.79500. The user supplied it after completing the first model and receiving instructions to submit its best checkpoint. The completed model's best local validation accuracy is 95.26%; the screenshot verifies the reported public score, not which local file was uploaded. Private accuracy and evaluation loss remain unknown.

The score comes from the user's Kaggle screenshot. The user confirmed generating `submissions/submission.csv` with `python -m cifar10 predict --checkpoint runs/resnet_baseline_sweep000_20261029T224315Z_18e3b7/best.pt --device mps --output submissions/submission.csv`.

The local CSV at the first-submission inspection contained 2,000 unique IDs, matching the sample-submission order and test ID set exactly; all labels were integers 0–9. Its SHA-256 was `a4de188c140a2177d1292b4bd648cfdb169847ddcb9a2d241bb838b66b2b9263`. This hash refers to the earlier local CSV, not necessarily the second submission. The screenshot alone does not verify the uploaded file's hash.

`best.pt` is replaced when validation accuracy improves. Without a prediction-time checkpoint record, its filename alone does not identify the exact weights used in an earlier submission. The plot snapshot's best validation accuracy was 91.22%, but that is not yet a confirmed same-checkpoint comparison with the 79.50% public score. Private accuracy and hidden evaluation loss remain unknown.
