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

The latest screenshot confirms **88.80%** for `submission_attention600_tta.csv`, tying the 400-epoch plain control. Its best checkpoint, epoch 583, recorded **97.10% validation accuracy**, **0.4427 training loss**, and **0.1267 validation loss**. This establishes no public-score gain over the plain control. Relative to attention300, both the epoch budget and inference augmentation changed, as did training hardware/precision.

The plain-control screenshot reports **88.80%**, improving on 87.80% by **1.00 percentage point**. The pre-activation control recorded **97.26% validation accuracy**, **0.6549 training loss**, and **0.1689 validation loss** at its best epoch, 389. Validation is unaugmented; submission inference includes mirror/translation augmentation. The public comparison with the attention submission therefore does not isolate architecture or epoch count.

The attention screenshot reports **87.80%**, an improvement of **2.50 percentage points** over 85.30%. The completed attention model achieved **97.14% validation accuracy**, with **0.4666 training loss** and **0.1324 validation loss** at epoch 300.

The third screenshot reports **85.30%**, improving on 84.00% by **1.30 percentage points**. The matching local run has 95.82% best validation accuracy.

The second screenshot reports 0.84000, an improvement of **4.50 percentage points** over 0.79500. The user supplied it after completing the first model and receiving instructions to submit its best checkpoint. The completed model's best local validation accuracy is 95.26%; the screenshot verifies the reported public score, not which local file was uploaded. Private accuracy and evaluation loss remain unknown.

The score comes from the user's Kaggle screenshot. The user confirmed generating `submissions/submission.csv` with `python -m cifar10 predict --checkpoint runs/resnet_baseline_sweep000_20261029T224315Z_18e3b7/best.pt --device mps --output submissions/submission.csv`.

The local CSV at the first-submission inspection contained 2,000 unique IDs, matching the sample-submission order and test ID set exactly; all labels were integers 0–9. Its SHA-256 was `a4de188c140a2177d1292b4bd648cfdb169847ddcb9a2d241bb838b66b2b9263`. This hash refers to the earlier local CSV, not necessarily the second submission. The screenshot alone does not verify the uploaded file's hash.

`best.pt` is replaced when validation accuracy improves. Without a prediction-time checkpoint record, its filename alone does not identify the exact weights used in an earlier submission. The plot snapshot's best validation accuracy was 91.22%, but that is not yet a confirmed same-checkpoint comparison with the 79.50% public score. Private accuracy and hidden evaluation loss remain unknown.
