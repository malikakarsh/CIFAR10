# Reported Kaggle results

| Reported date | Architecture/run | Submitted checkpoint | Public accuracy | Exact checkpoint epoch |
|---|---|---|---:|---|
| 2026-10-29 | First ResNet; `resnet_baseline_sweep000_20261029T224315Z_18e3b7` | `best.pt`, confirmed by user | 79.50% | Not recorded at prediction time |
| 2026-10-29 | First ResNet, submission reported after training completed | Completed-model submission; checkpoint provenance not independently verified | **84.00%** | Completed best.pt is epoch 194; no prediction-time manifest |
| 2026-10-30 | Deeper + wider SGD; `resnet_baseline_sweep003_20261030T114542Z_9bc762` | Latest submission reported after best-model prediction instructions | 85.30% | Saved best checkpoint is epoch 195; no prediction-time manifest |
| 2026-10-30 | Attention ResNet control, intermediate | `submissions/attention_intermediate.pt` | 81.10% | 156; frozen checkpoint verified |
| 2026-10-31 | Attention ResNet control, completed; `attention_resnet_sweep000_20261031T025638Z_303819` | `best.pt`, prediction command supplied before reported score | **87.80%** | 300; checkpoint verified |

The latest screenshot reports **87.80%**, an improvement of **2.50 percentage points** over 85.30%. The completed attention model achieved **97.14% validation accuracy**, with **0.4666 training loss** and **0.1324 validation loss** at epoch 300.

The third screenshot reports **85.30%**, improving on 84.00% by **1.30 percentage points**. The matching local run has 95.82% best validation accuracy.

The second screenshot reports 0.84000, an improvement of **4.50 percentage points** over 0.79500. The user supplied it after completing the first model and receiving instructions to submit its best checkpoint. The completed model's best local validation accuracy is 95.26%; the screenshot verifies the reported public score, not which local file was uploaded. Private accuracy and evaluation loss remain unknown.

The score comes from the user's Kaggle screenshot. The user confirmed generating `submissions/submission.csv` with `python -m cifar10 predict --checkpoint runs/resnet_baseline_sweep000_20261029T224315Z_18e3b7/best.pt --device mps --output submissions/submission.csv`.

The local CSV at the first-submission inspection contained 2,000 unique IDs, matching the sample-submission order and test ID set exactly; all labels were integers 0–9. Its SHA-256 was `a4de188c140a2177d1292b4bd648cfdb169847ddcb9a2d241bb838b66b2b9263`. This hash refers to the earlier local CSV, not necessarily the second submission. The screenshot alone does not verify the uploaded file's hash.

`best.pt` is replaced when validation accuracy improves. Without a prediction-time checkpoint record, its filename alone does not identify the exact weights used in an earlier submission. The plot snapshot's best validation accuracy was 91.22%, but that is not yet a confirmed same-checkpoint comparison with the 79.50% public score. Private accuracy and hidden evaluation loss remain unknown.
