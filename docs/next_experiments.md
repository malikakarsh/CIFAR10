# Follow-up experiments after the current SGD comparison

The user requested AdamW next and authorized exploring other optimizers and architectures for better generalization. The current SGD configs and checkpoints remain unchanged. A combined resumable plan is now available: `python -m cifar10 plan --config configs/all_experiments.yaml`. It alternates SGD/AdamW for each architecture, adopts matching existing checkpoints, and skips completed runs. Stop the old sweep before starting it; it has not been launched by the agent. The optional six-setting tuning grid below remains separate.

## 1. Establish the measured baseline

Wait for the current comparison to finish. Read each run's metrics and select the best architecture by validation accuracy, using validation loss as a tie-breaker between runs. Keep the 45,000/5,000 split and split seed unchanged. Retain SGD as a candidate if it wins; AdamW is not guaranteed to outperform it.

## 2. Tune AdamW on the same architecture

`configs/adamw.yaml` starts at learning rate 0.001 and weight decay 0.01, with 200 epochs, five warmup epochs, and cosine decay to 0.000001. These are initial trial settings, not measured optima. The engine uses PyTorch's default AdamW betas (0.9, 0.999) and epsilon (1e-8). The inherited SGD momentum/Nesterov fields are unused by AdamW. Its weight decay is decoupled from the gradient update; do not assume SGD's learning rate or decay strength transfers unchanged. See the [PyTorch AdamW documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html).

Start with one fresh run, replacing the block/channel values with the winning SGD architecture:

```bash
python -m cifar10 train --config configs/adamw.yaml \
  --set 'model.blocks=[3,3,3]' 'model.channels=[32,64,128]'
```

`configs/tune_adamw.yaml` defines six combinations: learning rates 0.0003/0.001/0.003 and weight decay 0.0001/0.01. Preview before running:

```bash
python -m cifar10 sweep --config configs/tune_adamw.yaml --dry-run
python -m cifar10 sweep --config configs/tune_adamw.yaml \
  --set 'model.blocks=[3,3,3]' 'model.channels=[32,64,128]'
```

The full grid costs six fresh 200-epoch runs. A shorter screening budget may help eliminate unstable settings, but compare finalists at equal full budgets; short-run rankings may change later. Do not resume SGD optimizer state into an AdamW experiment.

## 3. Improve architecture and training separately

- Use the measured width/depth comparison to choose how to spend the parameter budget. Wider residual networks are a research-backed candidate, not a guarantee for this implementation or dataset split: [Wide Residual Networks](https://arxiv.org/abs/1605.07146). Our current post-activation network is not an exact reproduction of that paper.
- Next architecture candidates: pre-activation residual blocks and small channel-attention modules. Implement and count their parameters before scheduling runs. Every candidate must retain residual CNN structure and stay at or below 5,000,000 trainable parameters.
- Evaluate augmentation and regularization individually using the existing RandAugment, mixup, dropout, and label-smoothing controls. Changing all at once prevents attributing an improvement to a particular choice.
- Consider alternative optimizers only as explicit experiments with matched budgets and measured MPS runtime. Do not treat novelty as evidence of better accuracy.
- Re-run the strongest recipes with several training seeds while preserving the validation split. Select based on validation results, never hidden evaluation images or labels.

Training loss can increase with stronger augmentation or regularization even when validation accuracy improves. Track both losses, but optimize for the competition's accuracy metric. Report all measured outcomes in README, including unsuccessful trials; no improvement claims before measurement.
