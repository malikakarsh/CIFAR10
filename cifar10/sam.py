"""Two-pass SAM building blocks; optimizer state stays in the underlying SGD.

Foret et al., https://arxiv.org/abs/2010.01412. This implementation deliberately
uses float32 training; AMP requires separate overflow handling for both passes.
"""
from contextlib import contextmanager

import torch
from torch import nn


@contextmanager
def perturb_weights(model, rho):
    parameters = [p for p in model.parameters() if p.grad is not None]
    if not parameters:
        raise RuntimeError('SAM requires first-pass gradients')
    norm = torch.linalg.vector_norm(torch.stack([p.grad.detach().norm(2) for p in parameters]))
    if not torch.isfinite(norm):
        raise RuntimeError('Nonfinite first-pass SAM gradient norm')
    scale = float(rho / norm.clamp_min(1e-12))
    originals = []
    try:
        with torch.no_grad():
            for parameter in parameters:
                originals.append((parameter, parameter.detach().clone()))
                parameter.add_(parameter.grad, alpha=scale)
        yield
    finally:
        with torch.no_grad():
            for parameter, original in originals:
                parameter.copy_(original)


@contextmanager
def batch_statistics_without_updates(model):
    """Use batch statistics on pass two, without recording a second BN update."""
    modules = [(m, m.track_running_stats) for m in model.modules()
               if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    try:
        for module, _ in modules:
            module.track_running_stats = False
        yield
    finally:
        for module, original in modules:
            module.track_running_stats = original
