import csv
import hashlib
import json
import math
import platform
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import torchvision
from torch import nn

from .config import save_config, validate
from .data import ImageDataset, loader, read_rows, train_rows, split_rows, split_manifest
from .models import build_model


def seed_all(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.backends.cudnn.benchmark = not deterministic


def device_for(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if name == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable to this Python process. Run from a macOS terminal with MPS-enabled PyTorch; no CPU fallback is used.")
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def rng_state(generator):
    state = {'python': random.getstate(), 'numpy': np.random.get_state()[1].tolist(),
             'numpy_rest': list(np.random.get_state()[2:]), 'torch': torch.get_rng_state(),
             'loader': generator.get_state()}
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state['mps'] = torch.mps.get_rng_state()
    return state


def restore_rng(state, generator):
    random.setstate(state['python'])
    np.random.set_state(('MT19937', np.array(state['numpy'], dtype=np.uint32), *state['numpy_rest']))
    torch.set_rng_state(state['torch'])
    generator.set_state(state['loader'])
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])
    if 'mps' in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state['mps'])


def atomic_save(value, path):
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def epoch_lr(config, epoch):
    base = config['optimizer']['lr']
    s, total = config['scheduler'], config['training']['epochs']
    if epoch < s['warmup_epochs']:
        return base * (epoch + 1) / s['warmup_epochs']
    if s['name'] == 'constant':
        return base
    progress = (epoch - s['warmup_epochs']) / max(1, total - s['warmup_epochs'] - 1)
    return s['min_lr'] + (base - s['min_lr']) * (1 + math.cos(math.pi * progress)) / 2


@torch.inference_mode()
def evaluate(model, batches, device):
    model.eval()
    loss_sum = correct = count = 0
    for images, labels in batches:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += nn.functional.cross_entropy(logits, labels, reduction='sum').item()
        correct += (logits.argmax(1) == labels).sum().item()
        count += labels.size(0)
    return {'loss': loss_sum / count, 'accuracy': correct / count}


def train(config, resume=None):
    checkpoint = torch.load(resume, map_location='cpu', weights_only=True) if resume else None
    if checkpoint:
        if checkpoint['config'] != config:
            raise ValueError("Resume requires the saved config unchanged. Start a new run for new settings.")
    validate(config)
    seed_all(config['experiment']['seed'], config['experiment']['deterministic'])
    model, parameters = build_model(config['model'])
    device = device_for(config['training']['device'])
    model.to(device)
    rows = train_rows(config)
    tr, va = split_rows(rows, config)
    manifest = split_manifest(rows, tr, va, config)
    if checkpoint and checkpoint['split'] != manifest:
        raise ValueError("Training metadata or split changed since checkpoint")
    generator = torch.Generator().manual_seed(config['experiment']['seed'])
    train_loader = loader(ImageDataset(tr, config, training=True), config, True, generator)
    val_loader = loader(ImageDataset(va, config), config) if va else None
    opt = config['optimizer']
    if opt['name'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=opt['lr'], momentum=opt['momentum'],
                                    nesterov=opt['nesterov'], weight_decay=opt['weight_decay'])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt['lr'], weight_decay=opt['weight_decay'])
    amp = config['training']['amp'] and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=config['training']['label_smoothing'])
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = Path(config['experiment']['output_dir']) / f"{config['experiment']['name']}_{stamp}_{uuid4().hex[:6]}"
    run.mkdir(parents=True, exist_ok=False)
    save_config(config, run / 'config.yaml')
    (run / 'split.json').write_text(json.dumps(manifest, indent=2))
    try:
        git_revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_revision = None
    metadata = {'parameters': parameters, 'device': str(device), 'python': platform.python_version(),
                'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
                'git_revision': git_revision, 'resume_from': str(resume) if resume else None}
    (run / 'environment.json').write_text(json.dumps(metadata, indent=2))
    start, best = 0, -1.0
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start, best = checkpoint['epoch'] + 1, checkpoint['best_accuracy']
        restore_rng(checkpoint['rng'], generator)
        if start >= config['training']['epochs']:
            raise ValueError("Checkpoint already completed all configured epochs")
    print(f"Run: {run}\nParameters: {parameters:,}; device: {device}; train: {len(tr)}; validation: {len(va)}", flush=True)
    best_path = checkpoint.get('best_checkpoint') if checkpoint else None
    best_metrics = checkpoint.get('best_metrics') if checkpoint else None
    for epoch in range(start, config['training']['epochs']):
        begin = time.monotonic()
        lr = epoch_lr(config, epoch)
        for group in optimizer.param_groups:
            group['lr'] = lr
        model.train()
        total_loss = correct = samples = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            alpha = config['augmentation']['mixup_alpha']
            lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
            order = torch.randperm(labels.size(0), device=device) if alpha > 0 else torch.arange(labels.size(0), device=device)
            mixed = lam * images + (1 - lam) * images[order] if alpha > 0 else images
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(mixed)
                loss = lam * criterion(logits, labels) + (1 - lam) * criterion(logits, labels[order])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at epoch {epoch + 1}")
            scaler.scale(loss).backward()
            if config['training']['grad_clip'] > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * labels.size(0)
            predicted = logits.argmax(1)
            correct += lam * (predicted == labels).sum().item() + (1 - lam) * (predicted == labels[order]).sum().item()
            samples += labels.size(0)
        validation = evaluate(model, val_loader, device) if val_loader else None
        improved = validation is not None and validation['accuracy'] > best
        if improved:
            best = validation['accuracy']
            best_path = str((run / 'best.pt').resolve())
        metrics = {'epoch': epoch + 1, 'lr': lr, 'train_loss': total_loss / samples,
                   'train_accuracy_mixup_weighted': correct / samples,
                   'val_loss': validation['loss'] if validation else None,
                   'val_accuracy': validation['accuracy'] if validation else None,
                   'seconds': time.monotonic() - begin}
        if improved:
            best_metrics = metrics
        with (run / 'metrics.csv').open('a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics))
            if epoch == start:
                writer.writeheader()
            writer.writerow(metrics)
        state = {'config': config, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'scaler': scaler.state_dict(), 'epoch': epoch, 'best_accuracy': best,
                 'best_checkpoint': best_path, 'best_metrics': best_metrics,
                 'split': manifest, 'rng': rng_state(generator)}
        atomic_save(state, run / 'last.pt')
        if improved:
            atomic_save(state, run / 'best.pt')
        summary = {'run': str(run), 'name': config['experiment']['name'],
                   'parameters': parameters, 'epochs_completed': epoch + 1,
                   'best_metrics': best_metrics, 'final_metrics': metrics,
                   'split_sha256': hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
                   'device': str(device),
                   'best_val_accuracy': best if va else None, 'best_checkpoint': best_path,
                   'last_checkpoint': str((run / 'last.pt').resolve())}
        (run / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(metrics), flush=True)
    return summary


@torch.inference_mode()
def predict(checkpoint_path, output, data_root=None, device_name='auto', tta=None):
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    config = state['config']
    if data_root:
        config['data']['root'] = str(data_root)
    d = config['data']
    rows = read_rows(Path(d['root']) / d['test_csv'], [d['id_column']])
    sample = read_rows(Path(d['root']) / d['sample_submission'], ['id', 'label'])
    if set(r[d['id_column']] for r in rows) != set(r['id'] for r in sample):
        raise ValueError("Test IDs and sample submission IDs do not match")
    model, count = build_model(config['model'])
    model.load_state_dict(state['model'])
    device = device_for(device_name)
    model.to(device).eval()
    use_tta = config['inference']['horizontal_flip_tta'] if tta is None else tta
    predictions = {}
    for images, ids in loader(ImageDataset(rows, config, test=True), config):
        images = images.to(device)
        probabilities = model(images).softmax(1)
        if use_tta:
            probabilities = (probabilities + model(images.flip(-1)).softmax(1)) / 2
        predictions.update(zip(ids, probabilities.argmax(1).cpu().tolist()))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows((r['id'], predictions[r['id']]) for r in sample)
    print(f"Wrote {len(predictions)} predictions to {output}; parameters: {count:,}")
    return output
