import csv
import hashlib
import json
import math
import platform
import random
import subprocess
import time
from copy import deepcopy
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
    restart = s.get('restart_epoch', 0)
    epoch, total = epoch - restart, total - restart
    if epoch < s['warmup_epochs']:
        return base * (epoch + 1) / s['warmup_epochs']
    if s['name'] == 'constant':
        return base
    if s.get('cosine_epoch_end', False):
        progress = (epoch + 1 - s['warmup_epochs']) / (total - s['warmup_epochs'])
    else:
        progress = (epoch - s['warmup_epochs']) / max(1, total - s['warmup_epochs'] - 1)
    return s['min_lr'] + (base - s['min_lr']) * (1 + math.cos(math.pi * progress)) / 2


def optimizer_parameters(model, config):
    if not config.get('exclude_norm_bias_decay', False):
        return model.parameters()
    decayed, exempt = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (decayed if parameter.ndim > 1 else exempt).append(parameter)
    return [{'params': decayed, 'weight_decay': config['weight_decay']},
            {'params': exempt, 'weight_decay': 0.}]


def rotation_batch(images, targets):
    """One random quarter-turn per image; labels identify the chosen transform."""
    rotated = torch.zeros_like(images)
    for turn in range(4):
        mask = (targets == turn).view(-1, 1, 1, 1)
        rotated = torch.where(mask, images.rot90(turn, (-2, -1)), rotated)
    return rotated


@torch.inference_mode()
def evaluate(model, batches, device, progress_label=None):
    model.eval()
    loss_sum = correct = count = 0
    for batch_index, (images, labels) in enumerate(batches, 1):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += nn.functional.cross_entropy(logits, labels, reduction='sum').item()
        correct += (logits.argmax(1) == labels).sum().item()
        count += labels.size(0)
        if progress_label and (batch_index == 1 or batch_index % 10 == 0 or batch_index == len(batches)):
            print(f'{progress_label}: validation batch {batch_index}/{len(batches)}', flush=True)
    return {'loss': loss_sum / count, 'accuracy': correct / count}


def validate_extension(checkpoint, config, allow_selected_checkpoint=False):
    if checkpoint is None:
        raise ValueError('Extension requires a completed source checkpoint')
    old = checkpoint['config']
    completed = checkpoint['epoch'] + 1
    if config['training']['epochs'] <= completed or completed > old['training']['epochs']:
        raise ValueError('New total epoch budget must exceed the source epoch')
    if completed != old['training']['epochs'] and not allow_selected_checkpoint:
        raise ValueError('Extend from last.pt of a completed run to a larger total epoch budget')
    if allow_selected_checkpoint and (checkpoint.get('best_metrics') or {}).get('epoch') != completed:
        raise ValueError('Selected source must be its recorded best-metrics epoch')
    if config['scheduler'].get('restart_epoch') != completed:
        raise ValueError('Extension schedule must restart at the completed source epoch')
    expected = deepcopy(old)
    expected['experiment'].update(name=config['experiment']['name'], output_dir=config['experiment']['output_dir'])
    expected['training'].update(epochs=config['training']['epochs'], device=config['training']['device'])
    expected['data']['root'] = config['data']['root']
    expected['optimizer']['lr'] = config['optimizer']['lr']
    for key in ('name', 'warmup_epochs', 'min_lr', 'restart_epoch'):
        expected['scheduler'][key] = config['scheduler'][key]
    if expected != config:
        raise ValueError('Extension must preserve architecture, objective, split and optimizer settings except LR')


def extend_training(checkpoint_path, total_epochs, restart_lr=.001, warmup_epochs=5,
                    output_dir='runs_hpc/sla_continuation', device_name=None, data_root=None,
                    allow_selected_checkpoint=False):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    config = deepcopy(checkpoint['config'])
    config['experiment'].update(name=config['experiment']['name'] + f'_extend{total_epochs}',
                                output_dir=str(output_dir))
    if allow_selected_checkpoint:
        config['experiment']['name'] += f'_from{checkpoint["epoch"] + 1}'
    config['training']['epochs'] = total_epochs
    if device_name:
        config['training']['device'] = device_name
    if data_root:
        config['data']['root'] = str(data_root)
    config['optimizer']['lr'] = restart_lr
    config['scheduler'].update(name='cosine', warmup_epochs=warmup_epochs, min_lr=0.,
                               restart_epoch=checkpoint['epoch'] + 1)
    validate(config)
    validate_extension(checkpoint, config, allow_selected_checkpoint)
    return train(config, resume=checkpoint_path, extension=True,
                 allow_selected_checkpoint=allow_selected_checkpoint)


def train(config, resume=None, extension=False, allow_selected_checkpoint=False):
    checkpoint = torch.load(resume, map_location='cpu', weights_only=True) if resume else None
    if extension:
        validate_extension(checkpoint, config, allow_selected_checkpoint)
    elif checkpoint:
        if checkpoint['config'] != config:
            raise ValueError("Resume requires the saved config unchanged. Start a new run for new settings.")
    if config['scheduler'].get('restart_epoch', 0) and checkpoint is None:
        raise ValueError('A restarted schedule requires an extension source or a saved branch checkpoint')
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
    parameter_groups = optimizer_parameters(model, opt)
    if opt['name'] == 'sgd':
        optimizer = torch.optim.SGD(parameter_groups, lr=opt['lr'], momentum=opt['momentum'],
                                    nesterov=opt['nesterov'], weight_decay=opt['weight_decay'])
    else:
        optimizer = torch.optim.AdamW(parameter_groups, lr=opt['lr'], weight_decay=opt['weight_decay'])
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
        rotation_loss_sum = 0.
        rotation_weight = config['training'].get('rotation_weight', 0.)
        use_sla = config['training'].get('sla', False)
        sla_sums = dict.fromkeys(('joint', 'distillation', 'classification'), 0.)
        label = f'Epoch {epoch + 1}/{config["training"]["epochs"]}'
        print(f'{label}: starting training; workers={config["data"]["num_workers"]}; '
              f'waiting for first batch ({len(train_loader)} batches total)', flush=True)
        for batch_index, (images, labels) in enumerate(train_loader, 1):
            if config['scheduler'].get('per_step', False):
                restart = config['scheduler'].get('restart_epoch', 0)
                step = (epoch - restart) * len(train_loader) + batch_index - 1
                warmup = config['scheduler']['warmup_epochs'] * len(train_loader)
                steps = (config['training']['epochs'] - restart) * len(train_loader)
                peak = config['optimizer']['lr']
                floor = config['scheduler']['min_lr']
                if step < warmup:
                    lr = peak * (step + 1) / warmup
                elif config['scheduler']['name'] == 'constant':
                    lr = peak
                else:
                    progress = (step - warmup) / max(1, steps - warmup - 1)
                    lr = floor + (peak - floor) * (1 + math.cos(math.pi * progress)) / 2
                for group in optimizer.param_groups:
                    group['lr'] = lr
            if batch_index == 1:
                print(f'{label}: first batch loaded; starting forward/backward', flush=True)
            images, labels = images.to(device), labels.to(device)
            alpha = config['augmentation']['mixup_alpha']
            lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
            order = torch.randperm(labels.size(0), device=device) if alpha > 0 else torch.arange(labels.size(0), device=device)
            mixed = lam * images + (1 - lam) * images[order] if alpha > 0 else images
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                if use_sla:
                    from .sla import training_loss
                    loss, logits, sla_parts = training_loss(
                        model, images, labels, config['training']['label_smoothing'])
                else:
                    logits = model(mixed)
                    loss = lam * criterion(logits, labels) + (1 - lam) * criterion(logits, labels[order])
                if rotation_weight > 0:
                    rotation_targets = torch.randint(4, (images.size(0),), device=device)
                    rotation_logits = model.predict_rotation(rotation_batch(images, rotation_targets))
                    rotation_loss = nn.functional.cross_entropy(rotation_logits, rotation_targets)
                    loss = loss + rotation_weight * rotation_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at epoch {epoch + 1}")
            scaler.scale(loss).backward()
            sam_rho = opt.get('sam_rho', 0.)
            if sam_rho > 0:
                from .sam import perturb_weights, batch_statistics_without_updates
                with perturb_weights(model, sam_rho), batch_statistics_without_updates(model):
                    optimizer.zero_grad(set_to_none=True)
                    if use_sla:
                        perturbed_loss = training_loss(
                            model, images, labels, config['training']['label_smoothing'])[0]
                    else:
                        perturbed_logits = model(mixed)
                        perturbed_loss = (lam * criterion(perturbed_logits, labels)
                                          + (1 - lam) * criterion(perturbed_logits, labels[order]))
                    if not torch.isfinite(perturbed_loss):
                        raise RuntimeError(f'Nonfinite SAM second-pass loss at epoch {epoch + 1}')
                    perturbed_loss.backward()
            if config['training']['grad_clip'] > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * labels.size(0)
            if use_sla:
                for key, value in sla_parts.items():
                    sla_sums[key] += value.item() * labels.size(0)
            if rotation_weight > 0:
                rotation_loss_sum += rotation_loss.item() * labels.size(0)
            predicted = logits.argmax(1)
            correct += lam * (predicted == labels).sum().item() + (1 - lam) * (predicted == labels[order]).sum().item()
            samples += labels.size(0)
            if batch_index == 1 or batch_index % 50 == 0 or batch_index == len(train_loader):
                print(f'{label}: train batch {batch_index}/{len(train_loader)}; '
                      f'loss={total_loss / samples:.4f}; elapsed={time.monotonic() - begin:.1f}s', flush=True)
        if val_loader:
            print(f'{label}: training complete; starting validation', flush=True)
        validation = evaluate(model, val_loader, device, progress_label=label) if val_loader else None
        improved = validation is not None and validation['accuracy'] > best
        if improved:
            best = validation['accuracy']
            best_path = str((run / 'best.pt').resolve())
        metrics = {'epoch': epoch + 1, 'lr': lr, 'train_loss': total_loss / samples,
                   'train_accuracy_mixup_weighted': correct / samples,
                   'val_loss': validation['loss'] if validation else None,
                   'val_accuracy': validation['accuracy'] if validation else None,
                   'seconds': time.monotonic() - begin}
        if rotation_weight > 0:
            metrics['train_rotation_loss'] = rotation_loss_sum / samples
        if use_sla:
            metrics.update({f'train_{key}_loss': value / samples for key, value in sla_sums.items()})
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
def predict(checkpoint_path, output, data_root=None, device_name='auto', tta=None,
            translation_tta=None):
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
    use_translation = (use_tta and config['inference'].get('translation_tta', False)
                       if translation_tta is None else translation_tta)
    from .sla import probabilities as model_probabilities, inference_probabilities
    rotation_vote = config['inference'].get('rotation_vote', False)
    predictions = {}
    for images, ids in loader(ImageDataset(rows, config, test=True), config):
        images = images.to(device)
        if config['model'].get('architecture') in ('preact_resnet', 'resnext') or use_translation:
            probabilities = inference_probabilities(model, images, use_tta,
                use_translation, rotation_vote)
        else:
            probabilities = model_probabilities(model, images, rotation_vote)
            if use_tta:
                probabilities = (probabilities + model_probabilities(model, images.flip(-1), rotation_vote)) / 2
        predictions.update(zip(ids, probabilities.argmax(1).cpu().tolist()))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows((r['id'], predictions[r['id']]) for r in sample)
    metadata = {'checkpoint': str(checkpoint_path), 'checkpoint_epoch': state['epoch'] + 1,
                'parameters': count, 'rows': len(predictions), 'csv': str(output),
                'csv_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                'horizontal_flip_tta': bool(use_tta), 'translation_tta': bool(use_translation),
                'rotation_vote': rotation_vote, 'best_metrics': state.get('best_metrics')}
    output.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f"Wrote {len(predictions)} predictions to {output}; parameters: {count:,}")
    return output
