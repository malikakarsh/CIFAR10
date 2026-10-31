from copy import deepcopy
from pathlib import Path
import math
import yaml


# Optional extensions are accepted in inherited files without adding keys to old
# resolved configs. Existing checkpoint/config equality therefore stays intact.
OPTIONAL = {
    'model': {'architecture', 'se_reduction', 'rotation_head', 'sla_head'},
    'data': {'cache_images'},
    'augmentation': {'trivialaugment', 'cutout_size', 'trivialaugment_first'},
    'training': {'rotation_weight', 'sla'},
    'inference': {'rotation_vote', 'translation_tta'},
    'scheduler': {'per_step'},
    'optimizer': {'exclude_norm_bias_decay', 'sam_rho'},
}


def merge(base, updates, section=None):
    result = deepcopy(base)
    for key, value in updates.items():
        if key not in result and key not in OPTIONAL.get(section, set()):
            raise ValueError(f"Unknown config key: {key}")
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge(result[key], value, section=key)
        else:
            result[key] = deepcopy(value)
    return result


def set_value(config, key, value):
    parts = key.split(".")
    node = config
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            raise ValueError(f"Unknown config key: {key}")
        node = node[part]
    if parts[-1] not in node and parts[-1] not in OPTIONAL.get(parts[0] if len(parts) == 2 else None, set()):
        raise ValueError(f"Unknown config key: {key}")
    node[parts[-1]] = value


def load_config(path, overrides=(), _seen=None):
    path = Path(path).resolve()
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        raise ValueError(f"Circular config inheritance: {path}")
    seen.add(path)
    with path.open() as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping")
    parent = config.pop("extends", None)
    if parent:
        config = merge(load_config(path.parent / parent, _seen=seen), config)
    for item in overrides:
        key, value = item.split("=", 1)
        set_value(config, key, yaml.safe_load(value))
    validate(config)
    return config


def validate(c):
    m, t, d, a = c['model'], c['training'], c['data'], c['augmentation']
    if not m['blocks'] or len(m['blocks']) != len(m['channels']):
        raise ValueError("model.blocks and channels must be nonempty lists of equal length")
    if any(type(v) is not int or v < 1 for v in m['blocks'] + m['channels']):
        raise ValueError("Block counts and channels must be positive integers")
    if m['kernel_size'] < 1 or m['kernel_size'] % 2 != 1:
        raise ValueError("kernel_size must be a positive odd integer")
    if m['num_classes'] != 10 or not 0 < m['max_parameters'] <= 5_000_000:
        raise ValueError("Competition requires 10 classes and at most 5,000,000 parameters")
    for name, value in [('dropout', m['dropout']), ('val_fraction', d['val_fraction']),
                        ('label_smoothing', t['label_smoothing'])]:
        if not 0 <= value < 1:
            raise ValueError(f"{name} must be in [0, 1)")
    if t['epochs'] < 1 or t['batch_size'] < 1 or c['inference']['batch_size'] < 1:
        raise ValueError("Epochs and batch sizes must be positive")
    if d['num_workers'] < 0 or len(d['mean']) != 3 or len(d['std']) != 3 or min(d['std']) <= 0:
        raise ValueError("Invalid workers or RGB normalization")
    if c['optimizer']['name'] not in ('sgd', 'adamw') or c['optimizer']['lr'] <= 0:
        raise ValueError("Use sgd or adamw with positive learning rate")
    if c['scheduler']['name'] not in ('cosine', 'constant'):
        raise ValueError("scheduler.name must be cosine or constant")
    if not 0 <= c['scheduler']['warmup_epochs'] < t['epochs']:
        raise ValueError("warmup_epochs must be >= 0 and < epochs")
    if not 0 <= c['scheduler']['min_lr'] <= c['optimizer']['lr']:
        raise ValueError("min_lr must be between 0 and optimizer.lr")
    if a['mixup_alpha'] < 0 or not 0 <= a['random_erasing'] <= 1:
        raise ValueError("Invalid mixup_alpha or random_erasing")
    if m.get('architecture', 'resnet') not in ('resnet', 'attention_resnet', 'preact_resnet'):
        raise ValueError('Unknown architecture')
    if type(m.get('se_reduction', 16)) is not int or m.get('se_reduction', 16) < 1:
        raise ValueError('se_reduction must be a positive integer')
    if type(a.get('cutout_size', 0)) is not int or not 0 <= a.get('cutout_size', 0) <= 32:
        raise ValueError('cutout_size must be in [0, 32]')
    if a.get('trivialaugment', False) and a['randaugment_ops']:
        raise ValueError('Select either TrivialAugment or RandAugment')
    weight = t.get('rotation_weight', 0.)
    if weight < 0 or (weight > 0 and not (m.get('rotation_head') and m.get('architecture') == 'attention_resnet')):
        raise ValueError('Positive rotation_weight requires an attention_resnet rotation head')
    if weight > 0 and a['mixup_alpha']:
        raise ValueError('Disable mixup for the auxiliary rotation comparison')
    if m.get('sla_head', False) and m.get('architecture') != 'preact_resnet':
        raise ValueError('SLA head requires preact_resnet')
    if bool(t.get('sla', False)) != bool(m.get('sla_head', False)):
        raise ValueError('Enable training.sla and model.sla_head together')
    if t.get('sla', False) and (a['mixup_alpha'] or weight or m.get('rotation_head', False)):
        raise ValueError('SLA requires unmixed labels and no auxiliary rotation task')
    if c['inference'].get('rotation_vote', False) and not m.get('sla_head', False):
        raise ValueError('Rotation voting requires an SLA head')
    rho = c['optimizer'].get('sam_rho', 0.)
    if not isinstance(rho, (float, int)) or not math.isfinite(rho) or rho < 0:
        raise ValueError('sam_rho must be finite and nonnegative')
    if rho > 0 and (c['optimizer']['name'] != 'sgd' or t['amp'] or t.get('sla', False) or weight):
        raise ValueError('SAM currently requires SGD, training.amp=false, and no auxiliary/SLA objective')


def save_config(config, path):
    Path(path).write_text(yaml.safe_dump(config, sort_keys=False))
