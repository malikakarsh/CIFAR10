"""Materialize later experiment stages from a selected resolved configuration."""
from copy import deepcopy
from pathlib import Path

import yaml

from .config import load_config, validate
from .models import build_model


def prepare_stage(source, stage, output_dir):
    config = load_config(source)
    if config['data']['val_fraction'] == 0:
        raise ValueError('Select a configuration evaluated with a validation split first')
    if config['optimizer']['name'] != 'sgd':
        raise ValueError('This protocol is defined for the selected SGD recipe')
    if stage == 'confirm' and (config['experiment']['seed'] != 42 or config['training']['epochs'] != 200 or
                              config['model']['blocks'] != [3, 3, 3] or
                              config['model']['channels'] != [48, 96, 192]):
        raise ValueError('Confirmation expects a 200-epoch stage-1 wider-model configuration')
    configs = []

    def add(name, seeds, update):
        c = deepcopy(config)
        c['experiment']['name'] = name
        update(c)
        validate(c)
        build_model(c['model'])
        configs.append((name, c, seeds))

    if stage == 'confirm':
        # Matched control seeds distinguish repeatable gains from seed variation.
        def control(c):
            c['augmentation']['randaugment_ops'] = 0
            c['augmentation']['randaugment_magnitude'] = 9
            c['augmentation']['mixup_alpha'] = 0.0
        add('stage2_control', [43], control)
        if config['augmentation']['randaugment_ops'] or config['augmentation']['mixup_alpha']:
            add('stage2_candidate', [43], lambda c: None)
    elif stage == 'budget':
        if config['training']['epochs'] != 200:
            raise ValueError('Select the 200-epoch recipe before testing a 300-epoch schedule')
        # The source run supplies the 200-epoch reference for the SAME seed.
        add('stage3_budget300', [config['experiment']['seed']],
            lambda c: c['training'].update(epochs=300))
    elif stage == 'scale':
        add('stage4_large', [config['experiment']['seed']],
            lambda c: c['model'].update(blocks=[5, 5, 5], channels=[48, 96, 192]))
    elif stage == 'final':
        add('stage5_full_data', [config['experiment']['seed']],
            lambda c: c['data'].update(val_fraction=0.0))
    else:
        raise ValueError(f'Unknown stage: {stage}')

    out = Path(output_dir)
    files = {}
    sweeps = []
    for name, c, seeds in configs:
        files[f'{name}.yaml'] = yaml.safe_dump(c, sort_keys=False)
        sweep = f'{name}_sweep.yaml'
        files[sweep] = yaml.safe_dump({'base': f'{name}.yaml', 'grid': {'experiment.seed': seeds}}, sort_keys=False)
        sweeps.append(sweep)
    files['plan.yaml'] = yaml.safe_dump({'sweeps': sweeps, 'interleave': stage == 'confirm'}, sort_keys=False)
    # Do not silently overwrite a plan made from a different selected recipe.
    for name, content in files.items():
        path = out / name
        if path.exists() and path.read_text() != content:
            raise ValueError(f'{path} already contains different settings; choose a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (out / name).write_text(content)
    print(f'Prepared {stage}: {out / "plan.yaml"}. No training started.')
    return out / 'plan.yaml'
