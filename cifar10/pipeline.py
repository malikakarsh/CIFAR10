"""Restartable sequential experiment plans using saved epoch checkpoints."""
from contextlib import ExitStack
import fcntl
import itertools
import json
from pathlib import Path

import torch
import yaml

from .data import train_rows, split_rows, split_manifest
from .engine import device_for, train
from .models import build_model


def plan_configs(path, overrides):
    from .cli import sweep_configs
    path = Path(path)
    spec = yaml.safe_load(path.read_text())
    sweeps = [list(sweep_configs(path.parent / item, overrides)) for item in spec['sweeps']]
    if spec.get('interleave', False):
        return [c for group in itertools.zip_longest(*sweeps) for c in group if c is not None]
    return list(itertools.chain.from_iterable(sweeps))


def expected_split(config):
    rows = train_rows(config)
    tr, va = split_rows(rows, config)
    return split_manifest(rows, tr, va, config)


def find_checkpoint(config, manifest):
    """Use the furthest valid checkpoint, never a possibly stale summary.json."""
    matches = []
    for saved in sorted(Path(config['experiment']['output_dir']).glob('*/config.yaml')):
        if yaml.safe_load(saved.read_text()) != config:
            continue
        checkpoint_path = saved.parent / 'last.pt'
        if not checkpoint_path.exists():
            continue
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if state['config'] != config or state['split'] != manifest:
            continue
        matches.append((state['epoch'] + 1, checkpoint_path))
    if not matches:
        return None, 0
    epochs, path = max(matches, key=lambda item: (item[0], str(item[1])))
    return path, epochs


def run_plan(path, overrides=(), readme='README.md', dry_run=False):
    from .cli import report
    configs = plan_configs(path, overrides)
    manifests = {}
    # Prevent two new plan commands from training the same queue concurrently.
    # Legacy train/sweep processes do not hold this lock; stop them first.
    with ExitStack() as stack:
        if not dry_run:
            for root in sorted({c['experiment']['output_dir'] for c in configs}):
                Path(root).mkdir(parents=True, exist_ok=True)
                handle = stack.enter_context((Path(root) / '.pipeline.lock').open('a'))
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise RuntimeError('Another experiment plan is already running in this output directory') from exc
        for index, config in enumerate(configs, 1):
            _, parameters = build_model(config['model'])
            key = json.dumps(config['data'], sort_keys=True)
            if key not in manifests:
                manifests[key] = expected_split(config)
            checkpoint, completed = find_checkpoint(config, manifests[key])
            epochs = config['training']['epochs']
            action = 'SKIP completed' if completed >= epochs else 'RESUME' if checkpoint else 'START fresh'
            print(f"[{index}/{len(configs)}] {action}: {config['experiment']['name']} "
                  f"({config['optimizer']['name']}, {parameters:,} parameters, {completed}/{epochs} epochs)", flush=True)
            if checkpoint:
                print(f'  Checkpoint: {checkpoint}', flush=True)
            if dry_run:
                continue
            if completed < epochs:
                device_for(config['training']['device'])
                try:
                    train(config, resume=checkpoint)
                except KeyboardInterrupt:
                    print('\nStopped. Rerun this plan command to resume from the last saved epoch. '
                          'Work since that checkpoint will be repeated.', flush=True)
                    raise SystemExit(130)
            report(config['experiment']['output_dir'], readme)
