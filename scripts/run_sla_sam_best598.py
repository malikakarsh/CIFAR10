"""Continue the exact 91.20% epoch-598 checkpoint to epoch 700; rerun to resume."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
import torch
from cifar10.config import validate
from cifar10.engine import device_for, extend_training, train, validate_extension
from cifar10.models import build_model
from cifar10.pipeline import expected_split, find_checkpoint
from scripts.queue_a100 import queue_guard
from scripts.run_sla_sam_randaugment import BASELINE_RUN, EXCLUSIVE_LOCKS

RUN_ROOT = 'runs_hpc/sla_sam_best598_extend700'
CONTINUATION_LOCKS = [RUN_ROOT + '/.launcher.lock', RUN_ROOT + '/.pipeline.lock']


def branch_config(state, project):
    if state['epoch'] + 1 != 598:
        raise ValueError('Expected the exact epoch-598 SLA+SD+SAM checkpoint')
    config = deepcopy(state['config'])
    config['experiment'].update(name=config['experiment']['name'] + '_extend700_from598',
                                output_dir=str(project / RUN_ROOT))
    config['training'].update(epochs=700, device='cuda')
    config['data']['root'] = str(project / 'dataset')
    config['optimizer']['lr'] = .0001
    config['scheduler'].update(name='cosine', warmup_epochs=5, min_lr=0., restart_epoch=598)
    validate(config)
    validate_extension(state, config, allow_selected_checkpoint=True)
    return config


def baseline_history(project, source_metrics):
    path = project / BASELINE_RUN / 'metrics.csv'
    with path.open(newline='') as handle:
        rows = [row for row in csv.DictReader(handle) if int(float(row['epoch'])) <= 598]
    if [int(float(row['epoch'])) for row in rows] != list(range(1, 599)):
        raise ValueError('Original baseline metrics must include consecutive epochs 1..598')
    if any(abs(float(rows[-1][key]) - value) > 1e-10 for key, value in source_metrics.items()):
        raise ValueError('Baseline epoch-598 metrics differ from source checkpoint')
    return rows, {'metrics': str(path), 'through_epoch': 598,
                  'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def prepare(project):
    source = project / BASELINE_RUN / 'best.pt'
    record = json.loads((project / 'docs/sla_sam600_submission.json').read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest() != record['checkpoint_sha256']:
        raise ValueError('Source checksum differs from the verified 91.20% checkpoint')
    state = torch.load(source, map_location='cpu', weights_only=True)
    config = branch_config(state, project)
    _, count = build_model(config['model'])
    if count != 4_706_674 or state['best_metrics'] != record['best_metrics']:
        raise ValueError('Source model/metrics differ from the verified baseline')
    manifest = expected_split(config)
    if manifest != state['split']:
        raise ValueError('Dataset or split differs from the epoch-598 source')
    baseline_history(project, state['best_metrics'])
    print('Verified 91.20% source SHA-256 and epoch 598; '
          '102 more epochs to 700, original TrivialAugment and SAM settings, '
          '4,706,674 parameters; peak LR 0.0001, five warmup epochs.', flush=True)
    return config, manifest, source


def run_extension(config, manifest, source):
    checkpoint, completed = find_checkpoint(config, manifest)
    output = Path(config['experiment']['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    provenance = {'source_checkpoint': str(source),
                  'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                  'source_epoch': 598, 'total_epochs': 700, 'additional_epochs': 102,
                  'restart_lr': .0001, 'warmup_epochs': 5,
                  'source_public_accuracy_reported': .912}
    (output / 'continuation_source.json').write_text(json.dumps(provenance, indent=2) + '\n')
    if completed >= 700:
        print('SKIP completed epoch-598-to-700 continuation.', flush=True)
    elif checkpoint:
        print(f'RESUME epoch-598 branch at {completed}/700: {checkpoint}', flush=True)
        train(config, resume=checkpoint)
    else:
        print(f'START continuation from exact 91.20% weights: {source}', flush=True)
        extend_training(source, 700, restart_lr=.0001, warmup_epochs=5,
                        output_dir=str(output), device_name='cuda',
                        data_root=config['data']['root'], allow_selected_checkpoint=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).absolute().parents[1])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    project = args.project.absolute()
    config, manifest, source = prepare(project)
    if args.dry_run:
        checkpoint, completed = find_checkpoint(config, manifest)
        print(f'Dry run: branch progress {completed}/700; checkpoint {checkpoint}; no training.', flush=True)
        return
    device_for('cuda')
    with ExitStack() as locks:
        for relative in ['runs_hpc/.sla_sam_15h_queue.lock'] + EXCLUSIVE_LOCKS + CONTINUATION_LOCKS:
            locks.enter_context(queue_guard(project / relative))
        run_extension(config, manifest, source)


if __name__ == '__main__':
    main()
