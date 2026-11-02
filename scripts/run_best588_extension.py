"""Wait for current training, then branch from the exact 90.10% submission checkpoint."""
import argparse
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import fcntl
import hashlib
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
import torch
from cifar10.config import validate
from cifar10.engine import device_for, extend_training, train, validate_extension
from cifar10.pipeline import expected_split, find_checkpoint
from scripts.queue_a100 import queue_guard

SOURCE = ('runs_hpc/a100_experiments/'
          'a100_sla600_sweep000_20261101T013008Z_62b447/best.pt')
SOURCE_SHA256 = '926bf2190c2eb61bdaa74d4a4da679486f09e5aa0345cb099db0cc895063033f'


def branch_config(state, project):
    if state['epoch'] + 1 != 588:
        raise ValueError('Expected the original submitted epoch-588 best.pt')
    config = deepcopy(state['config'])
    config['experiment'].update(name=config['experiment']['name'] + '_extend800_from588',
                                output_dir=str(project / 'runs_hpc/sla_best588_extend800'))
    config['training'].update(epochs=800, device='cuda')
    config['data']['root'] = str(project / 'dataset')
    config['optimizer']['lr'] = .001
    config['scheduler'].update(name='cosine', warmup_epochs=5, min_lr=0., restart_epoch=588)
    validate(config)
    validate_extension(state, config, allow_selected_checkpoint=True)
    return config


@contextmanager
def wait_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                print(f'Waiting for current training: {path}', flush=True)
                time.sleep(30)
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).absolute().parents[1])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    project = args.project.absolute()
    source = project / SOURCE
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError('Source checksum differs from the verified 90.10% submission checkpoint')
    state = torch.load(source, map_location='cpu', weights_only=True)
    config = branch_config(state, project)
    del state
    print('Verified original epoch-588 checkpoint hash; 212 additional epochs to 800; '
          '45k/5k split preserved; peak LR 0.001, five warmup epochs.', flush=True)
    if args.dry_run:
        print('Dry run: no locks acquired and no training started.', flush=True)
        return
    output = Path(config['experiment']['output_dir'])
    with ExitStack() as locks:
        locks.enter_context(queue_guard(output / '.launcher.lock'))
        for relative in ['runs_hpc/.sla_followup.lock', 'runs_hpc/.a100_queue.lock',
                         'runs_hpc/a100_800_val1k/.launcher.lock',
                         'runs_hpc/a100_800_val1k/.pipeline.lock',
                         'runs_hpc/a100_finalists/.finalists_launcher.lock',
                         'runs_hpc/a100_finalists/.pipeline.lock',
                         'runs_hpc/a100_experiments/.pipeline.lock',
                         'runs_hpc/attention600/.pipeline.lock',
                         'runs_hpc/robustness/.pipeline.lock',
                         'runs_hpc/sla600_extend800/.pipeline.lock',
                         'runs_hpc/sla_best588_extend800/.pipeline.lock']:
            locks.enter_context(wait_lock(project / relative))
        device_for('cuda')
        checkpoint, completed = find_checkpoint(config, expected_split(config))
        if completed >= 800:
            print('SKIP completed best588 continuation.', flush=True)
        elif checkpoint:
            print(f'RESUME best588 branch at {completed}/800: {checkpoint}', flush=True)
            train(config, resume=checkpoint)
        else:
            print('START branch from exact epoch-588 submission weights.', flush=True)
            extend_training(source, 800, output_dir=str(output), device_name='cuda',
                            data_root=config['data']['root'], allow_selected_checkpoint=True)


if __name__ == '__main__':
    main()
