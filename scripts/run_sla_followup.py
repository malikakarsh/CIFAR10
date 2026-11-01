"""Continue the proven SLA600, then train SLA+SD+SAM; rerun to resume."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
import torch
from cifar10.config import validate
from cifar10.engine import device_for, extend_training, train, validate_extension
from cifar10.pipeline import expected_split, find_checkpoint, run_plan
from scripts.queue_a100 import queue_guard

SOURCE = ('runs_hpc/a100_experiments/'
          'a100_sla600_sweep000_20261101T013008Z_62b447/last.pt')


def continuation_config(state, project):
    config = deepcopy(state['config'])
    config['experiment'].update(name=config['experiment']['name'] + '_extend800',
                                output_dir=str(project / 'runs_hpc/sla600_extend800'))
    config['data']['root'] = str(project / 'dataset')
    config['training'].update(epochs=800, device='cuda')
    config['optimizer']['lr'] = .001
    config['scheduler'].update(name='cosine', warmup_epochs=5, min_lr=0.,
                               restart_epoch=state['epoch'] + 1)
    validate(config)
    validate_extension(state, config)
    return config


def continue_source(source, project):
    state = torch.load(source, map_location='cpu', weights_only=True)
    config = continuation_config(state, project)
    checkpoint, completed = find_checkpoint(config, expected_split(config))
    del state
    if completed >= 800:
        print('SKIP completed SLA600 continuation.', flush=True)
    elif checkpoint:
        print(f'RESUME continuation at {completed}/800: {checkpoint}', flush=True)
        train(config, resume=checkpoint)
    else:
        print('Continue original SLA600 to 800 with a small LR restart.', flush=True)
        extend_training(source, 800, output_dir=config['experiment']['output_dir'],
                        device_name='cuda', data_root=config['data']['root'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).absolute().parents[1])
    parser.add_argument('--source-checkpoint', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    project = args.project.absolute()
    source = args.source_checkpoint or project / SOURCE
    overrides = ['training.device=cuda', 'data.num_workers=4',
                 f'data.root={project / "dataset"}',
                 f'experiment.output_dir={project / "runs_hpc/robustness"}']
    plan = project / 'configs/robustness/next_plan.yaml'
    report = str(project / 'runs_hpc/robustness/report.md')
    if args.dry_run:
        print(f'First: original SLA600 -> 800; source exists: {source.is_file()}; {source}', flush=True)
        run_plan(plan, overrides, report, dry_run=True)
        return
    if not source.is_file():
        raise FileNotFoundError(f'Original epoch-600 last.pt required: {source}')
    device_for('cuda')
    # Fail promptly when an old queue/plan holds a lock; never launch competing GPU work.
    with ExitStack() as locks:
        for relative in ['runs_hpc/.sla_followup.lock', 'runs_hpc/.a100_queue.lock',
                         'runs_hpc/a100_800_val1k/.launcher.lock',
                         'runs_hpc/a100_800_val1k/.pipeline.lock',
                         'runs_hpc/a100_finalists/.finalists_launcher.lock',
                         'runs_hpc/a100_finalists/.pipeline.lock',
                         'runs_hpc/a100_experiments/.pipeline.lock',
                         'runs_hpc/attention600/.pipeline.lock',
                         'runs_hpc/sla600_extend800/.pipeline.lock']:
            locks.enter_context(queue_guard(project / relative))
        continue_source(source, project)
        print('Starting SLA+SD+SAM600 on the original 45k/5k split.', flush=True)
        run_plan(plan, overrides, report)


if __name__ == '__main__':
    main()
