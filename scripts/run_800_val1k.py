"""Prepare and run SLA800 then SAM800 on a fixed 49k/1k split after prior queues."""
import fcntl
import os
from pathlib import Path
import sys
import yaml

project = Path(__file__).absolute().parents[1]
os.chdir(project)
sys.path.insert(0, str(project))
from cifar10.config import load_config, validate
from cifar10.models import build_model
from cifar10.pipeline import run_plan
from scripts.queue_a100 import queue_guard

output = project / 'runs_hpc/a100_800_val1k'
configs = project / 'configs/generated/a100_800_val1k'
configs.mkdir(parents=True, exist_ok=True)
sweeps = []
for tag, source in [('sla800', 'sla600'), ('sam800', 'sam')]:
    config = load_config(f'configs/hpc_a100/{source}.yaml')
    config['experiment'].update(name=f'{tag}_49k_val1k', output_dir=str(output))
    config['data'].update(root=str(project / 'dataset'), val_fraction=.02, num_workers=4)
    config['training'].update(epochs=800, device='cuda')
    validate(config)
    _, count = build_model(config['model'])
    (configs / f'{tag}.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    sweep = f'{tag}_sweep.yaml'
    (configs / sweep).write_text(yaml.safe_dump({'base': f'{tag}.yaml', 'grid': {}}))
    sweeps.append(sweep)
    print(f'Prepared {tag}: 800 epochs, 49k/1k split, {count:,} parameters', flush=True)
plan = configs / 'plan.yaml'
plan.write_text(yaml.safe_dump({'sweeps': sweeps, 'interleave': False}))
with queue_guard(output / '.launcher.lock'):
    for relative in ['runs_hpc/a100_finalists/.finalists_launcher.lock',
                     'runs_hpc/a100_finalists/.pipeline.lock',
                     'runs_hpc/.a100_queue.lock',
                     'runs_hpc/a100_experiments/.pipeline.lock',
                     'runs_hpc/attention600/.pipeline.lock']:
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as handle:
            print(f'Waiting for prior queue: {relative}', flush=True)
            fcntl.flock(handle, fcntl.LOCK_EX)
            fcntl.flock(handle, fcntl.LOCK_UN)
    print('Starting SLA800 then SAM800; validation enabled.', flush=True)
    run_plan(plan, readme=str(output / 'report.md'))
