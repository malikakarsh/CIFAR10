"""Run/resume one controlled SLA+SD+SAM RandAugment600 trial, then export results."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
import yaml
from cifar10.config import load_config
from cifar10.engine import device_for
from cifar10.models import build_model
from cifar10.pipeline import expected_split, plan_configs, run_plan
from scripts.queue_a100 import queue_guard

PLAN = 'configs/robustness/sla_sam_randaugment_plan.yaml'
RUN_ROOT = 'runs_hpc/sla_sam_randaugment'
BASELINE_RUN = 'runs_hpc/robustness/sla_sam005_600_sweep000_20261101T154821Z_411960'
EXCLUSIVE_LOCKS = [
    'runs_hpc/.sla_sam_randaugment.lock', 'runs_hpc/.sla_followup.lock',
    'runs_hpc/.a100_queue.lock',
    'runs_hpc/a100_800_val1k/.launcher.lock', 'runs_hpc/a100_800_val1k/.pipeline.lock',
    'runs_hpc/a100_finalists/.finalists_launcher.lock', 'runs_hpc/a100_finalists/.pipeline.lock',
    'runs_hpc/a100_experiments/.pipeline.lock', 'runs_hpc/attention600/.pipeline.lock',
    'runs_hpc/robustness/.pipeline.lock', 'runs_hpc/sla600_extend800/.pipeline.lock',
    'runs_hpc/sla_best588_extend800/.launcher.lock',
    'runs_hpc/sla_best588_extend800/.pipeline.lock',
]


def verify_recipe(candidate, baseline):
    expected = deepcopy(baseline)
    expected['experiment'].update(name=candidate['experiment']['name'],
                                  output_dir=candidate['experiment']['output_dir'])
    expected['augmentation'].update(trivialaugment=False, randaugment_ops=2,
                                    randaugment_magnitude=5)
    if candidate != expected:
        raise ValueError('Trial differs from baseline outside augmentation and experiment output settings')


def prepare(project):
    data = str(project / 'dataset')
    output = project / RUN_ROOT
    overrides = ['training.device=cuda', 'data.num_workers=4', f'data.root={data}',
                 f'experiment.output_dir={output}']
    configs = plan_configs(project / PLAN, overrides)
    if len(configs) != 1:
        raise ValueError('Expected exactly one trial')
    baseline = load_config(project / 'configs/robustness/sla_sam.yaml',
                           ['training.device=cuda', 'data.num_workers=4', f'data.root={data}'])
    verify_recipe(configs[0], baseline)
    saved = project / BASELINE_RUN / 'config.yaml'
    if saved.exists():
        original = yaml.safe_load(saved.read_text())
        original['data']['root'] = data
        verify_recipe(configs[0], original)
        print('Recipe checked against the saved 91.20% run config.', flush=True)
    _, count = build_model(configs[0]['model'])
    if count != 4_706_674:
        raise ValueError(f'Unexpected parameter count: {count}')
    manifest = expected_split(configs[0])
    record = json.loads((project / 'docs/sla_sam600_submission.json').read_text())
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if fingerprint != record['split_sha256']:
        raise ValueError('Dataset/split fingerprint differs from the verified 91.20% baseline')
    print(f'Ready: {count:,} parameters; 600 epochs; '
          f'{len(manifest["train_ids"]):,}/{len(manifest["val_ids"]):,} train/validation; '
          'float32; SAM rho=0.05; RandAugment N=2 M=5.', flush=True)
    print('This is a fresh training trial from seed 42; reruns resume matching saved epochs.', flush=True)
    return overrides, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).absolute().parents[1])
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--skip-export', action='store_true', help='Train only; export can be run separately')
    args = parser.parse_args()
    project = args.project.absolute()
    overrides, output = prepare(project)
    if args.dry_run:
        run_plan(project / PLAN, overrides, str(output / 'report.md'), dry_run=True)
        return
    if not args.skip_export and importlib.util.find_spec('matplotlib') is None:
        raise RuntimeError('Install plotting dependencies first: python -m pip install -r requirements-plot.txt')
    device = device_for('cuda')
    import torch
    print(f'GPU: {torch.cuda.get_device_name(device)}; '
          f'PyTorch: {torch.__version__}; output: {output}', flush=True)
    # Fail promptly rather than compete with another known queue or wait away the allocation.
    with ExitStack() as locks:
        for relative in EXCLUSIVE_LOCKS:
            locks.enter_context(queue_guard(project / relative))
        run_plan(project / PLAN, overrides, str(output / 'report.md'))
        if not args.skip_export:
            print('Training complete; generating submission, curves and download bundle.', flush=True)
            subprocess.run([sys.executable, str(project / 'scripts/export_sla_sam_randaugment.py'),
                            '--project', str(project)], cwd=project, check=True)


if __name__ == '__main__':
    main()
