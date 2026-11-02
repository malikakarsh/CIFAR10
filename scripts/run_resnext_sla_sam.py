"""Run/resume one grouped-architecture trial with a training deadline, then export."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
from cifar10.config import load_config
from cifar10.engine import device_for
from cifar10.models import build_model
from cifar10.pipeline import expected_split, plan_configs, run_plan
from scripts.queue_a100 import queue_guard
from scripts.run_sla_sam_randaugment import EXCLUSIVE_LOCKS
from scripts.run_sla_sam_best598 import CONTINUATION_LOCKS

PLAN = 'configs/architecture/resnext_sla_sam_plan.yaml'
RUN_ROOT = 'runs_hpc/resnext_sla_sam'


def verify_recipe(candidate, baseline, epochs=600):
    expected = deepcopy(baseline)
    expected['model'] = deepcopy(candidate['model'])
    expected['training']['epochs'] = epochs
    expected['experiment'].update(name=candidate['experiment']['name'],
                                  output_dir=candidate['experiment']['output_dir'])
    if candidate != expected:
        raise ValueError('Architecture trial must preserve the baseline training recipe and split')


def trial_paths(epochs):
    if epochs not in (360, 600):
        raise ValueError('Select the explicit 360- or 600-epoch schedule')
    return ((PLAN, RUN_ROOT) if epochs == 600 else
            ('configs/architecture/resnext_sla_sam360_plan.yaml', 'runs_hpc/resnext_sla_sam360'))


def prepare(project, epochs=600):
    plan, root = trial_paths(epochs)
    output = project / root
    common = ['training.device=cuda', 'data.num_workers=4', f'data.root={project / "dataset"}']
    overrides = common + [f'experiment.output_dir={output}']
    configs = plan_configs(project / plan, overrides)
    if len(configs) != 1:
        raise ValueError('Expected exactly one architecture experiment')
    baseline = load_config(project / 'configs/robustness/sla_sam.yaml', common)
    verify_recipe(configs[0], baseline, epochs)
    _, count = build_model(configs[0]['model'])
    if count != 4_942_098:
        raise ValueError(f'Unexpected architecture parameter count: {count}')
    split = expected_split(configs[0])
    fingerprint = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
    record = json.loads((project / 'docs/sla_sam600_submission.json').read_text())
    if fingerprint != record['split_sha256']:
        raise ValueError('Dataset/split fingerprint differs from the verified 91.20% baseline')
    print(f'Ready: ResNeXt-style 29-layer, 8 groups, base width 14; {count:,} parameters; '
          f'{epochs} epochs; original 45k/5k split; SLA+SD+SAM rho=0.05; float32; '
          'original TrivialAugment and cutout. Training from scratch, seed 42.', flush=True)
    return overrides, output


def run_with_deadline(command, project, seconds):
    """Stop only this child process group; completed-epoch checkpoints are retained."""
    process = subprocess.Popen(command, cwd=project, start_new_session=True)
    try:
        code = process.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        print('Training time cap reached. Stopping at the last saved completed epoch; '
              'rerun this launcher in another allocation to resume.', flush=True)
        for stop_signal, grace in [(signal.SIGINT, 120), (signal.SIGTERM, 30), (signal.SIGKILL, 30)]:
            try:
                os.killpg(process.pid, stop_signal)
            except ProcessLookupError:
                break
            try:
                process.wait(timeout=grace)
                break
            except subprocess.TimeoutExpired:
                continue
        return False
    except KeyboardInterrupt:
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        process.wait(timeout=120)
        raise
    if code:
        raise subprocess.CalledProcessError(code, command)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).absolute().parents[1])
    parser.add_argument('--max-hours', type=float, default=14.,
                        help='Training wall-time cap, at most 14.25 hours; reserve remaining allocation for export')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--epochs', type=int, choices=(360, 600), default=600)
    args = parser.parse_args()
    if not math.isfinite(args.max_hours) or not 0 < args.max_hours <= 14.25:
        raise ValueError('--max-hours must be positive and at most 14.25')
    project = args.project.absolute()
    plan, _ = trial_paths(args.epochs)
    overrides, output = prepare(project, args.epochs)
    if args.dry_run:
        run_plan(project / plan, overrides, str(output / 'report.md'), dry_run=True)
        print(f'No training started. Training cap: {args.max_hours:g} hours, then export.', flush=True)
        return
    if importlib.util.find_spec('matplotlib') is None:
        raise RuntimeError('Install plotting dependencies: python -m pip install -r requirements-plot.txt')
    device = device_for('cuda')
    import torch
    print(f'GPU: {torch.cuda.get_device_name(device)}; PyTorch: {torch.__version__}; '
          f'training cap: {args.max_hours:g} hours. Measured first-epoch runtime was approximately 140 seconds; later epochs may vary.', flush=True)
    with ExitStack() as locks:
        for relative in ['runs_hpc/.resnext_architecture.lock', 'runs_hpc/.sla_sam_15h_queue.lock'] + EXCLUSIVE_LOCKS + CONTINUATION_LOCKS:
            locks.enter_context(queue_guard(project / relative))
        command = [sys.executable, '-u', '-m', 'cifar10', 'plan', '--config', str(project / plan),
                   '--readme', str(output / 'report.md'), '--set'] + overrides
        complete = run_with_deadline(command, project, args.max_hours * 3600)
        export = [sys.executable, str(project / 'scripts/export_resnext_sla_sam.py'),
                  '--project', str(project), '--epochs', str(args.epochs)]
        if not complete:
            export.append('--partial')
        subprocess.run(export, cwd=project, check=True)


if __name__ == '__main__':
    main()
