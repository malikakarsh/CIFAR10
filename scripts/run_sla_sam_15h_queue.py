"""Sequential queue: RandAugment600, then the 91.20% checkpoint continued to 700."""
import argparse
from contextlib import ExitStack
import importlib.util
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
from cifar10.engine import device_for
from cifar10.pipeline import run_plan
from scripts.queue_a100 import queue_guard
from scripts.run_sla_sam_randaugment import EXCLUSIVE_LOCKS, PLAN, prepare as prepare_randaugment
from scripts.run_sla_sam_best598 import CONTINUATION_LOCKS, prepare as prepare_extension, run_extension


def run_queue(project):
    # Verify BOTH trials up front so a missing continuation source is caught before GPU hours are spent.
    overrides, output = prepare_randaugment(project)
    config, manifest, source = prepare_extension(project)
    device = device_for('cuda')
    import torch
    print(f'GPU: {torch.cuda.get_device_name(device)}. Exactly two experiments, sequentially.', flush=True)
    failed_exports = []

    def export(script):
        try:
            subprocess.run([sys.executable, str(project / 'scripts' / script),
                            '--project', str(project)], cwd=project, check=True)
        except subprocess.CalledProcessError:
            failed_exports.append(script)
            print(f'Export failed: {script}. Saved training is retained; '
                  'continuing the queue. Rerun this export script after training.', flush=True)

    print('[1/2] RandAugment600: train/resume/skip, then export.', flush=True)
    run_plan(project / PLAN, overrides, str(output / 'report.md'))
    export('export_sla_sam_randaugment.py')
    print('[2/2] Original 91.20% epoch-598 checkpoint -> 700: train/resume/skip.', flush=True)
    run_extension(config, manifest, source)
    export('export_sla_sam_best598.py')
    print(f'Queue training complete. Exports needing retry: {failed_exports or "none"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).absolute().parents[1])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    project = args.project.absolute()
    if args.dry_run:
        overrides, output = prepare_randaugment(project)
        prepare_extension(project)
        run_plan(project / PLAN, overrides, str(output / 'report.md'), dry_run=True)
        print('Dry run complete: RandAugment600 first, epoch-598-to-700 continuation second.', flush=True)
        return
    if importlib.util.find_spec('matplotlib') is None:
        raise RuntimeError('Install plotting dependencies: python -m pip install -r requirements-plot.txt')
    with ExitStack() as locks:
        for relative in ['runs_hpc/.sla_sam_15h_queue.lock'] + EXCLUSIVE_LOCKS + CONTINUATION_LOCKS:
            locks.enter_context(queue_guard(project / relative))
        run_queue(project)


if __name__ == '__main__':
    main()
