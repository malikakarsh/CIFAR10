"""Wait for the existing A100 queue, then train the selected full-data recipes."""
import argparse
import fcntl
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).absolute().parents[1])
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    project = args.project.absolute()
    data = (args.data_root or project / 'dataset').absolute()
    output = (args.run_root or project / 'runs_hpc/a100_finalists').absolute()
    sys.path.insert(0, str(project))
    from cifar10.pipeline import run_plan
    from scripts.queue_a100 import queue_guard

    def run():
        if not args.dry_run:
            # The launcher lock covers the whole old queue. Also cover direct
            # plan invocations, which hold only their output pipeline lock.
            locks = [project / 'runs_hpc/.a100_queue.lock',
                     project / 'runs_hpc/a100_experiments/.pipeline.lock',
                     project / 'runs_hpc/attention600/.pipeline.lock']
            for path in locks:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('a') as handle:
                    print(f'Waiting for prior queue lock: {path}', flush=True)
                    fcntl.flock(handle, fcntl.LOCK_EX)
                    fcntl.flock(handle, fcntl.LOCK_UN)
        run_plan(project / 'configs/hpc_finalists/plan.yaml',
                 ['training.device=cuda', 'data.num_workers=4', f'data.root={data}',
                  f'experiment.output_dir={output}'], str(output / 'report.md'), args.dry_run)

    if args.dry_run:
        run()
    else:
        with queue_guard(output / '.finalists_launcher.lock'):
            run()


if __name__ == '__main__':
    main()
