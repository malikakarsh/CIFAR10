"""Finish/resume the priority 800-epoch pair, then resume the original A100 queue."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).absolute().parents[1])
    args = parser.parse_args()
    project = args.project.absolute()
    sys.path.insert(0, str(project))
    from scripts.queue_a100 import queue_guard

    with queue_guard(project / 'runs_hpc/.resume_old_after800.lock'):
        for relative in ['runs_hpc/a100_800_val1k/.launcher.lock',
                         'runs_hpc/a100_800_val1k/.pipeline.lock']:
            path = project / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a') as handle:
                print(f'Waiting for priority 800-epoch queue: {relative}', flush=True)
                fcntl.flock(handle, fcntl.LOCK_EX)
                fcntl.flock(handle, fcntl.LOCK_UN)
        # On a later allocation, unfinished priority runs must resume first.
        # If already complete this call only checks/skips them, then returns.
        print('Checking/resuming SLA800 and SAM800 before the original queue.', flush=True)
        subprocess.run([sys.executable, '-u', 'scripts/run_800_val1k.py'],
                       cwd=project, check=True)
        print('Priority pair complete. Resuming original A100 experiments.', flush=True)
        subprocess.run([sys.executable, '-u', 'scripts/queue_a100.py',
                        '--project', str(project)], cwd=project, check=True)


if __name__ == '__main__':
    main()
