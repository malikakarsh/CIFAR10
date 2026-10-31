"""Wait for attention600, resume it if needed, then run the A100 variation queue."""
import argparse
from contextlib import contextmanager
import fcntl
from pathlib import Path
import sys
import time


@contextmanager
def queue_guard(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('An A100 queue launcher is already running for this project') from exc
        yield


def wait_for_attention(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                print('Waiting for the existing attention600 plan; it remains uninterrupted.', flush=True)
                time.sleep(30)
        # run_plan acquires this same lock itself. Release before calling it.
        fcntl.flock(handle, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--attention-root', type=Path)
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    # Keep the lexical paths passed to the original command: symlink-resolving
    # saved data/output paths would defeat exact config matching on resume.
    project = args.project.absolute()
    data = (args.data_root or project / 'dataset').absolute()
    attention = (args.attention_root or project / 'runs_hpc/attention600').absolute()
    following = (args.run_root or project / 'runs_hpc/a100_experiments').absolute()
    sys.path.insert(0, str(project))
    from cifar10.pipeline import run_plan
    common = ['training.device=cuda', 'data.num_workers=4', f'data.root={data}']

    def run():
        if not args.dry_run:
            wait_for_attention(attention / '.pipeline.lock')
        # Preserve the original attention600 config exactly, including AMP.
        run_plan(project / 'configs/attention/600_plan.yaml',
                 common + [f'experiment.output_dir={attention}'],
                 str(attention / 'report.md'), args.dry_run)
        run_plan(project / 'configs/hpc_a100/plan.yaml',
                 common + [f'experiment.output_dir={following}'],
                 str(following / 'report.md'), args.dry_run)

    if args.dry_run:
        run()
    else:
        with queue_guard(project / 'runs_hpc/.a100_queue.lock'):
            run()


if __name__ == '__main__':
    main()
