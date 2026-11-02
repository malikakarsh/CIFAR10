"""Replace only this user's/project's old 600-epoch job with the fresh 360-epoch plan."""
import argparse
import os
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))


def target_kind(arguments):
    if not arguments or not Path(arguments[0]).name.startswith('python'):
        return None
    if any(Path(arg).name == 'run_resnext_sla_sam.py' for arg in arguments):
        epochs = arguments[arguments.index('--epochs') + 1] if '--epochs' in arguments else '600'
        return f'monitor{epochs}' if epochs in ('600', '360') else None
    if '--config' in arguments:
        name = Path(arguments[arguments.index('--config') + 1]).name
        if name == 'resnext_sla_sam_plan.yaml':
            return 'plan600'
        if name == 'resnext_sla_sam360_plan.yaml':
            return 'plan360'
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).absolute().parents[1])
    parser.add_argument('--max-hours', type=float, default=14.15)
    args = parser.parse_args()
    if not 0 < args.max_hours <= 14.25:
        raise ValueError('--max-hours must be positive and at most 14.25')
    project = args.project.absolute()
    if not Path('/proc').is_dir():
        raise RuntimeError('Run this restart helper on the Linux HPC host, not locally')
    from scripts.run_resnext_sla_sam import prepare
    prepare(project, epochs=360)
    targets = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            pid = int(entry.name)
            if entry.stat().st_uid != os.getuid() or (entry / 'cwd').resolve() != project.resolve():
                continue
            arguments = (entry / 'cmdline').read_bytes().decode().strip('\0').split('\0')
            kind = target_kind(arguments)
            # DataLoader workers inherit their parent's command line; select only session leaders.
            if kind and kind.startswith('plan') and os.getpgid(pid) != pid:
                continue
            if kind in ('monitor360', 'plan360'):
                print(f'A 360-epoch job already exists (PID {pid}); leaving it running.', flush=True)
                return
            if kind:
                targets.append((pid, kind))
        except (OSError, ProcessLookupError, UnicodeError):
            continue
    # Stop the old monitor first so it cannot launch an export or interfere with the replacement.
    for pid, kind in sorted(targets, key=lambda item: item[1] != 'monitor600'):
        print(f'Stopping this project\'s {kind}, PID {pid}; saved run files are preserved.', flush=True)
        try:
            if kind == 'plan600':
                os.killpg(pid, signal.SIGTERM)
            else:
                os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        live = []
        for pid, _ in targets:
            try:
                if Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0] != 'Z':
                    live.append(pid)
            except OSError:
                pass
        if not live:
            break
        time.sleep(.2)
    else:
        raise RuntimeError(f'Old processes have not exited: {live}; replacement was not launched')
    print('Starting fresh 360-epoch cosine schedule in a separate output folder.', flush=True)
    os.execv(sys.executable, [sys.executable, '-u', str(project / 'scripts/run_resnext_sla_sam.py'),
                             '--project', str(project), '--epochs', '360', '--max-hours', str(args.max_hours)])


if __name__ == '__main__':
    main()
