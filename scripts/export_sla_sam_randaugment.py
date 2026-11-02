"""Export a completed RandAugment trial's best submission, curves and checkpoints."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
import torch
import yaml
from cifar10.engine import predict, validate_extension


def select_completed(root, pattern='sla_sam_randaugment600_sweep000_*/summary.json', epochs=600):
    matches = []
    for path in root.glob(pattern):
        summary = json.loads(path.read_text())
        if summary['epochs_completed'] >= epochs:
            matches.append((path, summary))
    if not matches:
        raise ValueError(f'No completed {epochs}-epoch trial in {root}')
    return max(matches, key=lambda item: item[0].stat().st_mtime)


def merge_history(root, config, split, epochs, start_epoch=1,
                  pattern='sla_sam_randaugment600_sweep000_*/metrics.csv'):
    """A normal resume starts a new folder; assemble its matching epoch segments."""
    rows, segments = {}, []
    for path in sorted(root.glob(pattern)):
        if yaml.safe_load((path.parent / 'config.yaml').read_text()) != config:
            continue
        if json.loads((path.parent / 'split.json').read_text()) != split:
            continue
        with path.open(newline='') as handle:
            for row in csv.DictReader(handle):
                epoch = int(float(row['epoch']))
                if start_epoch <= epoch <= epochs:
                    rows[epoch] = row
        segments.append({'metrics': str(path),
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    if sorted(rows) != list(range(start_epoch, epochs + 1)):
        raise ValueError('Matching run folders do not contain a complete epoch history')
    return [rows[epoch] for epoch in range(start_epoch, epochs + 1)], segments


def export_run(project, run, summary, history, segments, tag, title,
               allowed_source=None):
    checkpoint = Path(summary['best_checkpoint'])
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    config = yaml.safe_load((run / 'config.yaml').read_text())
    if state['config'] != config:
        # A continuation that never improves legitimately retains its source best.
        if allowed_source is None or checkpoint.resolve() != allowed_source.resolve():
            raise ValueError('Checkpoint experiment differs from summary')
        validate_extension(state, config, allow_selected_checkpoint=True)
    if state['split'] != json.loads((run / 'split.json').read_text()):
        raise ValueError('Best checkpoint split differs from selected run')
    if state['epoch'] + 1 != summary['best_metrics']['epoch']:
        raise ValueError('Best checkpoint epoch differs from summary')
    if state['best_metrics'] != summary['best_metrics']:
        raise ValueError('Best checkpoint metrics differ from summary')
    del state
    # Each resumed training segment has a new folder; keep exports separate per run.
    export = project / f'exports/{tag}_results' / run.name
    export.mkdir(parents=True, exist_ok=True)
    output = export / f'submission_{tag}_tta.csv'
    predict(checkpoint, output, data_root=project / 'dataset', device_name='cuda',
            tta=True, translation_tta=True)
    with output.open() as handle:
        rows = list(csv.DictReader(handle))
    with (project / 'dataset/sample_submission.csv').open() as handle:
        sample = list(csv.DictReader(handle))
    if [r['id'] for r in rows] != [r['id'] for r in sample]:
        raise ValueError('Submission ID order differs from sample')
    for name in ('last.pt', 'config.yaml', 'summary.json',
                 'environment.json', 'split.json'):
        if (run / name).exists():
            shutil.copy2(run / name, export / name)
    # The best weights can remain in an earlier segment if a resumed segment never improves.
    shutil.copy2(checkpoint, export / 'best.pt')
    with (export / 'metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    (export / 'history_segments.json').write_text(json.dumps(segments, indent=2) + '\n')
    subprocess.run([sys.executable, str(project / 'scripts/plot_training.py'),
                    '--run', str(export), '--output-dir', str(export / 'curves'),
                    '--title', title,
                    '--subtitle', 'Training: composite SLA objective; validation: cross-entropy'],
                   cwd=project, check=True)
    manifest = {'source_run': str(run), 'best_epoch': summary['best_metrics']['epoch'],
                'best_metrics': summary['best_metrics'], 'parameters': summary['parameters'],
                'selected_checkpoint': str(checkpoint),
                'continuation_source': ({'checkpoint': str(allowed_source),
                    'sha256': hashlib.sha256(allowed_source.read_bytes()).hexdigest()}
                    if allowed_source is not None else None),
                'files_sha256': {str(p.relative_to(export)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(export.rglob('*'))
                                 if p.is_file() and p.name != 'export_manifest.json'}}
    (export / 'export_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    archive = export.parent / f'{run.name}_results.tar.gz'
    with tarfile.open(archive, 'w:gz') as bundle:
        bundle.add(export, arcname=run.name)
    print(f'\nSUBMIT: {output}\nCURVE: {export / "curves" / run.name / "learning_curves.png"}'
          f'\nDOWNLOAD: {archive}', flush=True)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).absolute().parents[1])
    args = parser.parse_args()
    project = args.project.absolute()
    path, summary = select_completed(project / 'runs_hpc/sla_sam_randaugment')
    run = path.parent
    config = yaml.safe_load((run / 'config.yaml').read_text())
    split = json.loads((run / 'split.json').read_text())
    history, segments = merge_history(run.parent, config, split, 600)
    export_run(project, run, summary, history, segments, 'sla_sam_randaugment600',
               'SLA + SD + SAM + RandAugment | 600 epochs')


if __name__ == '__main__':
    main()
