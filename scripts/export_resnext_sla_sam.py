"""Export the grouped-architecture trial's selected checkpoint and full saved history."""
import argparse
import json
import os
from pathlib import Path
import sys
import shutil

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
import yaml
from scripts.export_sla_sam_randaugment import export_run, merge_history, select_completed

ROOT = 'runs_hpc/resnext_sla_sam'
PREFIX = 'resnext29_8x14_sla_sam600_sweep000_'


def publish_downloads(project, run, archive, tag, summary, target_epochs):
    """Publish predictable portal download paths; signal completion only after all copies finish."""
    source = archive.parent / run.name
    partial = summary['epochs_completed'] < target_epochs
    destination = project / 'exports' / (f'resnext{target_epochs}' + ('_partial' if partial else '') + '_results')
    destination.mkdir(parents=True, exist_ok=True)
    ready = destination / 'DOWNLOAD_READY.json'
    ready.unlink(missing_ok=True)
    files = {
        f'submission_resnext{target_epochs}_tta.csv': source / f'submission_{tag}_tta.csv',
        f'submission_resnext{target_epochs}_tta.metadata.json': source / f'submission_{tag}_tta.metadata.json',
        'learning_curves.png': source / 'curves' / run.name / 'learning_curves.png',
        'learning_curves.pdf': source / 'curves' / run.name / 'learning_curves.pdf',
        'results.tar.gz': archive,
    }
    for name, original in files.items():
        temporary = destination / (name + '.tmp')
        shutil.copy2(original, temporary)
        os.replace(temporary, destination / name)
    ready.write_text(json.dumps({'training_complete': not partial,
        'epochs_completed': summary['epochs_completed'], 'target_epochs': target_epochs,
        'best_epoch': summary['best_metrics']['epoch'], 'best_metrics': summary['best_metrics'],
        'source_run': str(run), 'files': list(files)}, indent=2) + '\n')
    print(f'PORTAL DOWNLOAD FOLDER: {destination}\nReady marker: {ready}', flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).absolute().parents[1])
    parser.add_argument('--partial', action='store_true', help='Export saved progress after a training deadline')
    parser.add_argument('--epochs', type=int, choices=(360, 600), default=600)
    args = parser.parse_args()
    project = args.project.absolute()
    root = ROOT if args.epochs == 600 else 'runs_hpc/resnext_sla_sam360'
    prefix = f'resnext29_8x14_sla_sam{args.epochs}_sweep000_'
    if args.partial:
        matches = []
        for path in (project / root).glob(prefix + '*/summary.json'):
            summary = json.loads(path.read_text())
            matches.append((path, summary))
        if not matches:
            raise ValueError('No completed epoch saved yet; no checkpoint to export')
        path, summary = max(matches, key=lambda item: (item[1]['epochs_completed'], item[0].stat().st_mtime))
    else:
        path, summary = select_completed(project / root, prefix + '*/summary.json', args.epochs)
    run = path.parent
    config = yaml.safe_load((run / 'config.yaml').read_text())
    split = json.loads((run / 'split.json').read_text())
    epochs = summary['epochs_completed']
    history, segments = merge_history(run.parent, config, split, epochs,
                                      pattern=prefix + '*/metrics.csv')
    partial = epochs < args.epochs
    tag = f'resnext29_8x14_sla_sam{args.epochs}' + ('_partial' if partial else '')
    title = 'ResNeXt-style grouped CNN + SLA + SD + SAM'
    if partial:
        title += f' | INCOMPLETE: {epochs}/{args.epochs} epochs'
        print(f'Exporting INCOMPLETE training: {epochs}/{args.epochs}. Rerun the launcher to resume.', flush=True)
    archive = export_run(project, run, summary, history, segments, tag, title)
    publish_downloads(project, run, archive, tag, summary, args.epochs)


if __name__ == '__main__':
    main()
