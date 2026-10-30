import argparse
from copy import deepcopy
import itertools
import json
from pathlib import Path

import yaml

from .config import load_config, set_value, validate
from .engine import train, predict, device_for
from .models import build_model
from .data import check_data


def sweep_configs(path, overrides):
    path = Path(path)
    spec = yaml.safe_load(path.read_text())
    base = load_config(path.parent / spec['base'])
    for key, value in spec.get('overrides', {}).items():
        set_value(base, key, value)
    for item in overrides:
        key, value = item.split('=', 1)
        set_value(base, key, yaml.safe_load(value))
    keys = list(spec['grid'])
    for index, values in enumerate(itertools.product(*(spec['grid'][k] for k in keys))):
        config = deepcopy(base)
        for key, value in zip(keys, values):
            set_value(config, key, value)
        config['experiment']['name'] += f'_sweep{index:03}'
        validate(config)
        yield config


def report(run_root, readme):
    summaries = []
    for path in sorted(Path(run_root).glob('*/summary.json')):
        summary = json.loads(path.read_text())
        summaries.append(summary)
    lines = ['<!-- RESULTS:START -->',
             'Measured results from saved runs. Losses below are from the epoch with the highest validation accuracy.', '',
             '| Run | Device | Parameters | Epochs completed | Selected epoch | Training loss | Validation loss | Validation accuracy |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        m = s.get('best_metrics') or s['final_metrics']
        loss = f"{m['val_loss']:.4f}" if m['val_loss'] is not None else 'N/A'
        accuracy = f"{m['val_accuracy']:.2%}" if m['val_accuracy'] is not None else 'N/A'
        lines.append(f"| {Path(s['run']).name} | {s['device']} | {s['parameters']:,} | {s['epochs_completed']} | {m['epoch']} | {m['train_loss']:.4f} | {loss} | {accuracy} |")
    if not summaries:
        lines.append('| No competition runs completed | — | — | — | — | — | — | — |')
    groups = {}
    for s in summaries:
        if s['best_val_accuracy'] is not None:
            groups.setdefault(s['split_sha256'], []).append(s)
    for fingerprint, runs in groups.items():
        best = max(runs, key=lambda s: (s['best_val_accuracy'], -s['best_metrics']['val_loss']))
        lines += ['', f"Best observed validation accuracy for split `{fingerprint[:12]}`: **{best['best_val_accuracy']:.2%}**, run `{Path(best['run']).name}`.",
                  f"Checkpoint: `{best['best_checkpoint']}`."]
    lines += ['', 'Training loss uses the configured augmentation, mixup, and label smoothing; validation loss is ordinary cross-entropy on unaugmented held-out training images. These objectives can differ.',
              '', 'Hidden evaluation/test loss and accuracy: **unavailable (labels are not provided)**. No evaluation images are used in training or checkpoint selection.',
              '<!-- RESULTS:END -->']
    path = Path(readme)
    text = path.read_text() if path.exists() else '# Experiment results\n'
    start, end = '<!-- RESULTS:START -->', '<!-- RESULTS:END -->'
    section = '\n'.join(lines)
    if start in text and end in text:
        before, rest = text.split(start, 1)
        _, after = rest.split(end, 1)
        text = before + section + after
    else:
        text += '\n\n' + section + '\n'
    path.write_text(text)
    print(f"Updated {path} from {len(summaries)} runs")


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 ResNet experiment lab')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('inspect', 'train', 'sweep', 'check-data'):
        p = commands.add_parser(name)
        p.add_argument('--config', required=True)
        p.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE')
        if name == 'train':
            p.add_argument('--resume')
        if name == 'sweep':
            p.add_argument('--dry-run', action='store_true')
            p.add_argument('--readme', default='README.md')
    p = commands.add_parser('predict')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', default='submissions/submission.csv')
    p.add_argument('--data-root')
    p.add_argument('--device', default='auto')
    p.add_argument('--tta', action=argparse.BooleanOptionalAction, default=None)
    p = commands.add_parser('report')
    p.add_argument('--runs', default='runs')
    p.add_argument('--readme', default='README.md')
    p = commands.add_parser('plan', help='Run/resume a sequential plan of SGD and AdamW experiments')
    p.add_argument('--config', default='configs/all_experiments.yaml')
    p.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE')
    p.add_argument('--readme', default='README.md')
    p.add_argument('--dry-run', action='store_true')
    commands.add_parser('download')
    args = parser.parse_args()
    if args.command == 'plan':
        from .pipeline import run_plan
        run_plan(args.config, args.set, args.readme, args.dry_run)
    elif args.command == 'download':
        try:
            import kagglehub
        except ImportError:
            parser.error('Install download support with: pip install -e ".[download]"')
        print(kagglehub.competition_download('cifar-10-res-net-architecture-challenge-dl'))
        print('Set data.root to the printed path. Competition rules must be accepted on Kaggle.')
    elif args.command == 'predict':
        predict(args.checkpoint, args.output, args.data_root, args.device, args.tta)
    elif args.command == 'report':
        report(args.runs, args.readme)
    elif args.command == 'sweep':
        configs = list(sweep_configs(args.config, args.set))
        valid = []
        for config in configs:
            try:
                _, count = build_model(config['model'])
            except ValueError as exc:
                print(f"SKIP {config['experiment']['name']}: {exc}")
                continue
            print(f"{config['experiment']['name']}: {count:,} parameters; blocks={config['model']['blocks']}; channels={config['model']['channels']}")
            valid.append(config)
        if not args.dry_run:
            for config in valid:
                device_for(config['training']['device'])
                train(config)
                report(config['experiment']['output_dir'], args.readme)
    else:
        config = load_config(args.config, args.set)
        if args.command == 'check-data':
            print(json.dumps(check_data(config), indent=2))
        elif args.command == 'inspect':
            model, count = build_model(config['model'])
            print(model)
            print(f'\nTrainable parameters: {count:,} / {config["model"]["max_parameters"]:,}')
        else:
            train(config, args.resume)


if __name__ == '__main__':
    main()
