"""Rebuild the README comparison from tracked metric snapshots and score records."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    root = Path(__file__).resolve().parents[1]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')

    def mean(values):
        return [sum(values[max(0, i-9):i+1]) / min(i+1, 10) for i in range(len(values))]

    for column, (tag, title) in enumerate([('sam600', 'SAM'), ('sla600', 'SLA+SD')]):
        record = json.loads((root / f'docs/{tag}_submission.json').read_text())
        with (root / f'docs/figures/{tag}/metrics_snapshot.csv').open() as handle:
            rows = [{key: float(value) for key, value in r.items()} for r in csv.DictReader(handle)]
        epochs = [r['epoch'] for r in rows]
        loss_axis, accuracy_axis = axes[:, column]
        for axis, keys in [(loss_axis, [('train_loss', '#2364AA', 'Training objective'),
                                       ('val_loss', '#D66A20', 'Validation CE')]),
                           (accuracy_axis, [('train_accuracy_mixup_weighted', '#2364AA', 'Training'),
                                            ('val_accuracy', '#20866B', 'Validation')])]:
            for key, color, label in keys:
                values = [row[key] for row in rows]
                axis.plot(epochs, values, color=color, alpha=.22, lw=.7)
                axis.plot(epochs, mean(values), color=color, lw=2, label=label)
            axis.set_xlabel('Epoch')
            axis.grid(alpha=.15)
            axis.legend(frameon=False, fontsize=9)
        loss_axis.set_title(f'{title} | Public Kaggle {record["public_accuracy_reported"]:.2%}',
                            loc='left', weight='bold')
        loss_axis.set_ylabel('Loss')
        loss_axis.set_ylim(bottom=0)
        accuracy_axis.set_ylabel('Accuracy')
        accuracy_axis.yaxis.set_major_formatter(PercentFormatter(1))
        best = record['best_metrics']
        accuracy_axis.scatter(best['epoch'], best['val_accuracy'], color='#20866B', zorder=5)
        accuracy_axis.annotate(f'Best validation {best["val_accuracy"]:.2%}\nepoch {best["epoch"]}',
            (best['epoch'], best['val_accuracy']), xytext=(-145, -65),
            textcoords='offset points', arrowprops={'arrowstyle': '->', 'color': '#20866B'},
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .9})
    figure.suptitle('SAM and SLA+SD: 600 epochs on A100\n'
                   'Thin: per-epoch metrics; bold: trailing 10-epoch means', fontsize=14, weight='bold')
    for extension in ['png', 'pdf']:
        output = root / f'docs/figures/sam_sla600_comparison.{extension}'
        figure.savefig(output, dpi=160)
        print(output)
    plt.close(figure)


if __name__ == '__main__':
    main()
