"""Plot a consistent snapshot of a live run without loading or changing its model."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('plots'))
    parser.add_argument('--title', default='Training progress')
    parser.add_argument('--subtitle', default='Thin lines: per-epoch values; bold lines: trailing 10-epoch means')
    args = parser.parse_args()
    raw = (args.run / 'metrics.csv').read_text()
    reader = csv.DictReader(io.StringIO(raw))
    rows = []
    for record in reader:
        try:
            rows.append({key: float(value) for key, value in record.items()})
        except (ValueError, TypeError):
            # A live writer can have an incomplete final row.
            continue
    if not rows:
        raise SystemExit('No completed metric rows to plot')
    out = args.output_dir / args.run.name
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'metrics_snapshot.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    epochs = [r['epoch'] for r in rows]
    best = max(rows, key=lambda r: r['val_accuracy'])
    lowest = min(rows, key=lambda r: r['val_loss'])
    last = rows[-1]
    stats = {'run': str(args.run), 'epochs_plotted': len(rows), 'latest': last,
             'best_accuracy_epoch': best, 'lowest_validation_loss_epoch': lowest,
             'source_metrics_sha256': hashlib.sha256(raw.encode()).hexdigest()}
    (out / 'statistics.json').write_text(json.dumps(stats, indent=2) + '\n')

    def smooth(values, window=10):
        return [sum(values[max(0, i-window+1):i+1]) / min(i+1, window)
                for i in range(len(values))]

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), layout='constrained')
    blue, orange, green = '#2364AA', '#D66A20', '#20866B'
    for ax in axes.flat:
        ax.set_xlabel('Epoch')
        ax.grid(alpha=.15)
    for ax in axes[0]:
        for key, color, label in [('train_loss', blue, 'Training'), ('val_loss', orange, 'Validation')]:
            values = [r[key] for r in rows]
            ax.plot(epochs, values, color=color, alpha=.3, lw=1)
            ax.plot(epochs, smooth(values), color=color, lw=2, label=f'{label} (10-epoch mean)')
        ax.set_ylabel('Cross-entropy loss')
        ax.legend(frameon=False, fontsize=9)
    axes[0, 0].set_title('Loss across the run', loc='left', weight='bold')
    axes[0, 0].set_ylim(bottom=0)
    axes[0, 1].set_title('Recent losses: raw values and trend', loc='left', weight='bold')
    cutoff = max(1, epochs[-1] - 59)
    recent = [r for r in rows if r['epoch'] >= cutoff]
    axes[0, 1].set_xlim(cutoff, epochs[-1]+1)
    axes[0, 1].set_ylim(0, max(r['val_loss'] for r in recent)*1.12)
    axes[0, 1].text(.03, .96, f"Latest train: {last['train_loss']:.3f}\nLatest validation: {last['val_loss']:.3f}",
                    transform=axes[0, 1].transAxes, va='top', fontsize=10,
                    bbox={'facecolor': 'white', 'alpha': .8, 'edgecolor': 'none'})
    acc = axes[1, 0]
    for key, color, label in [('train_accuracy_mixup_weighted', blue, 'Training'), ('val_accuracy', green, 'Validation')]:
        values = [r[key] for r in rows]
        acc.plot(epochs, values, color=color, alpha=.25, lw=1)
        acc.plot(epochs, smooth(values), color=color, lw=2, label=f'{label} (10-epoch mean)')
    acc.scatter([best['epoch']], [best['val_accuracy']], color=green, zorder=5)
    acc.annotate(f"Best validation: {best['val_accuracy']:.2%}\nepoch {int(best['epoch'])}",
                 (best['epoch'], best['val_accuracy']), xytext=(-100, -65),
                 textcoords='offset points', arrowprops={'arrowstyle':'->', 'color':green},
                 bbox={'facecolor':'white','alpha':.9,'edgecolor':'none'})
    acc.set_title('Accuracy on training and validation', loc='left', weight='bold')
    acc.set_ylabel('Accuracy')
    acc.yaxis.set_major_formatter(PercentFormatter(1))
    acc.legend(frameon=False, loc='lower right', fontsize=9)
    axes[1, 1].plot(epochs, [r['lr'] for r in rows], color='#7151A1', lw=2)
    axes[1, 1].set_title('Learning rate actually used', loc='left', weight='bold')
    axes[1, 1].set_ylabel('Learning rate')
    axes[1, 1].set_ylim(bottom=0)
    fig.suptitle(f'{args.title} | through epoch {int(epochs[-1])}\n{args.subtitle}',
                 fontsize=15, weight='bold')
    fig.savefig(out / 'learning_curves.png', dpi=160)
    fig.savefig(out / 'learning_curves.pdf')
    plt.close(fig)
    print(json.dumps(stats, indent=2))
    print(f'Plots saved in {out}')


if __name__ == '__main__':
    main()
