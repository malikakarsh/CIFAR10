"""Export the epoch-598 continuation, including its original prefix and resumed segments."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
import torch
import yaml
from scripts.export_sla_sam_randaugment import export_run, merge_history, select_completed
from scripts.run_sla_sam_best598 import BASELINE_RUN, RUN_ROOT, baseline_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).absolute().parents[1])
    args = parser.parse_args()
    project = args.project.absolute()
    path, summary = select_completed(project / RUN_ROOT, '*_extend700_from598_*/summary.json', 700)
    run = path.parent
    config = yaml.safe_load((run / 'config.yaml').read_text())
    split = json.loads((run / 'split.json').read_text())
    source = project / BASELINE_RUN / 'best.pt'
    state = torch.load(source, map_location='cpu', weights_only=True)
    prefix, prefix_record = baseline_history(project, state['best_metrics'])
    del state
    history, segments = merge_history(run.parent, config, split, 700, start_epoch=599,
                                      pattern='*_extend700_from598_*/metrics.csv')
    export_run(project, run, summary, prefix + history, [prefix_record] + segments,
               'sla_sam_best598_extended700',
               'SLA + SD + SAM | epoch-598 checkpoint continued to 700', allowed_source=source)


if __name__ == '__main__':
    main()
