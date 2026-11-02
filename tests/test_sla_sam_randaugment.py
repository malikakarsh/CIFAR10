from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml
from cifar10.config import load_config
from cifar10.models import build_model
from cifar10.pipeline import plan_configs
from scripts.export_sla_sam_randaugment import merge_history, select_completed
from scripts.queue_a100 import queue_guard
from scripts.run_sla_sam_randaugment import main, verify_recipe


class RandAugmentTrialTests(unittest.TestCase):
    def test_single_trial_preserves_winning_recipe_except_augmentation(self):
        configs = plan_configs('configs/robustness/sla_sam_randaugment_plan.yaml', [])
        self.assertEqual(len(configs), 1)
        baseline = load_config('configs/robustness/sla_sam.yaml')
        verify_recipe(configs[0], baseline)
        self.assertEqual(build_model(configs[0]['model'])[1], 4_706_674)
        for section, key, value in [('training', 'amp', True),
                                    ('training', 'epochs', 800),
                                    ('data', 'val_fraction', .02),
                                    ('experiment', 'seed', 43),
                                    ('optimizer', 'sam_rho', 0.)]:
            altered = deepcopy(configs[0])
            altered[section][key] = value
            with self.assertRaises(ValueError):
                verify_recipe(altered, baseline)

    def test_held_original_queue_lock_prevents_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            with queue_guard(project / 'runs_hpc/.a100_queue.lock'), \
                 patch('sys.argv', ['launcher', '--project', tmp, '--skip-export']), \
                 patch('scripts.run_sla_sam_randaugment.prepare', return_value=([], project / 'runs')), \
                 patch('scripts.run_sla_sam_randaugment.device_for'), \
                 patch('torch.cuda.get_device_name', return_value='Mock GPU'), \
                 patch('scripts.run_sla_sam_randaugment.run_plan') as plan:
                with self.assertRaisesRegex(RuntimeError, 'already running'):
                    main()
                plan.assert_not_called()

    def test_training_failure_prevents_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            with patch('sys.argv', ['launcher', '--project', tmp]), \
                 patch('scripts.run_sla_sam_randaugment.prepare', return_value=([], project / 'runs')), \
                 patch('scripts.run_sla_sam_randaugment.importlib.util.find_spec', return_value=object()), \
                 patch('scripts.run_sla_sam_randaugment.device_for'), \
                 patch('torch.cuda.get_device_name', return_value='Mock GPU'), \
                 patch('scripts.run_sla_sam_randaugment.run_plan', side_effect=RuntimeError('interrupted')), \
                 patch('scripts.run_sla_sam_randaugment.subprocess.run') as export:
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    main()
                export.assert_not_called()

    def test_export_stitches_matching_resume_segments_and_rejects_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, split = {'recipe': 'same'}, {'train_ids': ['a'], 'val_ids': ['b']}
            for name, epochs, recipe in [('001', [1, 2], config), ('002', [3, 4], config),
                                         ('003', [1, 2, 3, 4], {'recipe': 'other'})]:
                folder = root / f'sla_sam_randaugment600_sweep000_{name}'
                folder.mkdir()
                (folder / 'config.yaml').write_text(yaml.safe_dump(recipe))
                (folder / 'split.json').write_text(json.dumps(split))
                with (folder / 'metrics.csv').open('w', newline='') as handle:
                    writer = csv.writer(handle)
                    writer.writerow(['epoch', 'val_accuracy'])
                    writer.writerows((epoch, epoch / 10) for epoch in epochs)
            rows, sources = merge_history(root, config, split, 4)
            self.assertEqual([int(row['epoch']) for row in rows], [1, 2, 3, 4])
            self.assertEqual(len(sources), 2)
            with self.assertRaisesRegex(ValueError, 'complete epoch history'):
                merge_history(root, config, split, 5)

    def test_export_selects_completed_run_not_newer_partial_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            complete = None
            for name, epochs in [('001', 600), ('002', 200)]:
                folder = root / f'sla_sam_randaugment600_sweep000_{name}'
                folder.mkdir()
                path = folder / 'summary.json'
                path.write_text(json.dumps({'epochs_completed': epochs}))
                if epochs == 600:
                    complete = path
            self.assertEqual(select_completed(root)[0], complete)


if __name__ == '__main__':
    unittest.main()
