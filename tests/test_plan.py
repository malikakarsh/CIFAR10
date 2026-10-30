from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from cifar10.config import load_config, save_config
from cifar10.pipeline import find_checkpoint, plan_configs, run_plan


class PlanTests(unittest.TestCase):
    def config(self, root):
        return load_config('configs/base.yaml', [f'experiment.output_dir={root}',
            'model.blocks=[1]', 'model.channels=[4]', 'training.epochs=2',
            'scheduler.warmup_epochs=0', 'training.device=cpu'])

    def checkpoint(self, config, folder, epoch, manifest):
        folder.mkdir(parents=True, exist_ok=True)
        save_config(config, folder / 'config.yaml')
        torch.save({'config': config, 'split': manifest, 'epoch': epoch}, folder / 'last.pt')
        return folder / 'last.pt'

    def test_existing_sgd_configs_match_and_optimizers_alternate(self):
        from cifar10.cli import sweep_configs
        configs = plan_configs('configs/all_experiments.yaml', [])
        self.assertEqual(len(configs), 8)
        self.assertEqual(configs[::2], list(sweep_configs('configs/compare_mps.yaml', [])))
        self.assertEqual([c['optimizer']['name'] for c in configs], ['sgd', 'adamw'] * 4)
        for sgd, adamw in zip(configs[::2], configs[1::2]):
            self.assertEqual(sgd['model'], adamw['model'])
            self.assertEqual(sgd['data'], adamw['data'])

    def test_furthest_checkpoint_and_config_split_matching(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self.config(tmp)
            manifest = {'train_ids': ['1'], 'val_ids': ['2']}
            self.checkpoint(config, root / 'older', 0, manifest)
            latest = self.checkpoint(config, root / 'newer', 1, manifest)
            # Stale summaries do not control resumption.
            (latest.parent / 'summary.json').write_text('{"epochs_completed": 0}')
            self.assertEqual(find_checkpoint(config, manifest), (latest, 2))
            self.assertEqual(find_checkpoint(config, {'train_ids': ['different']}), (None, 0))
            different = deepcopy(config)
            different['optimizer']['lr'] = 0.02
            self.assertEqual(find_checkpoint(different, manifest), (None, 0))

    def test_resume_then_adamw_then_skip_and_interrupt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sgd = self.config(tmp)
            adamw = deepcopy(sgd)
            adamw['experiment']['name'] = 'adamw'
            adamw['optimizer']['name'] = 'adamw'
            manifest = {'train_ids': ['1'], 'val_ids': ['2']}
            previous = self.checkpoint(sgd, root / 'original', 0, manifest)
            calls = []

            def fake_train(config, resume=None):
                calls.append((config['optimizer']['name'], resume))
                self.checkpoint(config, root / config['optimizer']['name'], 1, manifest)

            with patch('cifar10.pipeline.plan_configs', return_value=[sgd, adamw]), \
                 patch('cifar10.pipeline.expected_split', return_value=manifest), \
                 patch('cifar10.cli.report'), \
                 patch('cifar10.pipeline.train', side_effect=fake_train):
                run_plan('unused')
                self.assertEqual(calls, [('sgd', previous), ('adamw', None)])
                run_plan('unused')
                self.assertEqual(len(calls), 2)

            fresh = deepcopy(sgd)
            fresh['experiment']['name'] = 'interrupted'
            with patch('cifar10.pipeline.plan_configs', return_value=[fresh, adamw]), \
                 patch('cifar10.pipeline.expected_split', return_value=manifest), \
                 patch('cifar10.cli.report'), \
                 patch('cifar10.pipeline.train', side_effect=KeyboardInterrupt) as train:
                with self.assertRaises(SystemExit) as stopped:
                    run_plan('unused')
                self.assertEqual(stopped.exception.code, 130)
                self.assertEqual(train.call_count, 1)


if __name__ == '__main__':
    unittest.main()
