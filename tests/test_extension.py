from copy import deepcopy
import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from cifar10.config import load_config
from cifar10 import engine


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def configs(self):
        original = load_config('configs/hpc_a100/sla600.yaml')
        extended = deepcopy(original)
        extended['training']['epochs'] = 800
        extended['optimizer']['lr'] = .01
        extended['scheduler'].update(restart_epoch=600, min_lr=0.)
        return original, extended

    def test_only_completed_last_checkpoint_and_unchanged_recipe_can_extend(self):
        original, extended = self.configs()
        state = {'config': original, 'epoch': 599}
        engine.validate_extension(state, extended)
        with self.assertRaisesRegex(ValueError, 'completed run'):
            engine.validate_extension({'config': original, 'epoch': 587}, extended)
        changed = deepcopy(extended)
        changed['data']['val_fraction'] = .02
        with self.assertRaisesRegex(ValueError, 'preserve'):
            engine.validate_extension(state, changed)
        changed = deepcopy(extended)
        changed['model']['channels'] = [32, 64, 128]
        with self.assertRaisesRegex(ValueError, 'preserve'):
            engine.validate_extension(state, changed)

    def test_restart_is_a_new_200_epoch_schedule(self):
        _, extended = self.configs()
        self.assertAlmostEqual(engine.epoch_lr(extended, 600), .002)
        self.assertAlmostEqual(engine.epoch_lr(extended, 604), .01)
        self.assertAlmostEqual(engine.epoch_lr(extended, 799), 0.)

    def test_selected_checkpoint_requires_explicit_permission_and_matching_epoch(self):
        original, extended = self.configs()
        extended['scheduler']['restart_epoch'] = 588
        state = {'config': original, 'epoch': 587, 'best_metrics': {'epoch': 588}}
        with self.assertRaisesRegex(ValueError, 'completed run'):
            engine.validate_extension(state, extended)
        engine.validate_extension(state, extended, allow_selected_checkpoint=True)
        state['best_metrics']['epoch'] = 580
        with self.assertRaisesRegex(ValueError, 'best-metrics epoch'):
            engine.validate_extension(state, extended, allow_selected_checkpoint=True)

    def test_extension_preserves_momentum_source_and_exact_branch_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'train_images').mkdir()
            with (root / 'train_labels.csv').open('w') as handle:
                handle.write('id,label\n')
                for index in range(20):
                    Image.new('RGB', (32, 32), (index*10, 40, 90)).save(root / 'train_images' / f'{index}.png')
                    handle.write(f'{index},{index % 10}\n')
            config = load_config('configs/sla/distillation.yaml', [f'data.root={tmp}',
                f'experiment.output_dir={tmp}/original', 'data.filename_column=null',
                'data.num_workers=0', 'training.device=cpu', 'training.epochs=2',
                'scheduler.warmup_epochs=0', 'training.batch_size=10',
                'model.blocks=[1]', 'model.channels=[4]', 'training.amp=false'])
            save_original = engine.atomic_save
            early = root / 'selected_first_epoch.pt'

            def capture_original(value, path):
                save_original(value, path)
                if path.name == 'last.pt' and value['epoch'] == 0:
                    shutil.copyfile(path, early)

            with patch('cifar10.engine.atomic_save', capture_original):
                original = engine.train(config)
            source = Path(original['last_checkpoint'])
            checksum = hashlib.sha256(source.read_bytes()).hexdigest()
            state = torch.load(source, weights_only=True)
            step = torch.optim.SGD.step
            save = engine.atomic_save
            first = root / 'branch_first.pt'
            calls = []

            def check_step(optimizer, *args, **kwargs):
                if not calls:
                    for actual_group, saved_group in zip(optimizer.param_groups, state['optimizer']['param_groups']):
                        for parameter, index in zip(actual_group['params'], saved_group['params']):
                            expected = state['optimizer']['state'][index]['momentum_buffer']
                            self.assertTrue(torch.equal(optimizer.state[parameter]['momentum_buffer'], expected))
                calls.append(1)
                return step(optimizer, *args, **kwargs)

            def capture(value, path):
                save(value, path)
                if path.name == 'last.pt' and value['epoch'] == 2:
                    shutil.copyfile(path, first)

            with patch('torch.optim.SGD.step', check_step), patch('cifar10.engine.atomic_save', capture):
                branch = engine.extend_training(source, 4, .01, 1, root / 'continued')
            self.assertEqual(len(calls), 2)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), checksum)
            completed = torch.load(branch['last_checkpoint'], weights_only=True)
            self.assertTrue(any(not torch.equal(v, completed['model'][k])
                                for k, v in state['model'].items() if v.is_floating_point()))
            repeated = engine.train(completed['config'], resume=first)
            actual = torch.load(repeated['last_checkpoint'], weights_only=True)
            for key, value in completed['model'].items():
                self.assertTrue(torch.equal(value, actual['model'][key]), key)
            selected_hash = hashlib.sha256(early.read_bytes()).hexdigest()
            selected = engine.extend_training(early, 4, .01, 1, root / 'from_selected',
                                              allow_selected_checkpoint=True)
            selected_state = torch.load(selected['last_checkpoint'], weights_only=True)
            self.assertEqual(selected_state['config']['scheduler']['restart_epoch'], 1)
            self.assertEqual(selected_state['epoch'] + 1, 4)
            self.assertIn('_from1', selected_state['config']['experiment']['name'])
            self.assertEqual(hashlib.sha256(early.read_bytes()).hexdigest(), selected_hash)


if __name__ == '__main__':
    unittest.main()
