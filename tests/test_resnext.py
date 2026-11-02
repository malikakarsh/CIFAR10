import json
from copy import deepcopy
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import patch, Mock

from PIL import Image
import torch
from cifar10 import engine
from cifar10.config import load_config
from cifar10.models import build_model
from cifar10.sla import training_loss, inference_probabilities
from scripts.run_resnext_sla_sam import run_with_deadline, verify_recipe
from scripts.restart_resnext360 import target_kind
from scripts.export_resnext_sla_sam import publish_downloads


class GroupedArchitectureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_parameter_budget_and_recipe_are_preserved(self):
        config = load_config('configs/architecture/resnext_sla_sam.yaml')
        baseline = load_config('configs/robustness/sla_sam.yaml')
        verify_recipe(config, baseline)
        model, count = build_model(config['model'])
        self.assertEqual(count, 4_942_098)
        grouped = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d) and m.groups > 1]
        self.assertEqual(len(grouped), 9)
        self.assertTrue(all(m.groups == 8 for m in grouped))
        with self.assertRaisesRegex(ValueError, 'limit'):
            build_model(load_config('configs/architecture/resnext_sla_sam.yaml',
                                   ['model.base_width=15'])['model'])
        changed = deepcopy(config)
        changed['augmentation']['trivialaugment'] = False
        with self.assertRaisesRegex(ValueError, 'preserve'):
            verify_recipe(changed, baseline)
        self.assertEqual(build_model(baseline['model'])[1], 4_706_674)

    def test_joint_gradients_and_spatial_rotation_inference(self):
        config = load_config('configs/architecture/resnext_sla_sam.yaml',
                             ['model.blocks=[1,1]', 'model.channels=[8,16]',
                              'model.groups=2', 'model.base_width=8'])
        model, _ = build_model(config['model'])
        images = torch.randn(2, 3, 32, 32)
        loss, logits, _ = training_loss(model, images, torch.tensor([1, 4]), .1)
        self.assertEqual(logits.shape, (2, 10))
        loss.backward()
        for head in (model.classifier, model.joint_classifier):
            self.assertGreater(head.weight.grad.abs().sum().item(), 0)
        model.eval()
        with torch.no_grad():
            probabilities = inference_probabilities(model, images, True, True, True)
        self.assertEqual(probabilities.shape, (2, 10))
        self.assertTrue(torch.allclose(probabilities.sum(1), torch.ones(2), atol=1e-6))

    def test_360_schedule_is_separate_and_reaches_zero_lr(self):
        config = load_config('configs/architecture/resnext_sla_sam360.yaml')
        baseline = load_config('configs/robustness/sla_sam.yaml')
        verify_recipe(config, baseline, epochs=360)
        self.assertEqual(config['training']['epochs'], 360)
        self.assertEqual(build_model(config['model'])[1], 4_942_098)
        self.assertEqual(engine.epoch_lr(config, 359), 0.)
        self.assertNotEqual(config['experiment']['output_dir'],
                            load_config('configs/architecture/resnext_sla_sam.yaml')['experiment']['output_dir'])

    def test_portal_exports_publish_all_files_before_ready_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            run = project / 'runs' / 'trial'
            archive = project / 'exports' / 'nested' / 'results.tar.gz'
            source = archive.parent / run.name
            tag = 'resnext29_8x14_sla_sam360'
            originals = [source / f'submission_{tag}_tta.csv',
                         source / f'submission_{tag}_tta.metadata.json',
                         source / 'curves' / run.name / 'learning_curves.png',
                         source / 'curves' / run.name / 'learning_curves.pdf', archive]
            for original in originals:
                original.parent.mkdir(parents=True, exist_ok=True)
                original.write_bytes(b'exported artifact')
            summary = {'epochs_completed': 360, 'best_metrics': {'epoch': 359}}
            destination = publish_downloads(project, run, archive, tag, summary, 360)
            ready = json.loads((destination / 'DOWNLOAD_READY.json').read_text())
            self.assertTrue(ready['training_complete'])
            self.assertEqual(destination.name, 'resnext360_results')
            for name in ready['files']:
                self.assertEqual((destination / name).read_bytes(), b'exported artifact')
            summary['epochs_completed'] = 350
            partial = publish_downloads(project, run, archive, tag, summary, 360)
            self.assertEqual(partial.name, 'resnext360_partial_results')
            self.assertFalse(json.loads((partial / 'DOWNLOAD_READY.json').read_text())['training_complete'])
            self.assertTrue(ready['training_complete'])

    def test_restart_targets_only_exact_python_trial_commands(self):
        self.assertEqual(target_kind(['python', '-u', 'scripts/run_resnext_sla_sam.py']), 'monitor600')
        self.assertEqual(target_kind(['python', '-u', 'scripts/run_resnext_sla_sam.py',
                                     '--epochs', '360']), 'monitor360')
        self.assertEqual(target_kind(['python', '-m', 'cifar10', 'plan', '--config',
                                     '/project/configs/architecture/resnext_sla_sam_plan.yaml']), 'plan600')
        self.assertIsNone(target_kind(['tail', '-f', 'scripts/run_resnext_sla_sam.py']))
        self.assertIsNone(target_kind(['python', 'scripts/run_sla_sam_15h_queue.py']))

    def test_sam_sla_resume_matches_uninterrupted_grouped_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'train_images').mkdir()
            with (root / 'train_labels.csv').open('w') as handle:
                handle.write('id,label\n')
                for index in range(20):
                    Image.new('RGB', (32, 32), (index * 10, 50, 90)).save(root / 'train_images' / f'{index}.png')
                    handle.write(f'{index},{index % 10}\n')
            config = load_config('configs/architecture/resnext_sla_sam.yaml',
                [f'data.root={tmp}', f'experiment.output_dir={tmp}/runs',
                 'data.filename_column=null', 'data.num_workers=0', 'training.device=cpu',
                 'training.epochs=2', 'scheduler.warmup_epochs=0', 'training.batch_size=10',
                 'model.blocks=[1]', 'model.channels=[8]', 'model.groups=2', 'model.base_width=8'])
            saved = root / 'first.pt'
            original_save = engine.atomic_save
            def capture(state, path):
                original_save(state, path)
                if path.name == 'last.pt' and state['epoch'] == 0:
                    shutil.copyfile(path, saved)
            with patch('cifar10.engine.atomic_save', capture):
                completed = engine.train(config)
            repeated = engine.train(config, resume=saved)
            expected = torch.load(completed['last_checkpoint'], weights_only=True)
            actual = torch.load(repeated['last_checkpoint'], weights_only=True)
            for key, value in expected['model'].items():
                self.assertTrue(torch.equal(value, actual['model'][key]), key)

    def test_deadline_signals_only_owned_child_group_and_returns_partial(self):
        process = Mock(pid=4321)
        process.wait.side_effect = [subprocess.TimeoutExpired('training', 1), 130]
        with patch('scripts.run_resnext_sla_sam.subprocess.Popen', return_value=process), \
             patch('scripts.run_resnext_sla_sam.os.killpg') as kill:
            self.assertFalse(run_with_deadline(['train'], Path('/project'), 1))
        kill.assert_called_once_with(4321, signal.SIGINT)

    def test_nonzero_child_exit_is_not_treated_as_completed_training(self):
        process = Mock()
        process.wait.return_value = 1
        with patch('scripts.run_resnext_sla_sam.subprocess.Popen', return_value=process):
            with self.assertRaises(subprocess.CalledProcessError):
                run_with_deadline(['train'], Path('/project'), 1)


if __name__ == '__main__':
    unittest.main()
