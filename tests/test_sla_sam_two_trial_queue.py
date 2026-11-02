from copy import deepcopy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import yaml
from cifar10.config import load_config
from scripts.export_sla_sam_randaugment import export_run
from scripts.run_sla_sam_best598 import branch_config, run_extension
from scripts.run_sla_sam_15h_queue import run_queue


class TwoTrialQueueTests(unittest.TestCase):
    def test_branch_preserves_original_augmentation_and_restarts_at_exact_epoch(self):
        source = load_config('configs/robustness/sla_sam.yaml')
        state = {'config': source, 'epoch': 597, 'best_metrics': {'epoch': 598}}
        before = deepcopy(source)
        config = branch_config(state, Path('/project'))
        self.assertEqual(source, before)
        self.assertEqual(config['training']['epochs'], 700)
        self.assertEqual(config['scheduler']['restart_epoch'], 598)
        self.assertEqual(config['optimizer']['lr'], .0001)
        self.assertTrue(config['augmentation']['trivialaugment'])
        self.assertEqual(config['augmentation'], source['augmentation'])
        self.assertEqual(config['optimizer']['sam_rho'], .05)
        self.assertFalse(config['training']['amp'])
        state['epoch'] = 599
        with self.assertRaisesRegex(ValueError, 'epoch-598'):
            branch_config(state, Path('/project'))

    def exercise(self, fail_training=False, fail_export=False):
        events = []
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            def first(*args):
                events.append('randaugment')
                if fail_training:
                    raise RuntimeError('interrupted')
            def export(command, **kwargs):
                events.append(Path(command[1]).name)
                if fail_export:
                    raise subprocess.CalledProcessError(1, command)
            with patch('scripts.run_sla_sam_15h_queue.prepare_randaugment', return_value=([], project)), \
                 patch('scripts.run_sla_sam_15h_queue.prepare_extension', return_value=({}, {}, project)), \
                 patch('scripts.run_sla_sam_15h_queue.device_for'), \
                 patch('torch.cuda.get_device_name', return_value='Mock GPU'), \
                 patch('scripts.run_sla_sam_15h_queue.run_plan', side_effect=first), \
                 patch('scripts.run_sla_sam_15h_queue.run_extension',
                       side_effect=lambda *args: events.append('original598')), \
                 patch('scripts.run_sla_sam_15h_queue.subprocess.run', side_effect=export):
                if fail_training:
                    with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                        run_queue(project)
                else:
                    run_queue(project)
        return events

    def test_order_is_randaugment_export_then_original_checkpoint_extension_export(self):
        self.assertEqual(self.exercise(), ['randaugment', 'export_sla_sam_randaugment.py',
                                          'original598', 'export_sla_sam_best598.py'])

    def test_interrupted_first_training_never_starts_second_trial(self):
        self.assertEqual(self.exercise(fail_training=True), ['randaugment'])

    def test_export_failure_does_not_waste_remaining_training_allocation(self):
        self.assertEqual(self.exercise(fail_export=True), ['randaugment',
                         'export_sla_sam_randaugment.py', 'original598', 'export_sla_sam_best598.py'])

    def test_partial_continuation_resumes_instead_of_restarting_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.pt'
            source.write_bytes(b'source unchanged')
            config = {'experiment': {'output_dir': str(root / 'branch')}}
            with patch('scripts.run_sla_sam_best598.find_checkpoint',
                       return_value=(root / 'branch/last.pt', 650)), \
                 patch('scripts.run_sla_sam_best598.train') as train, \
                 patch('scripts.run_sla_sam_best598.extend_training') as extend:
                run_extension(config, {}, source)
                train.assert_called_once_with(config, resume=root / 'branch/last.pt')
                extend.assert_not_called()
            self.assertEqual(source.read_bytes(), b'source unchanged')

    def test_export_can_select_original_source_when_continuation_never_improves(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            run = project / 'runs/branch'
            run.mkdir(parents=True)
            source = project / 'original_best.pt'
            source.write_bytes(b'original submission weights')
            state = {'config': load_config('configs/robustness/sla_sam.yaml'),
                     'epoch': 597, 'best_metrics': {'epoch': 598},
                     'split': {'train_ids': ['train'], 'val_ids': ['val']}}
            config = branch_config(state, project)
            (run / 'config.yaml').write_text(yaml.safe_dump(config))
            (run / 'split.json').write_text(json.dumps(state['split']))
            summary = {'best_checkpoint': str(source), 'best_metrics': state['best_metrics'],
                       'parameters': 4_706_674}
            (run / 'summary.json').write_text(json.dumps(summary))
            (project / 'dataset').mkdir()
            (project / 'dataset/sample_submission.csv').write_text('id,label\n0,0\n')
            def predict(checkpoint, output, **kwargs):
                self.assertEqual(checkpoint, source)
                output.write_text('id,label\n0,2\n')
            with patch('scripts.export_sla_sam_randaugment.torch.load', return_value=state), \
                 patch('scripts.export_sla_sam_randaugment.predict', side_effect=predict), \
                 patch('scripts.export_sla_sam_randaugment.subprocess.run'):
                archive = export_run(project, run, summary, [{'epoch': 598}], [],
                                     'continuation', 'Test', allowed_source=source)
            self.assertTrue(archive.is_file())
            exported = project / 'exports/continuation_results/branch'
            self.assertEqual((exported / 'best.pt').read_bytes(), source.read_bytes())
            manifest = json.loads((exported / 'export_manifest.json').read_text())
            self.assertEqual(manifest['best_epoch'], 598)
            self.assertEqual(manifest['continuation_source']['checkpoint'], str(source))


if __name__ == '__main__':
    unittest.main()
