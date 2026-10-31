from copy import deepcopy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import shutil

import torch
from torch import nn
from PIL import Image

from cifar10.config import load_config
from cifar10.models import build_model
from cifar10.pipeline import plan_configs
from cifar10.sam import perturb_weights, batch_statistics_without_updates
from cifar10 import engine
from scripts.queue_a100 import queue_guard


class SAMTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_two_pass_update_matches_scalar_sam_and_restores_on_error(self):
        # L(w)=w^2/2, w=2, rho=.1: second-pass gradient is 2.1;
        # SGD(lr=.2) must produce w=1.58 at the ORIGINAL weights.
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(2.)
        optimizer = torch.optim.SGD(model.parameters(), lr=.2)
        (model.weight.square().sum() / 2).backward()
        with perturb_weights(model, .1):
            self.assertAlmostEqual(model.weight.item(), 2.1, places=6)
            optimizer.zero_grad(set_to_none=True)
            (model.weight.square().sum() / 2).backward()
        self.assertEqual(model.weight.item(), 2.)
        optimizer.step()
        self.assertAlmostEqual(model.weight.item(), 1.58, places=6)
        original = model.weight.detach().clone()
        with self.assertRaisesRegex(RuntimeError, 'second pass'):
            with perturb_weights(model, .1):
                raise RuntimeError('second pass failed')
        self.assertTrue(torch.equal(model.weight, original))

    def test_second_pass_uses_batch_statistics_without_duplicate_bn_updates(self):
        bn = nn.BatchNorm1d(2)
        bn.train()
        inputs = torch.tensor([[1., 2.], [3., 6.], [5., 10.]])
        first = bn(inputs)
        state = deepcopy(bn.state_dict())
        with batch_statistics_without_updates(bn):
            second = bn(inputs)
        self.assertTrue(torch.allclose(first, second))
        self.assertTrue(bn.track_running_stats)
        for key, value in state.items():
            self.assertTrue(torch.equal(value, bn.state_dict()[key]), key)

    def test_queue_budget_matched_control_and_no_local_sla_mutation(self):
        configs = plan_configs('configs/hpc_a100/plan.yaml', [])
        self.assertEqual(len(configs), 8)
        for config in configs:
            self.assertLessEqual(build_model(config['model'])[1], 5_000_000)
            self.assertEqual(config['training']['epochs'], 600)
        control, sam, sla = configs[:3]
        sam = deepcopy(sam)
        self.assertEqual(sam['optimizer'].pop('sam_rho'), .05)
        sam['experiment']['name'] = control['experiment']['name']
        self.assertEqual(sam, control)
        self.assertEqual(sla['training']['epochs'], 600)
        self.assertTrue(sla['training']['sla'])
        self.assertTrue(sla['training']['amp'])
        self.assertEqual(load_config('configs/sla/distillation.yaml')['training']['epochs'], 400)
        self.assertEqual(configs[3]['data']['val_fraction'], 0.)

    def test_rejects_unsupported_sam_amp_and_sla_combinations(self):
        with self.assertRaisesRegex(ValueError, 'SAM currently'):
            load_config('configs/hpc_a100/sam.yaml', ['training.amp=true'])
        with self.assertRaisesRegex(ValueError, 'SAM currently'):
            load_config('configs/hpc_a100/sla600.yaml', ['training.amp=false', 'optimizer.sam_rho=0.05'])

    def test_engine_sam_checkpoint_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'train_images').mkdir()
            with (root / 'train_labels.csv').open('w') as handle:
                handle.write('id,label\n')
                for index in range(20):
                    Image.new('RGB', (32, 32), (index * 10, 50, 90)).save(root / 'train_images' / f'{index}.png')
                    handle.write(f'{index},{index % 10}\n')
            config = load_config('configs/hpc_a100/sam.yaml', [f'data.root={tmp}',
                f'experiment.output_dir={tmp}/runs', 'data.filename_column=null',
                'data.num_workers=0', 'training.device=cpu', 'training.epochs=2',
                'scheduler.warmup_epochs=0', 'training.batch_size=10',
                'model.blocks=[1]', 'model.channels=[4]'])
            frozen = root / 'first_epoch.pt'
            original_save = engine.atomic_save

            def capture(state, path):
                original_save(state, path)
                if path.name == 'last.pt' and state['epoch'] == 0:
                    shutil.copyfile(path, frozen)

            with patch('cifar10.engine.atomic_save', side_effect=capture):
                complete = engine.train(config)
            resumed = engine.train(config, frozen)
            expected = torch.load(complete['last_checkpoint'], weights_only=True)
            actual = torch.load(resumed['last_checkpoint'], weights_only=True)
            self.assertEqual(actual['epoch'], 1)
            for name, value in expected['model'].items():
                self.assertTrue(torch.equal(value, actual['model'][name]), name)

    def test_only_one_queue_launcher_can_wait_for_attention(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / 'queue.lock'
            with queue_guard(lock):
                with self.assertRaisesRegex(RuntimeError, 'already running'):
                    with queue_guard(lock):
                        pass
            with queue_guard(lock):
                pass


if __name__ == '__main__':
    unittest.main()
