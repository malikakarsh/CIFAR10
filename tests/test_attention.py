import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from cifar10.config import load_config
from cifar10.data import ImageDataset
from cifar10.engine import optimizer_parameters, rotation_batch, train, predict
from cifar10.models import build_model


class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_parameter_counts_and_old_config_compatibility(self):
        for name, expected in [('base', 4_534_066), ('rotation', 4_534_966)]:
            c = load_config(f'configs/attention/{name}.yaml')
            model, count = build_model(c['model'])
            self.assertEqual(count, expected)
            self.assertLessEqual(count, 5_000_000)
            self.assertEqual(model(torch.randn(2, 3, 32, 32)).shape, (2, 10))
        old = load_config('configs/base.yaml')
        self.assertNotIn('architecture', old['model'])
        self.assertNotIn('cache_images', old['data'])
        self.assertEqual(build_model(old['model'])[1], 1_084_586)

    def test_rotation_targets_and_both_head_gradients(self):
        x = torch.arange(4 * 3 * 4 * 4).reshape(4, 3, 4, 4).float()
        targets = torch.arange(4)
        rotated = rotation_batch(x, targets)
        for i in range(4):
            self.assertTrue(torch.equal(rotated[i], x[i].rot90(i, (-2, -1))))
        c = load_config('configs/attention/rotation.yaml', ['model.blocks=[1,1]', 'model.channels=[8,16]'])
        model, _ = build_model(c['model'])
        images = torch.randn(4, 3, 32, 32)
        loss = model(images).square().mean() + model.predict_rotation(images).square().mean()
        loss.backward()
        self.assertGreater(model.classifier.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.rotation_classifier.weight.grad.abs().sum().item(), 0)
        groups = optimizer_parameters(model, c['optimizer'])
        actual = [id(p) for group in groups for p in group['params']]
        self.assertEqual(len(actual), len(set(actual)))
        self.assertEqual(set(actual), {id(p) for p in model.parameters()})
        self.assertEqual(groups[1]['weight_decay'], 0)
        self.assertTrue(all(p.ndim <= 1 for p in groups[1]['params']))

    def test_invalid_auxiliary_configuration(self):
        with self.assertRaisesRegex(ValueError, 'rotation head'):
            load_config('configs/base.yaml', ['training.rotation_weight=0.25'])
        with self.assertRaisesRegex(ValueError, 'mixup'):
            load_config('configs/attention/rotation.yaml', ['augmentation.mixup_alpha=0.2'])

    def test_cached_images_and_training_checkpoint_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'train_images').mkdir()
            (root / 'test_images').mkdir()
            rng = np.random.default_rng(5)
            rows = []
            for index in range(20):
                name = f'{index}.png'
                Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(root / 'train_images' / name)
                rows.append({'id': str(index), 'filename': name, 'label': str(index % 10)})
            with (root / 'train_labels.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['id', 'filename', 'label'])
                writer.writeheader(); writer.writerows(rows)
            Image.new('RGB', (32, 32)).save(root / 'test_images/one.png')
            (root / 'test.csv').write_text('id,filename\n001,one.png\n')
            (root / 'sample_submission.csv').write_text('id,label\n001,0\n')
            c = load_config('configs/attention/rotation.yaml', [f'data.root={tmp}',
                f'experiment.output_dir={tmp}/runs', 'training.device=cpu',
                'model.blocks=[1,1]', 'model.channels=[8,16]', 'training.epochs=1',
                'scheduler.warmup_epochs=0', 'training.batch_size=10'])
            dataset = ImageDataset(rows, c)
            with patch('PIL.Image.open', side_effect=AssertionError('Cache must avoid disk reads')):
                cached, label = dataset[3]
            c['data']['cache_images'] = False
            direct, label2 = ImageDataset(rows, c)[3]
            self.assertEqual(label, label2)
            self.assertTrue(torch.equal(cached, direct))
            c['data']['cache_images'] = True
            result = train(c)
            state = torch.load(result['last_checkpoint'], weights_only=True)
            self.assertIn('rotation_classifier.weight', state['model'])
            self.assertTrue(np.isfinite(result['final_metrics']['train_rotation_loss']))
            self.assertTrue(np.isfinite(result['final_metrics']['val_loss']))
            output = predict(result['best_checkpoint'], root / 'submission.csv', device_name='cpu')
            with output.open() as f:
                records = list(csv.DictReader(f))
            self.assertEqual(records[0]['id'], '001')
            self.assertIn(int(records[0]['label']), range(10))
