import csv
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch
from torch.nn import functional as F

from cifar10.config import load_config
from cifar10.engine import train, predict
from cifar10.models import build_model
from cifar10.sla import four_views, aggregate_logits, training_loss, probabilities, inference_probabilities


class SLATests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_budget_and_legacy_architecture(self):
        for filename, count in [('base', 4_696_394), ('distillation', 4_706_674)]:
            config = load_config(f'configs/sla/{filename}.yaml')
            model, actual = build_model(config['model'])
            self.assertEqual(actual, count)
            self.assertLessEqual(actual, 5_000_000)
            self.assertFalse(any('gate' in key for key in model.state_dict()))
        self.assertEqual(build_model(load_config('configs/attention/base.yaml')['model'])[1], 4_534_066)

    def test_view_order_and_aggregation_ignores_wrong_transform(self):
        images = torch.arange(2 * 3 * 4 * 4).reshape(2, 3, 4, 4)
        views = four_views(images)
        for k in range(4):
            self.assertTrue(torch.equal(views[2*k:2*k+2], images.rot90(k, (-2, -1))))
        joint = torch.full((4, 2, 10, 4), 999.)
        for k in range(4):
            joint[k, :, :, k] = torch.arange(10) + k
        expected = torch.arange(10).float().add(1.5).expand(2, -1)
        self.assertTrue(torch.equal(aggregate_logits(joint.flatten(0, 1)), expected))

    def test_objective_targets_detached_teacher_and_head_gradients(self):
        c = load_config('configs/sla/distillation.yaml', ['model.blocks=[1]', 'model.channels=[8]'])
        model, _ = build_model(c['model'])
        model.eval()
        images, labels = torch.randn(2, 3, 32, 32), torch.tensor([3, 8])
        loss, logits, parts = training_loss(model, images, labels, .1)
        joint = model.joint_classifier(model.features(four_views(images)))
        # Explicit semantic labels: sample 3 and 8 at rotations 0, 1, 2, 3.
        expected_targets = torch.tensor([12, 32, 13, 33, 14, 34, 15, 35])
        self.assertTrue(torch.allclose(parts['joint'], F.cross_entropy(joint, expected_targets)))
        # Distillation must update the ordinary head, never its teacher head.
        grads = torch.autograd.grad(parts['distillation'],
            (model.classifier.weight, model.joint_classifier.weight), retain_graph=True, allow_unused=True)
        self.assertGreater(grads[0].abs().sum().item(), 0)
        self.assertIsNone(grads[1])
        loss.backward()
        self.assertGreater(model.joint_classifier.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.stem.weight.grad.abs().sum().item(), 0)
        p = probabilities(model, images, rotation_vote=True)
        self.assertEqual(p.shape, (2, 10))
        self.assertTrue(torch.allclose(p.sum(1), torch.ones(2)))
        self.assertTrue(torch.allclose(p, inference_probabilities(model, images, rotation_vote=True)))
        self.assertTrue(torch.allclose(inference_probabilities(model, images), model(images).softmax(1)))
        augmented = inference_probabilities(model, images, True, True, True)
        self.assertTrue(torch.allclose(augmented.sum(1), torch.ones(2)))

    def test_checkpoint_and_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'train_images').mkdir()
            (root / 'test_images').mkdir()
            with (root / 'train_labels.csv').open('w') as f:
                f.write('id,filename,label\n')
                for i in range(20):
                    Image.new('RGB', (32, 32), (i * 10, 40, 80)).save(root / 'train_images' / f'{i}.png')
                    f.write(f'{i},{i}.png,{i % 10}\n')
            Image.new('RGB', (32, 32)).save(root / 'test_images/one.png')
            (root / 'test.csv').write_text('id,filename\n001,one.png\n')
            (root / 'sample_submission.csv').write_text('id,label\n001,0\n')
            c = load_config('configs/sla/distillation.yaml', [f'data.root={tmp}',
                f'experiment.output_dir={tmp}/runs', 'training.device=cpu',
                'model.blocks=[1]', 'model.channels=[8]', 'training.epochs=1',
                'scheduler.warmup_epochs=0', 'training.batch_size=10'])
            result = train(c)
            metrics = result['final_metrics']
            for key in ('train_joint_loss', 'train_distillation_loss', 'train_classification_loss', 'val_loss'):
                self.assertTrue(torch.isfinite(torch.tensor(metrics[key])))
            output = predict(result['best_checkpoint'], root / 'submission.csv', device_name='cpu')
            with output.open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]['id'], '001')
            self.assertIn(int(rows[0]['label']), range(10))

    def test_invalid_objective_configuration(self):
        with self.assertRaisesRegex(ValueError, 'together'):
            load_config('configs/sla/base.yaml', ['training.sla=true'])
        with self.assertRaisesRegex(ValueError, 'unmixed'):
            load_config('configs/sla/distillation.yaml', ['augmentation.mixup_alpha=0.2'])
