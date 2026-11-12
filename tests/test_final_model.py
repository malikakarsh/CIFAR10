import csv
from copy import deepcopy
from pathlib import Path
import unittest

import torch

from final_model import CONFIG, build_model
from cifar10.config import validate
from cifar10.engine import epoch_lr
from cifar10.models import build_model as build_pipeline_model
from cifar10.sla import training_loss


class FinalModelTests(unittest.TestCase):
    def test_epoch_schedule_matches_final_history(self):
        path = Path(__file__).resolve().parents[1] / 'plots/final_sla_sam600/metrics.csv'
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), CONFIG['training']['epochs'])
        self.assertFalse(CONFIG['scheduler']['per_step'])
        for epoch, row in enumerate(rows):
            self.assertEqual(int(row['epoch']), epoch + 1)
            self.assertAlmostEqual(epoch_lr(CONFIG, epoch), float(row['lr']), places=15)

    def test_epoch_schedule_rejects_per_step_updates(self):
        config = deepcopy(CONFIG)
        config['scheduler']['per_step'] = True
        with self.assertRaisesRegex(ValueError, 'per_step=false'):
            validate(config)

    def test_configuration_and_weights_work_with_training_model_factory(self):
        validate(CONFIG)
        standalone = build_model()
        pipeline, count = build_pipeline_model(CONFIG['model'])
        self.assertEqual(count, 4_956_062)
        pipeline.load_state_dict(standalone.state_dict(), strict=True)
        self.assertEqual(CONFIG['augmentation']['mixup_alpha'], 0.0)
        self.assertEqual(CONFIG['training']['rotation_weight'], 0.0)

    def test_three_heads_share_backbone(self):
        model = build_model()
        images = torch.randn(2, 3, 32, 32)
        outputs = model.forward_heads(images)
        self.assertEqual(outputs["classification"].shape, (2, 10))
        self.assertEqual(outputs["joint"].shape, (2, 40))
        self.assertEqual(outputs["rotation"].shape, (2, 4))
        labels = torch.tensor([2, 7])
        loss, _, parts = training_loss(
            model, images, labels, CONFIG['training']['label_smoothing'])
        torch.testing.assert_close(loss, parts['classification'] + parts['joint']
                                   + parts['distillation'])
        loss.backward()
        for layer in (
            model.stem, model.classifier, model.joint_classifier,
        ):
            self.assertIsNotNone(layer.weight.grad)
            self.assertTrue(torch.isfinite(layer.weight.grad).all())
            self.assertGreater(layer.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(model.rotation_classifier.weight.grad)
        self.assertEqual(
            sum(p.numel() for p in model.parameters() if p.requires_grad),
            4_956_062,
        )

    def test_existing_classification_and_rotation_interfaces(self):
        model = build_model().eval()
        images = torch.randn(2, 3, 32, 32)
        with torch.inference_mode():
            outputs = model.forward_heads(images)
            torch.testing.assert_close(model(images), outputs["classification"])
            torch.testing.assert_close(model.predict_rotation(images), outputs["rotation"])


if __name__ == "__main__":
    unittest.main()
