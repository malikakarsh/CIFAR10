import unittest
import torch

from cifar10.config import load_config
from cifar10.models import build_model
from cifar10.sla import inference_probabilities
from scripts.audit_sla_inference import common_validation, policies


class AuditTests(unittest.TestCase):
    def test_shared_holdout_excludes_every_checkpoint_training_set(self):
        states = [{'split': {'train_ids': ['a'], 'val_ids': ['b', 'c']}},
                  {'split': {'train_ids': ['d'], 'val_ids': ['b', 'c', 'e']}}]
        self.assertEqual(common_validation(states), {'b', 'c'})
        states[0]['split']['train_ids'].append('b')
        with self.assertRaisesRegex(ValueError, 'overlap'):
            common_validation(states)

    def test_reused_features_match_actual_submission_inference(self):
        torch.set_num_threads(1)
        config = load_config('configs/sla/distillation.yaml',
                             ['model.blocks=[1]', 'model.channels=[4]'])
        model, _ = build_model(config['model'])
        model.eval()
        images = torch.randn(2, 3, 32, 32)
        actual = policies(model, images)
        with torch.no_grad():
            self.assertTrue(torch.allclose(actual['head_only'], model(images).softmax(1)))
            for name, flip, translate, vote in [
                    ('head_mirror', True, False, False),
                    ('head_mirror_translation', True, True, False),
                    ('combined_mirror', True, False, True),
                    ('combined_mirror_translation', True, True, True)]:
                reference = inference_probabilities(model, images, flip, translate, vote)
                self.assertTrue(torch.allclose(actual[name], reference, atol=1e-6), name)
