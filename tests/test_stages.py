from pathlib import Path
import tempfile
import unittest

from cifar10.config import load_config, save_config
from cifar10.models import build_model
from cifar10.pipeline import plan_configs
from cifar10.stages import prepare_stage


class StageTests(unittest.TestCase):
    def test_first_stage_has_matched_control(self):
        configs = plan_configs('configs/stages/01_augmentation.yaml', [])
        self.assertEqual(len(configs), 3)
        self.assertEqual([(c['augmentation']['randaugment_ops'], c['augmentation']['mixup_alpha'])
                          for c in configs], [(0, 0), (2, 0), (2, .2)])
        for c in configs:
            self.assertEqual(c['training']['epochs'], 200)
            self.assertEqual(c['experiment']['seed'], 42)
            self.assertEqual(c['data']['split_seed'], 42)
            self.assertEqual(build_model(c['model'])[1], 2_436_346)

    def test_confirmation_matches_control_and_candidate_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = prepare_stage('configs/stages/randaugment_mixup.yaml', 'confirm', tmp)
            control, candidate = plan_configs(plan, [])
            self.assertEqual(control['experiment']['seed'], candidate['experiment']['seed'])
            self.assertEqual(candidate['experiment']['seed'], 43)
            self.assertEqual(control['data'], candidate['data'])
            self.assertEqual(control['optimizer'], candidate['optimizer'])
            self.assertEqual(control['model'], candidate['model'])
            self.assertEqual(control['augmentation']['mixup_alpha'], 0)
            self.assertEqual(candidate['augmentation']['mixup_alpha'], .2)
        with tempfile.TemporaryDirectory() as tmp:
            plan = prepare_stage('configs/stages/control.yaml', 'confirm', tmp)
            self.assertEqual(len(plan_configs(plan, [])), 1)

    def test_budget_scale_and_final_preserve_selected_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected = load_config('configs/stages/randaugment_mixup.yaml', ['experiment.seed=43'])
            save_config(selected, root / 'source.yaml')
            plan = prepare_stage(root / 'source.yaml', 'budget', root / 'budget')
            budget, = plan_configs(plan, [])
            self.assertEqual(budget['training']['epochs'], 300)
            self.assertEqual(budget['scheduler'], selected['scheduler'])
            self.assertEqual(budget['augmentation'], selected['augmentation'])
            self.assertEqual(budget['experiment']['seed'], 43)
            save_config(budget, root / 'selected_budget.yaml')
            plan = prepare_stage(root / 'selected_budget.yaml', 'scale', root / 'scale')
            scaled, = plan_configs(plan, [])
            self.assertEqual(build_model(scaled['model'])[1], 4_180_858)
            self.assertEqual(scaled['experiment']['seed'], 43)
            self.assertEqual(scaled['training'], budget['training'])
            save_config(scaled, root / 'selected_scale.yaml')
            plan = prepare_stage(root / 'selected_scale.yaml', 'final', root / 'final')
            final, = plan_configs(plan, [])
            self.assertEqual(final['data']['val_fraction'], 0)
            self.assertEqual(final['training']['epochs'], 300)
            self.assertEqual(final['experiment']['seed'], 43)
            self.assertEqual(final['model'], scaled['model'])

    def test_refuses_invalid_source_and_changed_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            prepare_stage('configs/stages/randaugment.yaml', 'confirm', tmp)
            prepare_stage('configs/stages/randaugment.yaml', 'confirm', tmp)
            with self.assertRaisesRegex(ValueError, 'different settings'):
                prepare_stage('configs/stages/randaugment_mixup.yaml', 'confirm', tmp)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'SGD'):
                prepare_stage('configs/adamw.yaml', 'budget', tmp)


if __name__ == '__main__':
    unittest.main()
