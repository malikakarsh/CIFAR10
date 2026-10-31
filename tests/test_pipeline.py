import csv
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from cifar10.config import load_config
from cifar10.data import check_data, split_rows, train_rows
from cifar10.models import build_model
from cifar10.engine import train, predict
from cifar10.cli import report


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_model_budget_and_shape(self):
        for path in ('base', 'wide', 'deep'):
            config = load_config(f'configs/{path}.yaml')
            model, count = build_model(config['model'])
            self.assertLessEqual(count, 5_000_000)
            self.assertEqual(model(torch.randn(2, 3, 32, 32)).shape, (2, 10))
        config['model']['channels'] = [256, 512, 1024]
        with self.assertRaisesRegex(ValueError, 'limit'):
            build_model(config['model'])

    def test_overrides_and_unknown_keys(self):
        config = load_config('configs/wide.yaml', ['model.channels=[16,32,64]', 'optimizer.lr=0.02'])
        self.assertEqual(config['model']['channels'], [16, 32, 64])
        self.assertEqual(config['optimizer']['lr'], 0.02)
        with self.assertRaises(ValueError):
            load_config('configs/base.yaml', ['model.channnels=[16,32,64]'])

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for directory in ('train_images', 'test_images'):
                (root / directory).mkdir()
            rng = np.random.default_rng(123)
            rows = []
            for label in range(10):
                for j in range(3):
                    identifier = f'{label}_{j}'
                    rows.append([identifier, label])
                    Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(root / 'train_images' / f'{identifier}.png')
            for name, header, data in [('train_labels.csv', ['id', 'label'], rows),
                                       ('test.csv', ['id'], [['009'], ['001']]),
                                       ('sample_submission.csv', ['id', 'label'], [['001', 0], ['009', 0]])]:
                with (root / name).open('w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    writer.writerows(data)
            for identifier in ('001', '009'):
                Image.new('RGB', (32, 32)).save(root / 'test_images' / f'{identifier}.png')
            config = load_config('configs/base.yaml', [
                f'data.root={tmp}', f'experiment.output_dir={tmp}/runs',
                'data.filename_column=null',
                'model.blocks=[1,1,1]', 'model.channels=[4,8,16]',
                'training.epochs=2', 'scheduler.warmup_epochs=0',
                'training.device=cpu', 'training.batch_size=10', 'augmentation.mixup_alpha=0.2'])
            raw = train_rows(config)
            audit = check_data(config)
            self.assertEqual(audit['train_split'], 20)
            self.assertEqual(audit['validation_split'], 10)
            self.assertEqual(audit['test_images'], 2)
            a, b = split_rows(raw, config)
            self.assertEqual((a, b), split_rows(list(reversed(raw)), config))
            self.assertFalse({r['id'] for r in a} & {r['id'] for r in b})
            progress = io.StringIO()
            with redirect_stdout(progress):
                result = train(config)
            self.assertIn('waiting for first batch', progress.getvalue())
            self.assertIn('first batch loaded', progress.getvalue())
            self.assertIn('train batch 1/2', progress.getvalue())
            self.assertIn('validation batch 1/1', progress.getvalue())
            self.assertTrue(Path(result['best_checkpoint']).is_file())
            self.assertTrue(np.isfinite(result['best_metrics']['val_loss']))
            output = predict(result['best_checkpoint'], root / 'submission.csv', device_name='cpu', tta=True)
            with output.open() as f:
                submission = list(csv.DictReader(f))
            self.assertEqual([r['id'] for r in submission], ['001', '009'])
            self.assertTrue(all(0 <= int(r['label']) <= 9 for r in submission))
            report(root / 'runs', root / 'README.md')
            self.assertIn('Validation loss', (root / 'README.md').read_text())
            # Resume epoch 1 and confirm epoch 2 exactly reproduces the original CPU run.
            first = torch.load(Path(result['run']) / 'best.pt', weights_only=True)
            if first['epoch'] == 0:
                resumed = train(config, Path(result['run']) / 'best.pt')
                expected = torch.load(result['last_checkpoint'], weights_only=True)['model']
                actual = torch.load(resumed['last_checkpoint'], weights_only=True)['model']
                for key in expected:
                    self.assertTrue(torch.equal(expected[key], actual[key]), key)
            # Hidden IDs must match exactly, including leading zeroes.
            (root / 'test.csv').write_text('id\nwrong\n')
            with self.assertRaisesRegex(ValueError, 'IDs'):
                predict(result['best_checkpoint'], root / 'bad.csv', device_name='cpu')


if __name__ == '__main__':
    unittest.main()
