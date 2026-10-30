import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class HPCLauncherTests(unittest.TestCase):
    def test_shell_syntax(self):
        for name in ('hpc_run.sh', 'hpc_train.sbatch'):
            subprocess.run(['bash', '-n', str(ROOT / 'scripts' / name)], check=True)

    def run_launcher(self, preflight_status=0):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = root / 'fake-python'
            fake.write_text('''#!/usr/bin/env python3
import json, os, sys
with open(os.environ['HPC_TEST_LOG'], 'a') as f:
    f.write(json.dumps(sys.argv[1:]) + '\\n')
if sys.argv[1:] == ['-']:
    sys.stdin.read()
    sys.exit(int(os.environ['HPC_TEST_STATUS']))
''')
            fake.chmod(0o755)
            log = root / 'calls.jsonl'
            env = dict(os.environ, CIFAR_PROJECT_DIR=str(ROOT), CIFAR_PYTHON=str(fake),
                       CIFAR_RUN_ROOT=str(root / 'runs with spaces'),
                       CIFAR_DATA_ROOT=str(root / 'data with spaces'),
                       HPC_TEST_LOG=str(log), HPC_TEST_STATUS=str(preflight_status))
            result = subprocess.run(['bash', str(ROOT / 'scripts/hpc_run.sh'),
                'configs/generalization_plan.yaml', 'training.epochs=2',
                'scheduler.warmup_epochs=0'], env=env, capture_output=True, text=True)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            return result, calls, str(root)

    def test_cuda_plan_arguments_and_custom_paths(self):
        result, calls, root = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls[0], ['-'])
        self.assertEqual(calls[1][:5], ['-m', 'cifar10', 'plan', '--config', 'configs/generalization_plan.yaml'])
        self.assertIn('training.device=cuda', calls[1])
        self.assertIn('data.num_workers=4', calls[1])
        self.assertIn(f'data.root={root}/data with spaces', calls[1])
        self.assertIn(f'experiment.output_dir={root}/runs with spaces', calls[1])
        self.assertEqual(calls[1][-2:], ['training.epochs=2', 'scheduler.warmup_epochs=0'])

    def test_failed_cuda_preflight_prevents_training(self):
        result, calls, _ = self.run_launcher(preflight_status=1)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(calls, [['-']])


if __name__ == '__main__':
    unittest.main()
