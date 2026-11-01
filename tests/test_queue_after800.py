from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.resume_old_after800 import main


class ResumeAfter800Tests(unittest.TestCase):
    def test_priority_pair_checked_before_original_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('sys.argv', ['resume_old_after800.py', '--project', tmp]), \
                 patch('scripts.resume_old_after800.subprocess.run') as run:
                main()
            self.assertEqual(run.call_count, 2)
            first, second = run.call_args_list
            self.assertEqual(first.args[0][-1], 'scripts/run_800_val1k.py')
            self.assertIn('scripts/queue_a100.py', second.args[0])
            self.assertEqual(first.kwargs['cwd'], Path(tmp))
            self.assertTrue(first.kwargs['check'])
            self.assertTrue(second.kwargs['check'])

    def test_priority_failure_never_starts_original_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('sys.argv', ['resume_old_after800.py', '--project', tmp]), \
                 patch('scripts.resume_old_after800.subprocess.run',
                       side_effect=subprocess.CalledProcessError(1, 'priority')) as run:
                with self.assertRaises(subprocess.CalledProcessError):
                    main()
                self.assertEqual(run.call_count, 1)


if __name__ == '__main__':
    unittest.main()
