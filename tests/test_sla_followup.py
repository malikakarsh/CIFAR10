from pathlib import Path
import unittest
from unittest.mock import patch

from scripts.run_sla_followup import continue_source


class FollowupResumeTests(unittest.TestCase):
    def exercise(self, checkpoint, completed):
        config = {'experiment': {'output_dir': 'branch'}, 'data': {'root': 'dataset'}}
        with patch('scripts.run_sla_followup.torch.load', return_value={}), \
             patch('scripts.run_sla_followup.continuation_config', return_value=config), \
             patch('scripts.run_sla_followup.expected_split', return_value={}), \
             patch('scripts.run_sla_followup.find_checkpoint', return_value=(checkpoint, completed)), \
             patch('scripts.run_sla_followup.train') as train, \
             patch('scripts.run_sla_followup.extend_training') as extend:
            continue_source(Path('source/last.pt'), Path('.'))
        return train, extend

    def test_partial_branch_resumes_without_restarting_source(self):
        train, extend = self.exercise(Path('branch/last.pt'), 650)
        self.assertEqual(train.call_args.kwargs['resume'], Path('branch/last.pt'))
        extend.assert_not_called()

    def test_completed_branch_is_skipped(self):
        train, extend = self.exercise(Path('branch/last.pt'), 800)
        train.assert_not_called()
        extend.assert_not_called()

    def test_first_invocation_extends_source(self):
        train, extend = self.exercise(None, 0)
        train.assert_not_called()
        self.assertEqual(extend.call_args.args, (Path('source/last.pt'), 800))


if __name__ == '__main__':
    unittest.main()
