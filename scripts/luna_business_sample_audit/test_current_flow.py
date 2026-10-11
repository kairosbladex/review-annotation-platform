"""防止错框、缺框和看不清被误计为通过。"""
import copy
import unittest
from current_flow import match_gold, normalize, validate


class CurrentFlowTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [dict(class_name='人员', bbox=[.1, .2, .3, .4])]
        self.body = dict(success=True, results=[dict(original_class='人员',
            original_bbox=[.1, .2, .3, .4], rawDecision='uncertain', decision='rejected',
            codexReview=dict(transport='official_codex_cli', requestedModel='gpt-5.6-luna', executionId='one'))])

    def test_point_without_extent_is_not_a_box(self):
        with self.assertRaisesRegex(ValueError, 'missing_complete_initial_bbox'):
            normalize([dict(x=1, y=2, class_name='人员')])

    def test_ambiguous_or_neighbor_box_not_matched(self):
        gold = [dict(self.candidates[0], label='true')]
        self.assertEqual(match_gold(self.candidates*2, gold), {})
        nearby = [dict(class_name='人员', bbox=[.11, .2, .3, .4])]
        self.assertEqual(match_gold(nearby, gold), {})

    def test_uncertain_must_be_rejected(self):
        self.assertEqual(validate(self.body, self.candidates)[0]['rawDecision'], 'uncertain')
        self.body['results'][0]['decision'] = 'confirmed'
        with self.assertRaisesRegex(ValueError, 'policy_mismatch'):
            validate(self.body, self.candidates)

    def test_failure_and_wrong_target_are_not_labels(self):
        with self.assertRaisesRegex(ValueError, 'execution_failed'):
            validate(dict(success=False, results=[]), self.candidates)
        self.body['results'][0]['original_bbox'][0] = .11
        with self.assertRaisesRegex(ValueError, 'target_mapping_mismatch'):
            validate(self.body, self.candidates)


if __name__ == '__main__':
    unittest.main()
