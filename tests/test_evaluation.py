import tempfile
from pathlib import Path
import unittest

from evaluate_inputs import read_boxes, score, summarize


class EvaluationTests(unittest.TestCase):
    def test_each_truth_box_matches_at_most_once_and_confidence_is_applied(self):
        truth = [[0, 0, .2, .2]]
        predictions = [[0, 0, .2, .2, .9], [0, 0, .2, .2, .8], [.5, .5, .8, .8, .2]]
        result = score(predictions, truth, .45)
        self.assertEqual((result['tp'], result['fp'], result['fn']), (1, 1, 0))
        self.assertEqual(score(predictions, truth, .95)['fn'], 1)

    def test_negative_frames_count_false_alarms_separately(self):
        result = summarize([
            dict(truth=[], predictions=[[0, 0, 1, 1, .9]], predict_ms=1),
            dict(truth=[[0, 0, .2, .2]], predictions=[], predict_ms=2)], .45)
        self.assertEqual((result['tp'], result['fp'], result['fn']), (0, 1, 1))
        self.assertEqual(result['negative_frames_with_fp'], 1)
        self.assertEqual(result['f1'], 0)

    def test_invalid_labels_fail_instead_of_becoming_negatives(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'frame.txt'
            path.write_text('0 0.99 0.5 0.2 0.2\n')
            with self.assertRaises(ValueError):
                read_boxes(path)
            path.write_text('')
            self.assertEqual(read_boxes(path), [])


if __name__ == '__main__':
    unittest.main()
