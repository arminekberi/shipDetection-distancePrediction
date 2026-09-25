import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import hard_mining


class GlareImportTests(unittest.TestCase):
    def run_import(self, candidates, wrong, negative):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'c.json').write_text(json.dumps(candidates))
            (root / 'w.txt').write_text(wrong)
            (root / 'n.txt').write_text(negative)
            args = SimpleNamespace(candidates=root / 'c.json', wrong=root / 'w.txt',
                                   confirmed_negatives=root / 'n.txt', out=root / 'ds', split='train')
            hard_mining.import_reviewed(args)

    def test_conflicting_verdicts_for_one_frame_are_refused(self):
        candidates = [dict(video='v.mp4', frame=3, conf=.5, box=[1, 1, 5, 5]),
                      dict(video='v.mp4', frame=3, conf=.4, box=[9, 9, 20, 20])]
        with self.assertRaisesRegex(SystemExit, 'shared verdict'):
            self.run_import(candidates, '0 1', '0')

    def test_two_boxes_of_one_frame_cannot_both_be_positive(self):
        candidates = [dict(video='v.mp4', frame=3, conf=.5, box=[1, 1, 5, 5]),
                      dict(video='v.mp4', frame=3, conf=.4, box=[9, 9, 20, 20])]
        with self.assertRaisesRegex(SystemExit, 'shared verdict'):
            self.run_import(candidates, '', '')


if __name__ == '__main__':
    unittest.main()
