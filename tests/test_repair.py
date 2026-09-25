import tempfile
import unittest
from pathlib import Path

import numpy as np

from audit_dataset import audit
from boatdet.dataset import save_annotation
from repair_dataset import repair

BOX = (10, 10, 30, 20)  # x, y, w, h inside a 64x48 frame


def frame(value):
    return np.full((48, 64, 3), value, np.uint8)


class RepairTests(unittest.TestCase):
    def build(self, root):
        """Dataset with one of each defect."""
        save_annotation(root, 'train', 'renkliTekneTekne', 0, frame(10), BOX)          # reserved recording
        save_annotation(root, 'val', 'renksizTekneTekne', 0, frame(20), BOX)           # reserved recording
        save_annotation(root, 'train', 'copied', 0, frame(30), BOX)                    # duplicate source
        save_annotation(root, 'val', 'other_name', 0, frame(30), (12, 11, 28, 18))     # identical pixels
        save_annotation(root, 'train', '20260904-112014_ShipCam0', 0, frame(40), BOX)  # session, majority
        save_annotation(root, 'train', '20260904-112014_ShipCam0', 1, frame(41), BOX)
        save_annotation(root, 'val', '20260904-112014_ShipCam1', 0, frame(42), BOX)    # same session, other camera
        save_annotation(root, 'train', 'clean', 0, frame(50), BOX)
        # Boxes written straight to disk: out of frame, and outside it entirely.
        Path(root, 'labels/train/clean_00001.txt').write_text('0 0.970000 0.500000 0.120000 0.200000\n')
        Path(root, 'images/train/clean_00001.jpg').write_bytes(Path(root, 'images/train/clean_00000.jpg').read_bytes())
        Path(root, 'labels/train/clean_00002.txt').write_text('0 1.500000 0.500000 0.100000 0.200000\n')
        Path(root, 'images/train/clean_00002.jpg').write_bytes(Path(root, 'images/train/clean_00000.jpg').read_bytes())

    def test_repair_resolves_every_audit_issue_and_repeats_cleanly(self):
        with tempfile.TemporaryDirectory() as root:
            self.build(root)
            self.assertTrue(audit(root, hash_images=True)['issues'])

            dry = repair(root, apply=False)
            self.assertTrue(dry)
            self.assertEqual(audit(root, hash_images=True)['issues'],
                             audit(root, hash_images=True)['issues'])  # dry run changed nothing
            self.assertTrue(Path(root, 'labels/train/renkliTekneTekne_00000.txt').exists())

            applied = repair(root, apply=True)
            self.assertEqual(dry, applied)
            self.assertEqual(audit(root, hash_images=True)['issues'], [])

            # reserved recordings moved to test, keeping their labels
            self.assertTrue(Path(root, 'labels/test/renkliTekneTekne_00000.txt').exists())
            self.assertTrue(Path(root, 'labels/test/renksizTekneTekne_00000.txt').exists())
            # the val copy of a training frame is quarantined, the training frame stays
            self.assertTrue(Path(root, '_quarantine/val/images/other_name_00000.jpg').exists())
            self.assertTrue(Path(root, 'images/train/copied_00000.jpg').exists())
            # the whole capture session, both cameras, ends up in train
            self.assertTrue(Path(root, 'labels/train/20260904-112014_ShipCam1_00000.txt').exists())
            # a box reaching outside the frame is clipped, not dropped
            clipped = [float(v) for v in Path(root, 'labels/train/clean_00001.txt').read_text().split()[1:]]
            self.assertAlmostEqual(clipped[0] + clipped[2] / 2, 1.0, places=5)
            self.assertGreater(clipped[2], 0)
            # a box fully outside quarantines the frame instead of inventing a negative
            self.assertFalse(Path(root, 'labels/train/clean_00002.txt').exists())
            self.assertTrue(Path(root, '_quarantine/train/labels/clean_00002.txt').exists())
            # dataset.yaml lists the splits that now exist
            self.assertIn('test: images/test', Path(root, 'dataset.yaml').read_text())

            self.assertEqual(repair(root, apply=True), {})  # idempotent


if __name__ == '__main__':
    unittest.main()
