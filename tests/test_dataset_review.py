"""Safety and persistence checks for manual dataset review; all writes use fixtures."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from viewer.dataset_review import ReviewConflict, ReviewStore, checked_boxes
from review_background import scan


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.evidence = self.root / 'evidence'
        self.evidence.mkdir()
        images, labels, plans, crosswalk = [], [], [], []
        for split, stem, content in [('train', 'same_00000', '0 0.3 0.3 0.2 0.2\n'),
                                     ('val', 'same_00000', ''),
                                     ('train', 'invalid_00000', '0 0.0 0.5 0.2 0.2\n')]:
            image = self.evidence / 'source_dataset' / 'images' / split / (stem+'.jpg')
            label = self.evidence / 'source_dataset' / 'labels' / split / (stem+'.txt')
            image.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes((split+stem).encode())
            label.write_text(content)
            ih = hashlib.sha256(image.read_bytes()).hexdigest()
            lh = hashlib.sha256(label.read_bytes()).hexdigest()
            images.append(dict(split=split, stem=stem, file=image.name, sha256=ih, bytes=image.stat().st_size))
            labels.append(dict(split=split, stem=stem, sha256=lh, lines=content.splitlines()))
            plans.append(dict(remote_split=split, remote_stem=stem, session='unknown', reasons=['recording_identity_unresolved'], status='held_for_review_or_existing'))
            matches = []
            if split == 'train' and stem == 'same_00000':
                local = self.root / 'yolo_dataset_v4_color/labels/train/local_00000.txt'
                local.parent.mkdir(parents=True)
                local.write_text('0 0.5 0.5 0.2 0.2\n')
                matches.append(dict(dataset='yolo_dataset_v4_color', split='train', stem='local_00000', label_sha256=hashlib.sha256(local.read_bytes()).hexdigest()))
            crosswalk.append(dict(split=split, stem=stem, matches=matches))
        for name, data in [('remote_images.json', images), ('remote_labels.json', {'files': labels}),
                           ('training_import_plan.json', {'frames': plans}), ('frame_crosswalk.json', crosswalk)]:
            (self.evidence/name).write_text(json.dumps(data))
        self.store = ReviewStore(self.root, self.evidence)

    def request(self, **overrides):
        return dict(id='train/same_00000', version=0, status='approved', boxes=[[.1,.2,.4,.6]],
                    chosen_version='manual', note='исправлено', recording='recording-A', **overrides)

    def test_shared_names_are_distinct_records_and_both_are_reviewable(self):
        records = {r['id']: r for r in self.store.index()['frames']}
        self.assertEqual(len(records), 3)
        self.assertIn('names', records['train/same_00000']['flags'])
        self.assertIn('names', records['val/same_00000']['flags'])
        self.assertIn('invalid', records['train/invalid_00000']['flags'])
        self.assertIn('conflict', records['train/same_00000']['flags'])
        self.assertEqual(self.store.detail('train/same_00000')['peers'], ['val/same_00000'])

    def test_save_persists_and_preserves_every_source_file(self):
        source = self.evidence/'source_dataset'
        before = {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
        self.store.save(self.request())
        reopened = ReviewStore(self.root, self.evidence)
        self.assertEqual(reopened.detail('train/same_00000')['decision']['note'], 'исправлено')
        self.assertEqual({p:p.read_bytes() for p in before}, before)
        self.assertIsNone(reopened.detail('val/same_00000')['decision'])

    def test_stale_second_tab_cannot_overwrite_and_history_is_retained(self):
        self.store.save(self.request())
        with self.assertRaises(ReviewConflict):
            self.store.save(self.request())
        request = self.request(); request.update(version=1, status='deferred')
        self.store.save(request)
        self.assertEqual([x['version'] for x in self.store.detail(request['id'])['history']], [2,1])

    def test_background_needs_explicit_confirmation(self):
        request = self.request(); request['boxes'] = []
        with self.assertRaisesRegex(ValueError, 'Подтвердите'):
            self.store.save(request)
        request['background_confirmed'] = True
        self.store.save(request)
        with zipfile.ZipFile(io.BytesIO(self.store.export())) as z:
            self.assertEqual(z.read('labels/train/same_00000.txt'), b'')

    def test_rejects_nonfinite_inverted_outside_and_non_numeric_boxes(self):
        for box in [[float('nan'),0,.4,.5], [-.1,0,.4,.5], [0,0,2,.5], [.4,0,.3,.5],
                    [0,0,0,.5], ['0',0,.3,.5], [False,0,.3,.5]]:
            with self.subTest(box=box), self.assertRaises(ValueError):
                checked_boxes([box])

    def test_source_mutation_blocks_detail_save_and_export(self):
        self.store.save(self.request())
        self.store.image_path('train/same_00000').write_bytes(b'changed')
        for action in [lambda:self.store.detail('train/same_00000'), lambda:self.store.save(self.request()), self.store.export]:
            with self.assertRaises(ReviewConflict): action()

    def test_changed_local_version_cannot_be_selected(self):
        d = self.store.detail('train/same_00000')
        request = self.request(); request.update(chosen_version='local-0', boxes=d['versions'][1]['boxes'])
        (self.root/'yolo_dataset_v4_color/labels/train/local_00000.txt').write_text('')
        self.assertTrue(self.store.detail(request['id'])['versions'][1]['unavailable'])
        with self.assertRaises(ReviewConflict): self.store.save(request)

    def test_selected_version_matches_exact_boxes_and_provenance(self):
        d = self.store.detail('train/same_00000')
        request = self.request(); request.update(chosen_version='local-0', boxes=d['versions'][1]['boxes'])
        saved = self.store.save(request)
        self.assertEqual(saved['chosen_version'], 'local-0')
        self.assertEqual(saved['original_split'], 'train')
        self.assertEqual(saved['import_hold_reasons'], ['recording_identity_unresolved'])

    def test_export_only_approved_labels_keeps_original_split_and_history(self):
        self.store.save(self.request())
        request = self.request(); request.update(id='val/same_00000', status='excluded', boxes=[])
        self.store.save(request)
        with zipfile.ZipFile(io.BytesIO(self.store.export())) as z:
            self.assertNotIn('labels/val/same_00000.txt', z.namelist())
            self.assertEqual(z.read('labels/train/same_00000.txt').decode().strip(), '0 0.2500000000 0.4000000000 0.3000000000 0.4000000000')
            manifest=json.loads(z.read('manifest.json'))
            self.assertFalse(manifest['training_ready'])
            self.assertEqual(len(manifest['decisions']), 2)
            self.assertEqual(len(json.loads(z.read('history.json'))), 2)

    def test_arbitrary_path_ids_are_rejected(self):
        for fid in ['../../README.md', 'train/../../README', None, {}, 'unknown']:
            with self.subTest(fid=fid), self.assertRaises(ValueError):
                self.store.detail(fid)

    def save_background(self, fid='train/same_00000'):
        request = self.request()
        request.update(id=fid, boxes=[], background_confirmed=True)
        return self.store.save(request)

    def test_yolo_scan_only_current_explicit_background_and_no_decision_changes(self):
        self.save_background()
        request = self.request()
        request.update(id='val/same_00000', status='excluded', boxes=[])
        self.store.save(request)
        before = self.store.db_path.read_bytes()
        seen = []
        def detect(path):
            seen.append(path)
            return [{'box': [.1,.2,.4,.6], 'confidence': .73}]
        report = scan(self.store, detect, {'weights': 'test.pt'})
        self.assertEqual(seen, [self.store.image_path('train/same_00000')])
        self.assertTrue(report['complete'])
        self.assertEqual(report['matches'], 1)
        self.assertEqual(self.store.db_path.read_bytes(), before)
        frames = {f['id']: f for f in self.store.index()['frames']}
        self.assertTrue(frames['train/same_00000']['yolo_background'])
        self.assertFalse(frames['val/same_00000']['yolo_background'])
        self.assertEqual(frames['train/same_00000']['yolo_confidence'], .73)
        self.assertEqual(self.store.detail('train/same_00000')['yolo_prediction']['detections'][0]['confidence'], .73)

    def test_reaffirming_background_or_correcting_boxes_removes_old_finding(self):
        for boxes in ([], [[.1,.2,.4,.6]]):
            with self.subTest(boxes=boxes):
                current = self.store.detail('train/same_00000')['decision']
                request = self.request()
                request.update(version=current['version'] if current else 0, boxes=[], background_confirmed=True)
                saved = self.store.save(request)
                scan(self.store, lambda _: [{'box':[.1,.2,.4,.6], 'confidence':.6}], {})
                request.update(version=saved['version'], boxes=boxes)
                self.store.save(request)
                self.assertFalse(self.store.index()['frames'][0]['yolo_background'])
                self.assertIsNone(self.store.detail(request['id'])['yolo_prediction'])

    def test_partial_and_no_detection_reports_do_not_claim_full_scan(self):
        self.save_background()
        self.save_background('val/same_00000')
        report = scan(self.store, lambda _: [], {}, limit=1)
        self.assertEqual((report['scanned'], report['total_background'], report['matches']), (1,2,0))
        self.assertFalse(report['complete'])
        self.assertFalse(any(f['yolo_background'] for f in self.store.index()['frames']))

    def test_yolo_report_must_match_source_hash(self):
        self.save_background()
        scan(self.store, lambda _: [{'box':[.1,.2,.4,.6], 'confidence':.6}], {})
        self.store.frames['train/same_00000']['sha256'] = 'changed'
        self.assertFalse(any(f['yolo_background'] for f in self.store.index()['frames']))


if __name__ == '__main__':
    unittest.main()
