"""Materialize reviewed YOLO labels into a new, provenance-checked session split."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil

from boatdet.dataset import session_key, write_dataset_yaml
from viewer.dataset_review import ReviewStore, checked_boxes, confirmed_background, digest

# Existing evaluation holds, including the session implicated by generic ShipCam0 matches.
TEST_SESSIONS = {'20260910_130317', '20260910_131100', '20260910_132246',
                 '20260910_134822', '20260909_123824'}
VAL_SESSIONS = {'20260909_123510', '20260909_123605', '20260910_131214'}


def resolved_session(frame, decision):
    explicit = session_key(decision.get('recording', ''))
    named = frame.get('session', '')
    if re.fullmatch(r'\d{8}_\d{6}', explicit):
        if re.fullmatch(r'\d{8}_\d{6}', named) and explicit != named:
            raise ValueError(f'Conflicting recording identities for {frame["id"]}')
        return explicit
    if re.fullmatch(r'\d{8}_\d{6}', named):
        return named
    matches = {session_key(m['stem']) for m in frame.get('matches', [])}
    matches = {s for s in matches if re.fullmatch(r'\d{8}_\d{6}', s)}
    return next(iter(matches)) if len(matches) == 1 else None


def build(store, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite dataset: {output}')
    with store.connect() as db:
        decisions = {fid: json.loads(payload) for fid, payload in db.execute('SELECT id, payload FROM decisions ORDER BY id')}
    if set(decisions) != set(store.frames):
        raise ValueError('Every source frame must have a saved decision before freezing this training snapshot')
    snapshot = json.dumps(decisions, sort_keys=True, ensure_ascii=False, allow_nan=False)
    records, counts = [], {s: Counter() for s in ('train', 'val', 'test', 'hold', 'excluded', 'deferred')}
    pixel_metadata = json.loads((store.evidence/'remote_frame_metadata.json').read_text())
    pixels = {(r['split'], r['stem']): r['pixel_sha256'] for r in pixel_metadata}
    # Validate the full plan before writing any dataset files.
    for fid, frame in store.frames.items():
        decision = decisions[fid]
        store.verify_source(fid)
        if (decision['source_image_sha256'] != frame['sha256'] or
                decision['source_label_sha256'] != frame['label_sha256']):
            raise ValueError(f'Stale saved decision: {fid}')
        status = decision['status']
        boxes = checked_boxes(decision['boxes'])
        session = resolved_session(frame, decision)
        if status != 'approved':
            if status not in ('excluded', 'deferred'):
                raise ValueError(f'Unfinished decision: {fid}')
            split = status
        else:
            if not boxes and not confirmed_background(decision):
                raise ValueError(f'Background was not explicitly confirmed: {fid}')
            split = 'hold' if session is None else 'test' if session in TEST_SESSIONS else 'val' if session in VAL_SESSIONS else 'train'
            if 'reserved_evaluation_session' in frame['reasons'] and split != 'test':
                raise ValueError(f'Unresolved evaluation hold: {fid}')
        pixel_hash = pixels[(frame['split'], frame['stem'])]
        filename = f'{session or "unresolved"}__{frame["split"]}__{frame["file"]}'
        records.append({'id': fid, 'split': split, 'session': session, 'file': filename,
                        'source_image_sha256': frame['sha256'], 'source_pixel_sha256': pixel_hash,
                        'decision_version': decision['version'], 'boxes': boxes,
                        'hold_reason': 'recording_identity_unresolved' if split == 'hold' else None})
    duplicates = defaultdict(list)
    for record in records:
        if record['split'] not in ('excluded','deferred'):
            duplicates[record['source_pixel_sha256']].append(record)
    for group in duplicates.values():
        if len(group) < 2:
            continue
        group.sort(key=lambda r:(r['session'] is None, '/ShipCam0_' in r['id'], r['id']))
        same_labels = all(r['boxes'] == group[0]['boxes'] for r in group)
        same_split = len({r['split'] for r in group}) == 1
        if same_labels and same_split:
            for record in group[1:]:
                record.update(split='hold',hold_reason='duplicate_of:'+group[0]['id'])
        else:
            for record in group:
                record.update(split='hold',hold_reason='conflicting_duplicate_labels_or_split')
    for record in records:
        boxes=record['boxes']
        counts[record['split']].update(frames=1,positive=int(bool(boxes)),background=int(not boxes),boxes=len(boxes))
    for split in ('train','val','test'):
        if not counts[split]['positive']:
            raise ValueError(f'{split} has no positive frames')
    output.mkdir(parents=True)
    write_dataset_yaml(output, ('train','val','test'))
    for record in records:
        split = record['split']
        if split in ('excluded','deferred'):
            continue
        image = output/'images'/split/record['file']
        label = output/'labels'/split/(Path(record['file']).stem+'.txt')
        image.parent.mkdir(parents=True,exist_ok=True)
        label.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(store.image_path(record['id']), image)
        if digest(image) != record['source_image_sha256']:
            raise ValueError(f'Image changed while copying: {record["id"]}')
        lines = [f'0 {(x1+x2)/2:.10f} {(y1+y2)/2:.10f} {x2-x1:.10f} {y2-y1:.10f}' for x1,y1,x2,y2 in record['boxes']]
        label.write_text('\n'.join(lines)+ ('\n' if lines else ''))
        record['label_sha256'] = digest(label)
    (output/'decisions.snapshot.json').write_text(snapshot+'\n')
    manifest = {'format':'reviewed-boat-training-v1', 'decision_snapshot_sha256':hashlib.sha256((snapshot+'\n').encode()).hexdigest(),
                'counts':{k:dict(v) for k,v in counts.items()}, 'test_sessions':sorted(TEST_SESSIONS),
                'val_sessions':sorted(VAL_SESSIONS), 'frames':records,
                'notes':'Only reviewed labels. Unknown recordings are held outside dataset.yaml. '
                        'Existing evaluation holds retained; historical checkpoint exposure must be assessed separately.'}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=build(ReviewStore(Path(__file__).resolve().parent),args.out)
    print(json.dumps(result['counts'],indent=2))
