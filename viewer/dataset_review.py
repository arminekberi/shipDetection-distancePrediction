"""Local, non-destructive review of the verified rig annotation snapshot."""
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import sqlite3
import threading
import zipfile

EVIDENCE = 'results/manual_training_review_20260914'


def confirmed_background(decision):
    return (decision.get('status') == 'approved' and
            decision.get('background_confirmed') is True and decision.get('boxes') == [])


class ReviewConflict(ValueError):
    """Source changed or a newer decision must be reloaded."""


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_labels(text):
    boxes, errors = [], []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            cls, x, y, w, h = map(float, line.split())
            if cls != 0 or not all(math.isfinite(v) for v in (x, y, w, h)):
                raise ValueError()
            box = [x-w/2, y-h/2, x+w/2, y+h/2]
            boxes.append(box)
            if w <= 0 or h <= 0 or min(box) < -1e-5 or max(box) > 1+1e-5:
                errors.append(number)
        except ValueError:
            errors.append(number)
    return {'boxes': boxes, 'invalid_lines': errors, 'text': text}


def checked_boxes(boxes):
    if not isinstance(boxes, list) or len(boxes) > 100:
        raise ValueError('Допустимо не более 100 рамок.')
    result = []
    for box in boxes:
        if not isinstance(box, list) or len(box) != 4 or any(type(v) not in (int, float) for v in box):
            raise ValueError('Рамка должна содержать четыре числа.')
        if not all(math.isfinite(v) for v in box):
            raise ValueError('Координаты рамки должны быть конечными.')
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError('Рамка должна находиться внутри кадра и иметь ненулевой размер.')
        result.append(list(box))
    return result


class ReviewStore:
    def __init__(self, root, evidence=None):
        self.root = Path(root).resolve()
        self.evidence = Path(evidence or self.root / EVIDENCE).resolve()
        self.source = self.evidence / 'source_dataset'
        self.lock = threading.RLock()
        self.frames = {}
        read = lambda name: json.loads((self.evidence / name).read_text())
        labels = {(r['split'], r['stem']): r for r in read('remote_labels.json')['files']}
        plans = {(r['remote_split'], r['remote_stem']): r for r in read('training_import_plan.json')['frames']}
        crosswalk = {(r['split'], r['stem']): r['matches'] for r in read('frame_crosswalk.json')}
        names = defaultdict(list)
        for r in read('remote_images.json'):
            key = (r['split'], r['stem'])
            fid = '/'.join(key)
            plan = plans.get(key, {})
            label = labels[key]
            matches = crosswalk.get(key, [])
            flags = []
            if parse_labels('\n'.join(label['lines']))['invalid_lines']:
                flags.append('invalid')
            if any(m['label_sha256'] != label['sha256'] for m in matches):
                flags.append('conflict')
            self.frames[fid] = dict(r, id=fid, label_sha256=label['sha256'], matches=matches,
                                    session=plan.get('session', ''), reasons=plan.get('reasons', []),
                                    flags=flags, candidate=plan.get('status') == 'candidate_for_training_import')
            names[r['stem']].append(fid)
        for ids in names.values():
            if len(ids) > 1:
                for fid in ids:
                    self.frames[fid]['flags'].append('names')
                    self.frames[fid]['peers'] = [other for other in ids if other != fid]
        self.db_path = self.evidence / 'review' / 'decisions.sqlite3'
        self.db_path.parent.mkdir(exist_ok=True)
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS decisions (id TEXT PRIMARY KEY, version INTEGER NOT NULL, payload TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS history (id TEXT NOT NULL, version INTEGER NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(id, version))')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.db_path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def frame(self, fid):
        if not isinstance(fid, str) or fid not in self.frames:
            raise ValueError('Кадр не найден.')
        return self.frames[fid]

    def image_path(self, fid):
        r = self.frame(fid)
        path = self.source / 'images' / r['split'] / r['file']
        if not path.resolve().is_relative_to(self.source.resolve()):
            raise ValueError('Изображение вне исходного набора.')
        return path

    def verify_source(self, fid):
        r = self.frame(fid)
        image = self.image_path(fid)
        label = self.source / 'labels' / r['split'] / (r['stem'] + '.txt')
        if not label.resolve().is_relative_to(self.source.resolve()):
            raise ValueError('Разметка вне исходного набора.')
        if digest(image) != r['sha256'] or digest(label) != r['label_sha256']:
            raise ReviewConflict('Исходный кадр или разметка изменились после аудита. Сначала нужна повторная сверка.')
        return label

    def index(self):
        with self.connect() as db:
            decisions = {fid: json.loads(payload) for fid, payload in db.execute('SELECT id, payload FROM decisions')}
        report = self.background_report()
        predictions = report.get('frames', {})
        frames = []
        for fid, r in self.frames.items():
            decision = decisions.get(fid, {})
            prediction = self.background_prediction(r, decision, predictions.get(fid))
            frames.append({k: r[k] for k in ('id', 'stem', 'split', 'session', 'flags', 'candidate', 'reasons')} |
                          {'status': decision.get('status', 'pending'), 'version': decision.get('version', 0),
                           'yolo_background': bool(prediction and prediction['detections']),
                           'yolo_confidence': max((d['confidence'] for d in prediction['detections']), default=0) if prediction else 0})
        return {'frames': frames, 'total': len(frames), 'source': 'Размеченный набор с рига · 14 сентября',
                'storage': str(self.db_path.relative_to(self.root)),
                'background_scan': {k: v for k, v in report.items() if k != 'frames'}}

    def background_report(self):
        path = self.db_path.parent / 'yolo_background.json'
        if not path.exists():
            return {}
        return json.loads(path.read_text())

    @staticmethod
    def background_prediction(frame, decision, prediction):
        # A re-reviewed decision or changed source must not reuse an old finding.
        if (prediction and confirmed_background(decision) and
                prediction['decision_version'] == decision['version'] and
                prediction['source_image_sha256'] == frame['sha256'] and
                prediction['source_label_sha256'] == frame['label_sha256']):
            return prediction
        return None

    def detail(self, fid):
        r = self.frame(fid)
        label = self.verify_source(fid)
        versions = [{'key': 'source', 'name': 'С рига', **parse_labels(label.read_text()), 'sha256': r['label_sha256']}]
        for i, match in enumerate(r['matches']):
            path = self.root / match['dataset'] / 'labels' / match['split'] / (match['stem'] + '.txt')
            if not path.resolve().is_relative_to(self.root):
                continue
            name = f"{match['dataset'].replace('yolo_dataset_v4_', '')} · {match['split']}"
            if not path.is_file() or digest(path) != match['label_sha256']:
                versions.append({'key': f'local-{i}', 'name': name, 'unavailable': True,
                                 'error': 'Версия изменилась или недоступна; выбор заблокирован.'})
            else:
                versions.append({'key': f'local-{i}', 'name': name, **parse_labels(path.read_text()),
                                 'sha256': match['label_sha256']})
        with self.connect() as db:
            row = db.execute('SELECT payload FROM decisions WHERE id=?', (fid,)).fetchone()
            history = [json.loads(x[0]) for x in db.execute('SELECT payload FROM history WHERE id=? ORDER BY version DESC LIMIT 20', (fid,))]
        decision = json.loads(row[0]) if row else {}
        report = self.background_report()
        prediction = self.background_prediction(r, decision, report.get('frames', {}).get(fid))
        return {**r, 'versions': versions, 'decision': decision or None, 'history': history,
                'yolo_prediction': prediction,
                'yolo_model': report.get('model', {}) if prediction else {}}

    def save(self, data):
        if not isinstance(data, dict):
            raise ValueError('Ожидается объект с решением.')
        fid = data.get('id')
        r = self.frame(fid)
        self.verify_source(fid)
        status = data.get('status')
        if status not in ('approved', 'excluded', 'deferred'):
            raise ValueError('Неизвестное решение.')
        boxes = checked_boxes(data.get('boxes'))
        if status == 'approved' and not boxes and data.get('background_confirmed') is not True:
            raise ValueError('Подтвердите, что на кадре нет кораблей.')
        version = data.get('version')
        if type(version) is not int or version < 0:
            raise ValueError('Неверная версия решения.')
        for field, limit in [('note', 2000), ('recording', 200), ('chosen_version', 100)]:
            if not isinstance(data.get(field, ''), str) or len(data.get(field, '')) > limit:
                raise ValueError(f'Недопустимое поле: {field}')
        chosen = data.get('chosen_version', 'manual')
        if status == 'approved' and chosen != 'manual':
            detail = self.detail(fid)
            selected = next((v for v in detail['versions'] if v['key'] == chosen), None)
            if selected is None or selected.get('unavailable') or selected['invalid_lines'] or selected['boxes'] != boxes:
                raise ReviewConflict('Выбранная версия недоступна или рамки изменились. Нарисуйте рамку заново либо обновите кадр.')
        decision = {'id': fid, 'version': version+1, 'status': status, 'boxes': boxes,
                    'background_confirmed': status == 'approved' and not boxes,
                    'note': data.get('note', '').strip(), 'recording': data.get('recording', '').strip(),
                    'chosen_version': chosen, 'updated_at': datetime.now(timezone.utc).isoformat(),
                    'source_image_sha256': r['sha256'], 'source_label_sha256': r['label_sha256'],
                    'original_split': r['split'], 'review_flags': r['flags'], 'import_hold_reasons': r['reasons']}
        payload = json.dumps(decision, ensure_ascii=False, allow_nan=False)
        with self.lock, self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT version FROM decisions WHERE id=?', (fid,)).fetchone()
            if (row[0] if row else 0) != version:
                raise ReviewConflict('Решение уже изменено в другой вкладке. Обновите кадр перед сохранением.')
            db.execute('INSERT OR REPLACE INTO decisions VALUES (?, ?, ?)', (fid, version+1, payload))
            db.execute('INSERT INTO history VALUES (?, ?, ?)', (fid, version+1, payload))
        return decision

    def export(self):
        with self.connect() as db:
            decisions = [json.loads(row[0]) for row in db.execute('SELECT payload FROM decisions ORDER BY id')]
            history = [json.loads(row[0]) for row in db.execute('SELECT payload FROM history ORDER BY id, version')]
        # Verify provenance again: do not export stale approvals as valid corrections.
        for d in decisions:
            self.verify_source(d['id'])
        buffer = io.BytesIO()
        manifest = {'format': 'ship-annotation-review-v1', 'created_at': datetime.now(timezone.utc).isoformat(),
                    'training_ready': False,
                    'limitations': 'Reviewed annotations only. Original split names are retained for provenance, not approved for training. Resolve recording identity and held-out sessions before building a training dataset.',
                    'source_root': str(self.source.relative_to(self.root)), 'decisions': decisions}
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
            z.writestr('history.json', json.dumps(history, ensure_ascii=False, indent=2)+'\n')
            z.writestr('README.txt', 'Проверенные метки и решения. Исходные изображения не изменены.\n'
                       'Это не готовый обучающий набор: ещё требуется сверка записей и train/val/test.\n'
                       'В labels включены только одобренные кадры. Пустой файл означает подтверждённое отсутствие кораблей.\n')
            for d in decisions:
                if d['status'] != 'approved':
                    continue
                lines = []
                for x1, y1, x2, y2 in d['boxes']:
                    lines.append(f'0 {(x1+x2)/2:.10f} {(y1+y2)/2:.10f} {x2-x1:.10f} {y2-y1:.10f}')
                z.writestr('labels/'+d['id']+'.txt', '\n'.join(lines)+ ('\n' if lines else ''))
        return buffer.getvalue()
