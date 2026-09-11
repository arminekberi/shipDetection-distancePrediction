#!/usr/bin/env python3
"""Local recording viewer and one-job-at-a-time offline processing API.

Run ``venv/bin/python panel_server.py`` and open
http://127.0.0.1:8001/control_panel.html. Server recordings remain accessible
through the existing shipcaps mounts. Newly processed results are written to
results/by_recording; use ?results=local in the viewer to inspect them.
"""
import http.server
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.parse import unquote, urlsplit
import uuid

PORT = 8001
ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / 'results' / 'by_recording'
RUN_PY = ROOT / 'run.py'
DEFAULT_WEIGHTS = ROOT / 'weights' / 'boat_v4_s_best.pt'
job_lock = threading.Lock()
state_lock = threading.Lock()
job_state = {'status': 'idle', 'video': None}


def update_state(**values):
    with state_lock:
        job_state.update(values)


def snapshot():
    with state_lock:
        return dict(job_state)


def validate_request(data):
    if not isinstance(data, dict):
        raise ValueError('Request body must be a JSON object')
    video = data.get('video')
    if not isinstance(video, str) or not video or Path(video).is_absolute():
        raise ValueError('video must be a relative path inside the project')
    path = (ROOT / video).resolve()
    if not path.is_relative_to(ROOT.resolve()) or path.suffix.lower() != '.mp4' or not path.is_file():
        raise ValueError('Video is missing or outside the project')
    target = data.get('target')
    if target is not None:
        if not isinstance(target, str):
            raise ValueError('target must be x1,y1,x2,y2')
        try:
            x1, y1, x2, y2 = map(int, target.split(','))
        except ValueError as exc:
            raise ValueError('target must be x1,y1,x2,y2') from exc
        if not (0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 360 and x2-x1 >= 2 and y2-y1 >= 2):
            raise ValueError('target must be inside the 640x360 working frame')
    elif not DEFAULT_WEIGHTS.is_file():
        raise ValueError(f'Default detector weights are missing: {DEFAULT_WEIGHTS.name}')
    return video, target


def run_job(video, target):
    folder = Path(video).stem + '_' + uuid.uuid4().hex[:12]
    out_dir = RESULTS_DIR / folder
    try:
        out_dir.mkdir(parents=True)
        out_mp4, out_csv = out_dir / 'result.mp4', out_dir / 'result.csv'
        cmd = [sys.executable, str(RUN_PY), '--camera', str(ROOT / video),
               '--no-display', '--sync', '--output', str(out_mp4), '--log', str(out_csv)]
        cmd += ['--target', target] if target is not None else ['--yolo-weights', str(DEFAULT_WEIGHTS)]
        update_state(status='running', video=video, started_at=time.time(),
                     out_csv=str(out_csv), out_mp4=str(out_mp4), error=None, finished_at=None)
        with (out_dir / 'process.log').open('w+') as log:
            proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            if proc.returncode:
                log.seek(max(0, log.tell() - 4000))
                raise RuntimeError(log.read() or f'Processing exited with status {proc.returncode}')
        if not out_mp4.is_file() or out_mp4.stat().st_size == 0 or not out_csv.is_file():
            raise RuntimeError('Processing finished without valid output files')
        metadata = json.loads(out_mp4.with_suffix('.meta.json').read_text())
        if metadata.get('status') != 'complete' or not metadata.get('frames'):
            raise RuntimeError('Processing metadata does not confirm a completed recording')
        manifest_path = RESULTS_DIR / 'manifest.json'
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'recordings': []}
        manifest['recordings'].append(dict(metadata, folder=folder, source_video=video,
                                           output_video='result.mp4', output_csv='result.csv'))
        temporary = manifest_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(manifest, indent=2) + '\n')
        temporary.replace(manifest_path)
        update_state(status='done', finished_at=time.time())
    except subprocess.TimeoutExpired:
        update_state(status='error', finished_at=time.time(), error='Processing exceeded 30 minutes and was stopped')
    except Exception as exc:
        update_state(status='error', finished_at=time.time(), error=str(exc))


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def _send_json(self, payload, code=200):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def send_head(self):
        path = unquote(urlsplit(self.path).path)
        if path in ('/', '/kontrol_paneli.html'):
            self.send_response(302)
            self.send_header('Location', '/control_panel.html')
            self.end_headers()
            return None
        parts = Path(path.lstrip('/')).parts
        allowed = {'control_panel.html', 'panel_data.js', 'results', 'thumbnails', 'shipcaps_remote', 'shipcaps1_remote'}
        if not parts or parts[0] not in allowed or any(p.startswith('.') for p in parts):
            self.send_error(403, 'Only viewer assets and recording folders are served')
            return None
        return super().send_head()

    def do_GET(self):
        if urlsplit(self.path).path == '/api/status':
            return self._send_json(snapshot())
        return super().do_GET()

    def do_POST(self):
        if urlsplit(self.path).path != '/api/process':
            return self._send_json({'error': 'Unknown endpoint'}, 404)
        if self.headers.get_content_type() != 'application/json':
            return self._send_json({'error': 'Content-Type must be application/json'}, 415)
        origin = self.headers.get('Origin')
        if origin and origin != 'http://' + self.headers.get('Host', ''):
            return self._send_json({'error': 'Cross-origin processing requests are not accepted'}, 403)
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 16384:
                raise ValueError('Request body must be between 1 and 16384 bytes')
            video, target = validate_request(json.loads(self.rfile.read(length)))
        except (ValueError, OSError) as exc:
            return self._send_json({'error': str(exc)}, 400)
        if not job_lock.acquire(blocking=False):
            return self._send_json({**snapshot(), 'error': 'Only one recording can be processed at a time'}, 409)
        update_state(status='queued', video=video, error=None, finished_at=None)

        def worker():
            try:
                run_job(video, target)
            finally:
                job_lock.release()

        try:
            threading.Thread(target=worker, daemon=True).start()
        except Exception:
            job_lock.release()
            raise
        return self._send_json({'status': 'started', 'video': video}, 202)

    def log_message(self, fmt, *args):
        pass


if __name__ == '__main__':
    server = http.server.ThreadingHTTPServer(('127.0.0.1', PORT), Handler)
    print(f'Control panel: http://127.0.0.1:{PORT}/control_panel.html')
    print(f'Processing interpreter: {sys.executable}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped')
    finally:
        server.server_close()
