#!/usr/bin/env python3
"""
Tekne mesafe kontrol paneli icin basit yerel sunucu.

Bu betik run.py'nin icerigine HIC dokunmaz. Sadece iki sey yapar:

1. Proje klasorunu statik olarak sunar (python -m http.server ile ayni davranis,
   dizin listelemesi dahil) - kontrol_paneli.html'nin results/ klasorunu otomatik
   tarayabilmesi icin gereken budur.
2. POST /api/process ile gelen istekte, mevcut run.py'yi degistirmeden bir alt
   surec olarak (headless + senkron modda) baslatir; sonucu results/<kayit adi>.csv
   ve results/<kayit adi>.mp4 olarak yazdirir. GET /api/status o an suren (varsa)
   isin durumunu doner.

Kullanim (proje klasorunde, sanal ortam aktifken):
    source venv/bin/activate        (ya da dogrudan: venv/bin/python panel_server.py)
    python panel_server.py

Sonra tarayicida http://localhost:8000/kontrol_paneli.html adresini acin.
"""
import http.server
import json
import os
import re
import subprocess
import sys
import threading
import time

PORT = 8000
ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")
RUN_PY = os.path.join(ROOT, "run.py")

# Path traversal'a karsi: sadece proje kokundeki duz .mp4 dosya adlarina izin ver
# (klasor ayirici veya ".." iceremez).
VIDEO_NAME_RE = re.compile(r'^[A-Za-z0-9_.-]+\.mp4$')
TARGET_RE = re.compile(r'^-?\d+,-?\d+,-?\d+,-?\d+$')

os.makedirs(RESULTS_DIR, exist_ok=True)

job_lock = threading.Lock()
job_state = {"status": "idle", "video": None}


def run_job(video_filename, target):
    base = os.path.splitext(video_filename)[0]
    out_mp4 = os.path.join(RESULTS_DIR, base + ".mp4")
    out_csv = os.path.join(RESULTS_DIR, base + ".csv")
    video_path = os.path.join(ROOT, video_filename)

    cmd = [
        sys.executable, RUN_PY,
        "--camera", video_path,
        "--no-display", "--sync",
        "--output", out_mp4,
        "--log", out_csv,
    ]
    if target:
        cmd += ["--target", target]

    job_state.update(
        status="running", video=video_filename, started_at=time.time(),
        out_csv=out_csv, out_mp4=out_mp4, error=None, finished_at=None,
    )
    print(f"[panel_server] baslatiliyor: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=60 * 30)
        if proc.returncode == 0:
            job_state.update(status="done", finished_at=time.time())
            print(f"[panel_server] tamamlandi: {video_filename}")
        else:
            err = (proc.stderr or proc.stdout or "")[-4000:]
            job_state.update(status="error", finished_at=time.time(), error=err)
            print(f"[panel_server] HATA ({video_filename}):\n{err}")
    except subprocess.TimeoutExpired:
        job_state.update(status="error", finished_at=time.time(),
                          error="islem 30 dakikayi asti, durduruldu")
    except Exception as e:
        job_state.update(status="error", finished_at=time.time(), error=str(e))


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def _send_json(self, payload, code=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/status"):
            return self._send_json(job_state)
        return super().do_GET()

    def do_POST(self):
        if not self.path.startswith("/api/process"):
            return self._send_json({"error": "bilinmeyen uc nokta"}, 404)

        length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            return self._send_json({"error": "gecersiz istek govdesi"}, 400)

        video = str(data.get("video", ""))
        target = data.get("target") or None

        if not VIDEO_NAME_RE.match(video) or not os.path.isfile(os.path.join(ROOT, video)):
            return self._send_json({"error": f"gecersiz ya da bulunamayan video: {video!r}"}, 400)
        if target is not None and not TARGET_RE.match(str(target)):
            return self._send_json({"error": "gecersiz target formati (x1,y1,x2,y2 bekleniyor)"}, 400)

        if not job_lock.acquire(blocking=False):
            return self._send_json({"error": "ayni anda sadece bir kayit islenebilir", **job_state}, 409)

        def worker():
            try:
                run_job(video, target)
            finally:
                job_lock.release()

        threading.Thread(target=worker, daemon=True).start()
        return self._send_json({"status": "started", "video": video})

    def log_message(self, fmt, *args):
        pass  # konsolu statik dosya isteklerinden dolayi doldurmasin


if __name__ == "__main__":
    if not os.path.isfile(RUN_PY):
        raise SystemExit(f"run.py bulunamadi: {RUN_PY}")

    server = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Kontrol paneli sunucusu calisiyor: http://localhost:{PORT}/kontrol_paneli.html")
    print(f"Model calistirmak icin kullanilan python: {sys.executable}")
    print("Durdurmak icin Ctrl+C")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\ndurduruldu")
