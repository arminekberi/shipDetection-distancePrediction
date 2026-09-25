"""Read-only rig inventory. Can run through SSH stdin; no third-party packages needed.

python3 scripts/rig_inventory.py > rig.json
ssh RIG python3 - < scripts/rig_inventory.py > rig.json
"""
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import shutil
import statistics
import urllib.request


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return {'error': str(exc)}


def recording(camera_dir):
    manifest = read_json(camera_dir / 'manifest.json')
    files = sorted(p for p in camera_dir.iterdir() if p.suffix in ('.bgr', '.gray'))
    expected = manifest.get('bytes_per_frame')
    stamps, sizes, invalid = [], [], []
    for path in files:
        size = path.stat().st_size
        sizes.append(size)
        if expected and size != expected:
            invalid.append(path.name)
        match = re.search(r'_(\d{13})', path.name)
        if match:
            stamps.append(int(match.group(1)))
    filename_order_reversals = sum(b < a for a, b in zip(stamps, stamps[1:]))
    stamps.sort()
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    duration = (stamps[-1] - stamps[0]) / 1000 if len(stamps) == len(files) and len(stamps) > 1 else None
    return dict(session=camera_dir.parent.name, camera=camera_dir.name, frames=len(files),
                suffixes=sorted({p.suffix for p in files}),
                bytes=sum(sizes), manifest=manifest, invalid_size_frames=invalid,
                duration_s=duration,
                actual_fps=(len(files) - 1) / duration if duration and duration > 0 else None,
                frame_interval_ms_median=statistics.median(gaps) if gaps else None,
                nonpositive_timestamp_gaps=sum(gap <= 0 for gap in gaps),
                filename_order_reversals=filename_order_reversals,
                pipeline=read_json(camera_dir.parent / 'pipeline.json'))


def main():
    usage = shutil.disk_usage('/root/shipcaps')
    report = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  host=platform.node(), platform=platform.platform(), python=platform.python_version(),
                  disk=dict(total_bytes=usage.total, free_bytes=usage.free), services={},
                  versions={}, recordings=[])
    for route in ('status', 'camera/settings', 'pipeline/settings', 'shipdet/status', 'telemetry'):
        try:
            with urllib.request.urlopen('http://127.0.0.1:8080/' + route, timeout=5) as response:
                report['services'][route] = json.load(response)
        except Exception as exc:
            report['services'][route] = {'error': str(exc)}
    for name in ('torch', 'torchvision', 'ultralytics', 'numpy', 'opencv-python', 'tensorrt'):
        try:
            report['versions'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report['versions'][name] = None
    for session in sorted(Path('/root/shipcaps').glob('20*')):
        if session.is_dir():
            for camera in sorted(session.glob('ShipCam*')):
                if camera.is_dir():
                    report['recordings'].append(recording(camera))
    report['models'] = []
    for path in sorted(Path('/root/HydroRL/models').glob('*')):
        if path.suffix in ('.pt', '.engine'):
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            report['models'].append(dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest()))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
