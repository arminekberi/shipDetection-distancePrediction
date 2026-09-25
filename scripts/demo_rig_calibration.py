"""Generate explicitly synthetic two-rig logs for an end-to-end pipeline smoke test."""
import argparse
import csv
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from boatdet.supervision import STATE_FIELDS, truth, write_json


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate(folder, duration=80):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {'format': 'two-rig-supervision-v1', 'world_frame': 'shared_local_xyz_z_up',
                'world_frame_id': 'SYNTHETIC-WORLD', 'synthetic': True,
                'max_state_gap_s': .6, 'sessions': []}
    rng = np.random.default_rng(12)
    for phase, split in enumerate(('train','val','test')):
        observer, target, observations = [], [], []
        own_clock = {'offset_s': 1000., 'scale': 1.00001}
        target_clock = {'offset_s': 5000., 'scale': .99999}
        for i in range(int(duration*4)+1):
            t = i/4
            own = dict(zip(STATE_FIELDS, [.2*t, 3*math.sin(t/7+phase), 0.,
                        .2, 3/7*math.cos(t/7+phase), 0., .08*math.sin(t/10), 0., 0.]))
            other = dict(zip(STATE_FIELDS, [45+.1*t, 8.+phase, 0., .1, 0., 0., 0., 0., 0.]))
            for state in (own,other):
                state.update(position_sigma_m=.01, heading_sigma_rad=.002)
            gt = truth(own,other,[0,0,-1],[0,0,0])
            observer.append({'timestamp': own_clock['offset_s']+own_clock['scale']*t, **own, 'valid':1})
            target.append({'timestamp': target_clock['offset_s']+target_clock['scale']*t, **other, 'valid':1})
            missed = int(i % 17 == 16)
            observations.append({'timestamp':t, 'frame_id':i, 'track_id':1,
                                 'missed_frames':missed, 'tracking_status':'predicted' if missed else 'observed',
                                 'visual_bearing_deg':gt['relative_bearing_deg']+1.7+rng.normal(0,.1),
                                 'visual_depth_m':(gt['horizontal_range_m']-2)/1.3+rng.normal(0,.03)})
        write_csv(folder/f'{split}_observer.csv',observer)
        write_csv(folder/f'{split}_target.csv',target)
        write_csv(folder/f'{split}_observations.csv',observations)
        manifest['sessions'].append({'id':split, 'split':split, 'group':f'synthetic-run-{split}',
            'capture_id':f'synthetic-capture-{split}', 'camera_id':'synthetic-camera',
            'bearing_source':'logged_visual_bearing', 'observations':f'{split}_observations.csv',
            'observer':{'rig_id':'rig-A','path':f'{split}_observer.csv','clock':own_clock,
                        'reference_lever_frd_m':[0,0,-1]},
            'target':{'rig_id':'rig-B','path':f'{split}_target.csv','clock':target_clock,
                      'reference_lever_frd_m':[0,0,0]},
            'associations':[{'track_id':1,'start_s':0,'end_s':duration+.25}]})
    write_json(folder/'manifest.json',manifest)
    return folder/'manifest.json'


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    print(generate(args.out))
