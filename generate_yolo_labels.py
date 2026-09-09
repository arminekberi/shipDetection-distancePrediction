import argparse
import os

# v4 dataset plan: fully manual annotation (see annotate_manual.py), split at the VIDEO level so
# no two frames from the same source video ever land in different splits. The 4 laser calibration
# clips (3,7 / 5 / 6,6m / 9) are excluded entirely - the boat sits still in them, near-duplicate
# frames, and the user is re-shooting dedicated test footage separately instead of reusing these.
#
# Each entry: (video_path, split, sample_every, is_negative)
#   sample_every: label every Nth frame with annotate_manual.py --sample-every N
#   is_negative: video confirmed to have no boat at all (labeled entirely as background)
VIDEO_SPLITS = [
    # --- train ---
    ('signal-2026-09-02-10-36-28-854.mp4', 'train', 1, False),
    ('yeniTrain/20260902-150149_ShipCam0.mp4', 'train', 1, False),
    ('yeniTrain/20260902-150707_ShipCam0.mp4', 'train', 1, False),
    ('yeniTrain/20260902-150830_ShipCam0.mp4', 'train', 1, False),
    ('yeniTrain/20260902-151003_ShipCam0.mp4', 'train', 1, False),
    ('test1.mp4', 'train', 1, False),
    ('testGerçekSiyahBeyaz.mp4', 'train', 2, False),
    ('testt.mp4', 'train', 2, False),
    ('testtt.mp4', 'train', 1, False),
    ('yeniTrain/20260902-150054_ShipCam0.mp4', 'train', 1, True),  # confirmed no boat

    # --- train (teknedenTekneyeGörüntü batch, 2026-09-04) ---
    ('teknedenTekneyeGörüntü/20260904-112014_ShipCam0.mp4', 'train', 1, False),
    ('teknedenTekneyeGörüntü/20260904-112149_ShipCam0.mp4', 'train', 1, False),
    ('teknedenTekneyeGörüntü/20260904-112600_ShipCam0.mp4', 'train', 1, False),
    ('teknedenTekneyeGörüntü/20260904-112749_ShipCam0.mp4', 'train', 1, False),
    ('teknedenTekneyeGörüntü/20260904-113010_ShipCam0.mp4', 'train', 1, False),
    ('teknedenTekneyeGörüntü/20260904-113444_ShipCam0.mp4', 'train', 2, False),
    ('teknedenTekneyeGörüntü/20260904-113812_ShipCam0.mp4', 'train', 1, False),

    # --- val ---
    ('yeniTrain/20260902-150337_ShipCam0.mp4', 'val', 1, False),
    ('yeniTrain/20260902-151121_ShipCam0.mp4', 'val', 1, False),
    ('test2.mp4', 'val', 1, False),
    ('testGerçekRenkli.mp4', 'val', 1, False),
    ('yeniTrain/20260902-150054_ShipCam1.mp4', 'val', 1, True),  # confirmed no boat

    # --- val (teknedenTekneyeGörüntü batch, 2026-09-04) ---
    ('teknedenTekneyeGörüntü/20260904-112339_ShipCam0.mp4', 'val', 1, False),
    ('teknedenTekneyeGörüntü/20260904-112926_ShipCam0.mp4', 'val', 1, False),
    ('teknedenTekneyeGörüntü/20260904-113234_ShipCam0.mp4', 'val', 1, False),
    ('teknedenTekneyeGörüntü/20260904-113321_ShipCam0.mp4', 'val', 1, False),

    # --- test only, NEVER label into train/val ---
    # ('teknedenTekneyeGörüntü/renkliTekneTekne.mp4', 'test', ...)
    # ('teknedenTekneyeGörüntü/renksizTekneTekne.mp4', 'test', ...)
]

OUT_DIR = 'yolo_dataset_v4'


def write_dataset_yaml():
    abs_out = os.path.abspath(OUT_DIR)
    with open(os.path.join(OUT_DIR, 'dataset.yaml'), 'w') as f:
        f.write(f'path: {abs_out}\n')
        f.write('train: images/train\n')
        f.write('val: images/val\n')
        f.write('names:\n  0: boat\n')
    print(f'wrote {OUT_DIR}/dataset.yaml')


def main():
    parser = argparse.ArgumentParser(description='Prints the annotate_manual.py commands to run for the v4 dataset plan.')
    parser.parse_args()

    for sub in ('images', 'labels'):
        for split in ('train', 'val'):
            os.makedirs(os.path.join(OUT_DIR, sub, split), exist_ok=True)
    write_dataset_yaml()

    print('\nRun each of these yourself (needs an interactive window):\n')
    for video, split, sample_every, is_negative in VIDEO_SPLITS:
        flag = ' --negative' if is_negative else ''
        cmd = f'python label_video.py --video "{video}" --split {split} --out {OUT_DIR} --sample-every {sample_every}{flag}'
        print(cmd)

    n_train = sum(1 for _, s, _, _ in VIDEO_SPLITS if s == 'train')
    n_val = sum(1 for _, s, _, _ in VIDEO_SPLITS if s == 'val')
    print(f'\n{n_train} train videos, {n_val} val videos, {len(VIDEO_SPLITS)} total')


if __name__ == '__main__':
    main()
