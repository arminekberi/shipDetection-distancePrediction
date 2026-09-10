import argparse
import os
import pickle

import cv2
from generate_yolo_labels import check_training_sources
from dataset_io import require_mounted_destination, save_annotation

WIDTH, HEIGHT = 640, 360
OUT = 'yolo_dataset_v4'
SPLIT = 'train'
CANDIDATES_PKL = 'review_hardmine_v1/candidates.pkl'

WRONG_RAW = """
174 175 193 211 212 215 233 258 272 277 280 281 283 284 286 296 298 305 360 362
372 373 429 435 451 453 482-489 518 540 554 555-559 562 565 567 569 570 571 573
574 586 587 601 630 631 634 639 641 644 645 646-664 666 667 670 672-680 683 690
698 700 704 706-715 745 794 814-827 832 859 891 898 908-938 1051 1108 1116 1119
1125 1126 1133-1136 1138 1139 1140
"""


def parse_indices(raw):
    out = set()
    for tok in raw.split():
        if '-' in tok:
            a, b = tok.split('-')
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(tok))
    return out


def tag_for(video):
    return os.path.splitext(os.path.basename(video))[0].replace(',', '_').replace(' ', '_')


def stem_for(tag, idx):
    return f'{tag}_{idx:05d}'


def main():
    parser = argparse.ArgumentParser(description='Import a reviewed hard-mining batch. Incorrect boxes are skipped unless independently confirmed as empty frames.')
    parser.add_argument('--confirmed-negative-indices', help='Text file of candidate indices confirmed to contain no boats')
    args = parser.parse_args()
    wrong = parse_indices(WRONG_RAW)
    negative = set()
    if args.confirmed_negative_indices:
        with open(args.confirmed_negative_indices) as f:
            negative = parse_indices(f.read())
        if not negative <= wrong:
            parser.error('Confirmed negatives must be a subset of the incorrect-box indices')
    require_mounted_destination(OUT)
    with open(CANDIDATES_PKL, 'rb') as f:
        candidates = pickle.load(f)
    check_training_sources(video for video, *_ in candidates)
    print(f'{len(candidates)} total candidates, {len(wrong)} flagged as incorrect boxes (not automatically negative)')

    for sub in ('images', 'labels'):
        os.makedirs(os.path.join(OUT, sub, SPLIT), exist_ok=True)

    # group needed frame indices per video (candidate position -> (video, frame_idx, box, is_positive))
    by_video = {}
    for i, (video, frame_idx, conf, box) in enumerate(candidates):
        if i in wrong and i not in negative:
            continue
        by_video.setdefault(video, []).append((frame_idx, box, i not in wrong))

    n_pos = n_neg = n_skip_existing = 0
    for video, items in by_video.items():
        tag = tag_for(video)
        needed = {fi: (box, is_pos) for fi, box, is_pos in items}
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {video}")
        idx = 0
        found = 0
        while found < len(needed):
            ret, frame = cap.read()
            if not ret:
                break
            if idx in needed:
                found += 1
                box, is_pos = needed[idx]
                stem = stem_for(tag, idx)
                label_path = os.path.join(OUT, 'labels', SPLIT, stem + '.txt')
                img_path = os.path.join(OUT, 'images', SPLIT, stem + '.jpg')
                if os.path.exists(label_path):
                    n_skip_existing += 1
                    idx += 1
                    continue
                frame_r = cv2.resize(frame, (WIDTH, HEIGHT))
                if is_pos:
                    x1, y1, x2, y2 = box
                    save_annotation(OUT, SPLIT, tag, idx, frame_r, (x1, y1, x2-x1, y2-y1))
                    n_pos += 1
                else:
                    save_annotation(OUT, SPLIT, tag, idx, frame_r)
                    n_neg += 1
            idx += 1
        cap.release()
        if found < len(needed):
            raise RuntimeError(f"Decoded only {found}/{len(needed)} required frames from {video}")
        print(f'{video}: processed {found}/{len(needed)} needed frames')

    print(f'\ndone: {n_pos} positive, {n_neg} negative, {n_skip_existing} skipped (already labeled)')


if __name__ == '__main__':
    main()
