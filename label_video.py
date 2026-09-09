import argparse
import json
import os

import cv2

# Run this yourself in your own terminal (needs an interactive window).
#
# Flow: you draw ONE seed box on the first frame -> CSRT auto-tracks the box across the
# whole video, sampling every --sample-every frames -> then you flip through every sampled
# frame and fix whatever CSRT got wrong. This replaces drawing a box on every single frame
# by hand.
#
# Review controls (per frame):
#   SPACE / y   -> accept the shown box as-is, save, next
#   r           -> redraw the box for this frame (drag, ENTER/SPACE to confirm), save, next
#   n           -> mark this frame negative (no boat here), save empty label, next
#   d           -> discard this frame (don't save anything), next
#   b           -> go back one frame (re-review it)
#   q / ESC     -> quit, progress so far is already saved to disk (resumable)

WIDTH, HEIGHT = 640, 360
DRIFT_CORNER_FRAC = 0.04  # CSRT's known failure mode: collapses to a degenerate box pinned near (0,0)

SEED_CACHE = '_seed_boxes.json'


def tag_for(video, tag_override):
    return tag_override or os.path.splitext(os.path.basename(video))[0].replace(',', '_').replace(' ', '_')


def stem_for(tag, idx):
    return f'{tag}_{idx:05d}'


def load_seed_cache(out_dir):
    path = os.path.join(out_dir, SEED_CACHE)
    if os.path.exists(path):
        return json.load(open(path))
    return {}


def save_seed_cache(out_dir, cache):
    json.dump(cache, open(os.path.join(out_dir, SEED_CACHE), 'w'))


def decode_frames(video_path, sample_every):
    cap = cv2.VideoCapture(video_path)
    frames = {}
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_every == 0:
            frames[idx] = cv2.resize(frame, (WIDTH, HEIGHT))
        idx += 1
    cap.release()
    return frames


def csrt_autobox(frames, sorted_indices, seed_idx, seed_box):
    """CSRT-track forward from seed_idx across sorted_indices (must include seed_idx).
    Returns {idx: (x,y,w,h)} for every index it managed to track before drift/loss."""
    boxes = {}
    start_pos = sorted_indices.index(seed_idx)
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frames[seed_idx], seed_box)
    boxes[seed_idx] = seed_box
    prev_frame = frames[seed_idx]
    for idx in sorted_indices[start_pos + 1:]:
        frame = frames[idx]
        ok, box = tracker.update(frame)
        if not ok:
            break
        x, y, w, h = box
        cx, cy = (x + w / 2) / WIDTH, (y + h / 2) / HEIGHT
        if cx < DRIFT_CORNER_FRAC and cy < DRIFT_CORNER_FRAC:
            break
        boxes[idx] = box
    return boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--split', required=True, choices=['train', 'val', 'test'])
    parser.add_argument('--out', default='yolo_dataset_v4')
    parser.add_argument('--sample-every', type=int, default=1)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--negative', action='store_true', help='video confirmed to have no boat at all - skip seeding, label every sampled frame as background')
    parser.add_argument('--reseed', action='store_true', help='add a new seed box (e.g. CSRT lost track earlier and you want to pick up again further in)')
    parser.add_argument('--start-frame', type=int, default=None, help='frame index to start a new seed from (with --reseed); default: right after the last tracked segment')
    args = parser.parse_args()

    tag = tag_for(args.video, args.tag)
    for sub in ('images', 'labels'):
        os.makedirs(os.path.join(args.out, sub, args.split), exist_ok=True)

    print(f'decoding {args.video} (every {args.sample_every} frames)...')
    frames = decode_frames(args.video, args.sample_every)
    sorted_indices = sorted(frames.keys())
    print(f'{len(sorted_indices)} sampled frames')

    window = f'{tag} [{args.split}]'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window, 100, 100)
    cv2.imshow(window, next(iter(frames.values())))
    cv2.waitKey(1)
    try:
        cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        pass

    if args.negative:
        review_indices = sorted_indices
        boxes = {}
    else:
        seed_cache = load_seed_cache(args.out)
        raw = seed_cache.get(tag, [])
        segments = raw if isinstance(raw, list) else ([raw] if raw else [])  # back-compat w/ old single-dict cache

        if args.reseed or not segments:
            last_tracked = max((max(csrt_autobox(frames, sorted_indices, s['idx'], tuple(s['box'])).keys(), default=s['idx'])
                                 for s in segments), default=None)
            if args.start_frame is not None:
                seed_pos = min(range(len(sorted_indices)), key=lambda p: abs(sorted_indices[p] - args.start_frame))
            elif last_tracked is not None:
                seed_pos = min(sorted_indices.index(last_tracked) + 1, len(sorted_indices) - 1)
            else:
                seed_pos = 0
            seed_idx = sorted_indices[seed_pos]
            seed_box = None
            while seed_box is None:
                print(f'draw a box around the boat on frame {seed_idx} (drag, then ENTER/SPACE). '
                      f'ESC/no drag -> try the next frame instead.')
                cv2.imshow(window, frames[seed_idx])
                cv2.waitKey(1)
                box = cv2.selectROI(window, frames[seed_idx], showCrosshair=True)
                if box[2] >= 2 and box[3] >= 2:
                    seed_box = box
                    break
                seed_pos += 1
                if seed_pos >= len(sorted_indices):
                    print('reached end of video without a seed box, aborting')
                    cv2.destroyAllWindows()
                    return
                seed_idx = sorted_indices[seed_pos]
            segments.append({'idx': seed_idx, 'box': list(seed_box)})
            seed_cache[tag] = segments
            save_seed_cache(args.out, seed_cache)

        boxes = {}
        for s in segments:
            print(f'auto-tracking with CSRT from frame {s["idx"]}...')
            seg_boxes = csrt_autobox(frames, sorted_indices, s['idx'], tuple(s['box']))
            print(f'  tracked {len(seg_boxes)} frames from this seed before drift/loss (or end of video)')
            boxes.update(seg_boxes)
        print(f'tracked {len(boxes)}/{len(sorted_indices)} frames total across {len(segments)} seed(s)')
        review_indices = [i for i in sorted_indices if i in boxes]

        pending = [i for i in review_indices if not os.path.exists(
            os.path.join(args.out, 'labels', args.split, stem_for(tag, i) + '.txt'))]
        if not pending:
            last = max(review_indices) if review_indices else segments[-1]['idx']
            if last < sorted_indices[-1]:
                print(f'\nall {len(review_indices)} CSRT-tracked frames are already labeled (up to frame {last} '
                      f'of {sorted_indices[-1]}). To keep going further into the video, run:')
                print(f'  python label_video.py --video "{args.video}" --split {args.split} --out {args.out} '
                      f'--sample-every {args.sample_every} --reseed --start-frame {last + 1}')
            else:
                print(f'\nall {len(review_indices)} CSRT-tracked frames are already labeled - this video is done.')
            cv2.destroyAllWindows()
            return

    print(f'\nreviewing {len(review_indices)} frames - SPACE/y=accept  r=redraw  n=negative  d=discard  b=back  q=quit\n')

    i = 0
    n_ok = n_fix = n_neg = n_disc = 0
    while i < len(review_indices):
        idx = review_indices[i]
        stem = stem_for(tag, idx)
        label_path = os.path.join(args.out, 'labels', args.split, stem + '.txt')
        if os.path.exists(label_path):
            i += 1
            continue

        frame = frames[idx].copy()
        box = boxes.get(idx)
        disp = frame.copy()
        if box is not None:
            x, y, w, h = [int(v) for v in box]
            cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(disp, f'{tag} frame {idx}  ({i+1}/{len(review_indices)})', (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(disp, 'SPACE/y=ok  r=redraw  n=negative  d=discard  b=back  q=quit', (5, HEIGHT - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.imshow(window, disp)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord('b'):
            i = max(0, i - 1)
            continue
        elif key == ord('d'):
            n_disc += 1
            i += 1
            continue
        elif key == ord('n'):
            cv2.imwrite(os.path.join(args.out, 'images', args.split, stem + '.jpg'), frame)
            open(label_path, 'w').close()
            n_neg += 1
            i += 1
            continue
        elif key == ord('r'):
            new_box = cv2.selectROI(window, frame, showCrosshair=True)
            if new_box[2] < 2 or new_box[3] < 2:
                continue  # cancelled, stay on this frame
            box = new_box
            n_fix += 1
        elif key not in (ord(' '), ord('y')):
            continue  # unrecognized key, redraw same frame

        if box is None:
            continue  # no box to accept and no key handled it (e.g. blank frame) - stay put

        x, y, w, h = box
        cx, cy = (x + w / 2) / WIDTH, (y + h / 2) / HEIGHT
        nw, nh = w / WIDTH, h / HEIGHT
        cv2.imwrite(os.path.join(args.out, 'images', args.split, stem + '.jpg'), frame)
        with open(label_path, 'w') as f:
            f.write(f'0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n')
        n_ok += 1
        i += 1

    cv2.destroyAllWindows()
    print(f'\nthis session: {n_ok} accepted, {n_fix} redrawn, {n_neg} negative, {n_disc} discarded')
    print('run again on the same --video/--split to resume (already-labeled frames are skipped)')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()
        input('\ncrashed - press ENTER to close this window (see the error above)')
