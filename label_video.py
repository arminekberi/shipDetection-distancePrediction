import argparse
import os

import cv2

# Run this yourself in your own terminal (needs an interactive window).
#
# Flow: one continuous pass over every sampled frame. CSRT auto-tracks forward from
# whatever box you last drew/accepted; whenever it has no valid box for the current
# frame (first frame, lost track, or drifted into the corner) the window stays open and
# waits for you right there - draw a new box (r) or mark "no boat" (n). The window never
# closes on its own mid-video; it only closes when you quit (q) or the video ends.
#
# Controls (per frame):
#   SPACE / y   -> accept the shown (CSRT-tracked) box as-is, save, next
#   r           -> draw/redraw a box for this frame (drag, ENTER/SPACE to confirm), save, next
#                  (this is also how you (re)seed CSRT after it lost the target)
#   n           -> mark this frame negative (no boat here), save empty label, next
#   d           -> discard this frame (don't save anything), next
#   b           -> go back one frame (re-review it; CSRT is reseeded fresh from there)
#   q / ESC     -> quit, progress so far is already saved to disk (resumable)

WIDTH, HEIGHT = 640, 360
DRIFT_CORNER_FRAC = 0.04  # CSRT's known failure mode: collapses to a degenerate box pinned near (0,0)


def tag_for(video, tag_override):
    return tag_override or os.path.splitext(os.path.basename(video))[0].replace(',', '_').replace(' ', '_')


def stem_for(tag, idx):
    return f'{tag}_{idx:05d}'


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--split', required=True, choices=['train', 'val', 'test'])
    parser.add_argument('--out', default='yolo_dataset_v4')
    parser.add_argument('--sample-every', type=int, default=1)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--negative', action='store_true', help='video confirmed to have no boat at all - skip seeding, label every sampled frame as background')
    args = parser.parse_args()

    tag = tag_for(args.video, args.tag)
    for sub in ('images', 'labels'):
        os.makedirs(os.path.join(args.out, sub, args.split), exist_ok=True)

    print(f'decoding {args.video} (every {args.sample_every} frames)...')
    frames = decode_frames(args.video, args.sample_every)
    sorted_indices = sorted(frames.keys())
    print(f'{len(sorted_indices)} sampled frames')

    pending_indices = [i for i in sorted_indices if not os.path.exists(
        os.path.join(args.out, 'labels', args.split, stem_for(tag, i) + '.txt'))]
    if not pending_indices:
        print(f'\nall {len(sorted_indices)} sampled frames are already labeled - this video is done.')
        return

    window = f'{tag} [{args.split}]'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window, 100, 100)
    cv2.imshow(window, frames[pending_indices[0]])
    cv2.waitKey(1)
    try:
        cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        pass

    if args.negative:
        print(f'\nlabeling {len(pending_indices)} frames as negative (--negative)\n')
        for idx in pending_indices:
            stem = stem_for(tag, idx)
            cv2.imwrite(os.path.join(args.out, 'images', args.split, stem + '.jpg'), frames[idx])
            open(os.path.join(args.out, 'labels', args.split, stem + '.txt'), 'w').close()
        cv2.destroyAllWindows()
        print(f'done: {len(pending_indices)} negative frames saved')
        return

    print(f'\nreviewing {len(pending_indices)} frames - SPACE/y=accept  r=draw/redraw  n=negative  d=discard  b=back  q=quit\n')
    print('CSRT loses the target or you press b -> the window stays open and waits for you to')
    print('draw a new box (r) or mark negative (n) right there; it never closes mid-video.\n')

    tracker = None  # cv2.TrackerCSRT or None (no valid box for the upcoming frame yet)
    i = 0
    n_ok = n_fix = n_neg = n_disc = 0
    while i < len(pending_indices):
        idx = pending_indices[i]
        stem = stem_for(tag, idx)
        label_path = os.path.join(args.out, 'labels', args.split, stem + '.txt')
        if os.path.exists(label_path):
            i += 1
            continue

        frame = frames[idx].copy()

        box = None
        if tracker is not None:
            ok, tbox = tracker.update(frame)
            if ok:
                x, y, w, h = tbox
                cx, cy = (x + w / 2) / WIDTH, (y + h / 2) / HEIGHT
                if not (cx < DRIFT_CORNER_FRAC and cy < DRIFT_CORNER_FRAC):
                    box = tbox
            if box is None:
                tracker = None  # lost or drifted - drop it, this frame needs a fresh decision

        disp = frame.copy()
        if box is not None:
            x, y, w, h = [int(v) for v in box]
            cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 1)
            status = 'SPACE/y=accept  r=redraw  n=negative  d=discard  b=back  q=quit'
        else:
            status = 'no target - r=draw box  n=no boat here  d=skip  b=back  q=quit'
        cv2.putText(disp, f'{tag} frame {idx}  ({i+1}/{len(pending_indices)})', (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(disp, status, (5, HEIGHT - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.imshow(window, disp)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord('b'):
            tracker = None  # reseed fresh when we come back to (or past) this frame
            i = max(0, i - 1)
            continue
        elif key == ord('d'):
            n_disc += 1
            i += 1
            continue
        elif key == ord('n'):
            cv2.imwrite(os.path.join(args.out, 'images', args.split, stem + '.jpg'), frame)
            open(label_path, 'w').close()
            tracker = None  # whatever was (maybe) being tracked wasn't a real boat
            n_neg += 1
            i += 1
            continue
        elif key == ord('r'):
            new_box = cv2.selectROI(window, frame, showCrosshair=True)
            if new_box[2] < 2 or new_box[3] < 2:
                continue  # cancelled, stay on this frame
            box = new_box
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, tuple(int(v) for v in box))
            n_fix += 1
        elif key not in (ord(' '), ord('y')):
            continue  # unrecognized key, redraw same frame

        if box is None:
            continue  # nothing to accept yet (e.g. SPACE with no target) - stay put

        x, y, w, h = box
        cx, cy = (x + w / 2) / WIDTH, (y + h / 2) / HEIGHT
        nw, nh = w / WIDTH, h / HEIGHT
        cv2.imwrite(os.path.join(args.out, 'images', args.split, stem + '.jpg'), frame)
        with open(label_path, 'w') as f:
            f.write(f'0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n')
        n_ok += 1
        i += 1

    cv2.destroyAllWindows()
    print(f'\nthis session: {n_ok} accepted, {n_fix} drawn/redrawn, {n_neg} negative, {n_disc} discarded')
    print('run again on the same --video/--split to resume (already-labeled frames are skipped)')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()
        input('\ncrashed - press ENTER to close this window (see the error above)')
