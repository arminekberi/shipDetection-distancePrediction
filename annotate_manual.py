import argparse
import os

import cv2

# Run this yourself in your own terminal (needs an interactive window - camera/GUI
# permission doesn't work through the agent's sandboxed shell).
#
# Controls per frame:
#   r                               -> draw a box (drag, then ENTER/SPACE) -> save as positive (boat)
#   n                               -> save as negative (no boat) example, empty label
#   s                               -> skip this frame (don't save anything, move on)
#   b                               -> go back one frame (re-decide it)
#   q / ESC                        -> quit and save progress (resumable - already-labeled
#                                      frames are skipped automatically next run)

WIDTH, HEIGHT = 640, 360


def stem_for(tag, idx):
    return f'{tag}_{idx:05d}'


def already_labeled(out_dir, split, tag, idx):
    stem = stem_for(tag, idx)
    return os.path.exists(os.path.join(out_dir, 'labels', split, stem + '.txt'))


def save_positive(out_dir, split, tag, idx, frame, box_xywh):
    x, y, w, h = box_xywh
    cx = (x + w / 2) / WIDTH
    cy = (y + h / 2) / HEIGHT
    nw = w / WIDTH
    nh = h / HEIGHT
    stem = stem_for(tag, idx)
    cv2.imwrite(os.path.join(out_dir, 'images', split, stem + '.jpg'), frame)
    with open(os.path.join(out_dir, 'labels', split, stem + '.txt'), 'w') as f:
        f.write(f'0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n')


def save_negative(out_dir, split, tag, idx, frame):
    stem = stem_for(tag, idx)
    cv2.imwrite(os.path.join(out_dir, 'images', split, stem + '.jpg'), frame)
    open(os.path.join(out_dir, 'labels', split, stem + '.txt'), 'w').close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True, help='path to the source video')
    parser.add_argument('--split', required=True, choices=['train', 'val', 'test'])
    parser.add_argument('--out', default='yolo_dataset_v4')
    parser.add_argument('--sample-every', type=int, default=1, help='label every Nth frame (default: every frame)')
    parser.add_argument('--tag', default=None, help='override the label prefix (default: derived from filename)')
    args = parser.parse_args()

    tag = args.tag or os.path.splitext(os.path.basename(args.video))[0].replace(',', '_').replace(' ', '_')

    for sub in ('images', 'labels'):
        os.makedirs(os.path.join(args.out, sub, args.split), exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # decode all sampled frames into memory up front so 'b' (go back) is trivial and cheap
    frames = {}
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % args.sample_every == 0:
            frames[idx] = cv2.resize(frame, (WIDTH, HEIGHT))
        idx += 1
    cap.release()

    indices = sorted(frames.keys())
    print(f'{args.video}: {len(indices)} frames to review (of {total} total, every {args.sample_every})')

    window = f'annotate: {tag} [{args.split}]'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    i = 0
    n_pos = n_neg = n_skip = 0
    while i < len(indices):
        idx = indices[i]
        if already_labeled(args.out, args.split, tag, idx):
            i += 1
            continue

        frame = frames[idx].copy()
        overlay = frame.copy()
        cv2.putText(overlay, f'{tag} frame {idx}  ({i+1}/{len(indices)})', (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(overlay, 'r=draw box(boat)  n=no boat  s=skip  b=back  q=quit', (5, HEIGHT - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.imshow(window, overlay)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord('n'):
            save_negative(args.out, args.split, tag, idx, frame)
            n_neg += 1
            i += 1
        elif key == ord('s'):
            n_skip += 1
            i += 1
        elif key == ord('b'):
            i = max(0, i - 1)
        elif key == ord('r'):
            box = cv2.selectROI(window, frame, showCrosshair=True)
            x, y, w, h = box
            if w > 2 and h > 2:
                save_positive(args.out, args.split, tag, idx, frame, box)
                n_pos += 1
                i += 1
            # zero-size box (cancelled) -> stay on this frame, let them retry

    cv2.destroyAllWindows()
    print(f'\ndone (this session): {n_pos} positive, {n_neg} negative, {n_skip} skipped')
    print(f'run again on the same --video/--split to resume where you left off (already-saved frames are skipped)')


if __name__ == '__main__':
    main()
