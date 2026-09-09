import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description='Fit a linear correction (true = scale * raw + offset) from (raw, true) distance pairs, '
                    'in meters, as printed by run.py\'s "raw" readout.'
    )
    parser.add_argument('pairs', nargs='+', type=float,
                         help='flat list of raw,true,raw,true,... e.g. 2.35 1.0 3.10 1.5 4.40 2.0')
    args = parser.parse_args()

    if len(args.pairs) % 2 != 0 or len(args.pairs) < 4:
        raise SystemExit('provide at least 2 pairs: raw1 true1 raw2 true2 [raw3 true3 ...]')

    raw = np.array(args.pairs[0::2])
    true = np.array(args.pairs[1::2])

    scale, offset = np.polyfit(raw, true, 1)
    predicted = scale * raw + offset
    residual = true - predicted

    print(f'--calib-scale {scale:.4f} --calib-offset {offset:.4f}')
    for r, t, p, e in zip(raw, true, predicted, residual):
        print(f'  raw={r:.2f}  true={t:.2f}  corrected={p:.2f}  error={e:+.2f}')


if __name__ == '__main__':
    main()
