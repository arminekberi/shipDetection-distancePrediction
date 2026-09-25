"""Fit run.py --calib-scale/--calib-offset from (model, true) distance pairs in meters.

Use distance_model_m from run.py CSV output against independent measurements.
"""
import argparse

from boatdet.depth import fit_calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pairs', nargs='+', type=float, help='model1 true1 model2 true2 ..., e.g. 2.35 1.0 3.10 1.5')
    args = parser.parse_args()
    if len(args.pairs) % 2 or len(args.pairs) < 4:
        parser.error('provide at least 2 pairs: model1 true1 model2 true2 [...]')
    raw, true = args.pairs[0::2], args.pairs[1::2]
    scale, offset = fit_calibration(raw, true)
    print(f'--calib-scale {scale:.4f} --calib-offset {offset:.4f}')
    for r, t in zip(raw, true):
        corrected = scale * r + offset
        print(f'  model={r:.2f}  true={t:.2f}  corrected={corrected:.2f}  error={t - corrected:+.2f}')


if __name__ == '__main__':
    main()
