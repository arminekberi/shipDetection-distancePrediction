import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bearing_noise import chain_detections, measure, read_tracks, residuals, robust_sigma
from boatdet.telemetry import TRACK_CSV_HEADER


def track_csv(path, rows):
    """A track CSV with only the columns the measurement reads filled in."""
    lines = [','.join(TRACK_CSV_HEADER)]
    for row in rows:
        record = dict.fromkeys(TRACK_CSV_HEADER, '')
        record.update(row)
        lines.append(','.join(str(record[column]) for column in TRACK_CSV_HEADER))
    Path(path).write_text('\n'.join(lines) + '\n')


class ResidualTests(unittest.TestCase):
    def test_smooth_motion_leaves_no_residual(self):
        times = np.arange(40) * 0.1
        # A quadratic is exactly what the local fit removes: pure target motion, no jitter.
        values = 100.0 + 12.0 * times + 3.0 * times ** 2
        self.assertLess(max(abs(r) for r in residuals(times, values, 9, 2, 0.5)), 1e-6)

    def test_the_leverage_correction_recovers_the_true_sigma(self):
        rng = np.random.default_rng(0)
        times = np.arange(4000) * 0.1
        values = 50.0 + 2.0 * times + rng.normal(0.0, 3.0, times.size)
        measured = np.sqrt(np.mean(np.square(residuals(times, values, 9, 2, 0.5))))
        # Without the correction this lands near 2.4 px: the fit absorbs part of the noise.
        self.assertAlmostEqual(measured, 3.0, delta=0.2)

    def test_a_window_spanning_a_gap_is_not_fitted_across_it(self):
        times = np.concatenate([np.arange(10) * 0.1, np.arange(10) * 0.1 + 30.0])
        # The two stretches are smooth but far apart in value: a window fitted across
        # the gap would extrapolate wildly and show it in the residual.
        values = np.concatenate([np.arange(10, dtype=float), np.arange(10, dtype=float) + 5000.0])
        self.assertLess(max(abs(r) for r in residuals(times, values, 9, 2, 0.5)), 1e-6)
        self.assertGreater(max(abs(r) for r in residuals(times, values, 9, 2, 1e6)), 10.0)

    def test_robust_sigma_ignores_a_few_large_outliers(self):
        rng = np.random.default_rng(1)
        values = np.concatenate([rng.normal(0.0, 1.0, 500), [80.0, -95.0]])
        self.assertAlmostEqual(robust_sigma(values), 1.0, delta=0.15)
        self.assertGreater(np.sqrt(np.mean(np.square(values))), 4.0)   # RMS is not robust


class ChainTests(unittest.TestCase):
    def frames(self, count=20, step=3.0, drop=()):
        return [(index * 0.1, [] if index in drop else [(100 + step * index, 50.0,
                                                         130 + step * index, 80.0)])
                for index in range(count)]

    def test_a_moving_detection_forms_one_chain(self):
        chains = chain_detections(self.frames(), max_jump_px=20.0)
        self.assertEqual(len(chains), 1)
        self.assertEqual(len(chains[0]), 20)

    def test_a_short_gap_is_bridged_and_stays_a_gap_in_the_series(self):
        chains = chain_detections(self.frames(drop=(7, 8)), max_jump_px=20.0, max_miss=3)
        self.assertEqual(len(chains), 1)
        self.assertEqual(len(chains[0]), 18)   # bridged, not filled in
        times = [sample[0] for sample in chains[0]]
        self.assertAlmostEqual(max(np.diff(times)), 0.3, places=6)

    def test_a_long_gap_closes_the_chain_instead_of_coasting(self):
        chains = chain_detections(self.frames(drop=(7, 8, 9, 10)), max_jump_px=20.0, max_miss=2)
        self.assertEqual(sorted(len(chain) for chain in chains), [7, 9])

    def test_a_jump_beyond_the_gate_starts_a_new_chain(self):
        chains = chain_detections(self.frames(step=60.0), max_jump_px=20.0, max_miss=0)
        self.assertEqual(len(chains), 20)

    def test_two_targets_do_not_swap_chains(self):
        frames = [(index * 0.1, [(100 + index, 50, 130 + index, 80),
                                 (400 - index, 50, 430 - index, 80)]) for index in range(20)]
        chains = chain_detections(frames, max_jump_px=20.0)
        self.assertEqual(len(chains), 2)
        starts = sorted(chain[0][1] for chain in chains)
        self.assertAlmostEqual(starts[0], 115.0)
        self.assertAlmostEqual(starts[1], 415.0)


class CsvModeTests(unittest.TestCase):
    def test_coasted_rows_are_excluded_from_the_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tracks.csv'
            track_csv(path, [{'timestamp': index * 0.1, 'track_id': 1, 'x1': 100 + index,
                              'x2': 130 + index, 'missed_frames': 1 if index % 2 else 0}
                             for index in range(40)])
            rows = read_tracks(path)
        self.assertEqual(len(rows['1']), 20)
        self.assertEqual([row['x1'] for row in rows['1']], [100.0 + 2 * step for step in range(20)])

    def test_a_noisy_column_is_measured_close_to_its_true_sigma(self):
        rng = np.random.default_rng(2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tracks.csv'
            track_csv(path, [{'timestamp': round(index * 0.1, 3), 'track_id': 1,
                              'x1': 100 + index + rng.normal(0, 2.0),
                              'x2': 130 + index + rng.normal(0, 2.0)}
                             for index in range(2000)])
            rows, pooled = measure([path], window=9, degree=2, min_samples=30, max_gap_s=0.5)
        # Two independent edges at sigma 2 put the center at sigma 2/sqrt(2).
        self.assertAlmostEqual(rows[0]['rms_px'], 2.0 / math.sqrt(2), delta=0.15)
        self.assertEqual(rows[0]['reference'], 'box_center')
        self.assertNotIn('water_contact', pooled)   # the column was empty, not zero


if __name__ == '__main__':
    unittest.main()
