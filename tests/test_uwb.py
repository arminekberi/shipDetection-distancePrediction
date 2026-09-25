"""Bilal's self-test, re-run against `boatdet.uwb`.

`boatdet/uwb.py` carries COPIES of a production solver two removes from its
original, so the thing most likely to be wrong with it is the copying. These
push synthetic bursts of known geometry through the real entry point --
`TagSolver.accept` with wire-shaped payloads -- so the parser, the anchor map,
the burst rules and the solve are all on the path.

They catch a BOTCHED copy, not a STALE one. Only a `pool_digital_twin` checkout
can tell you the upstream has moved, and there is none here.
"""
import math
import unittest

from boatdet.uwb import RIG_TAGS, HullTracker, TagSolver, VENUES, ytu_setup

ANCHORS = VENUES['ytu']['anchors']
AMAP = VENUES['ytu']['anchor_map']
WIRE = {v: k for k, v in AMAP.items()}          # survey id -> wire id
TAG_Z = 1.1


def burst(truth_xy, survey_ids, burst_id, t0, tag='T04', tag_z=TAG_Z):
    """Exact ranges from `truth_xy` to the named anchors, as wire messages."""
    out = []
    for i, key in enumerate(survey_ids):
        ax, ay, az = ANCHORS[key]
        r = math.sqrt((truth_xy[0] - ax) ** 2 + (truth_xy[1] - ay) ** 2
                      + (tag_z - az) ** 2)
        wire = WIRE.get(key, key)
        out.append((f'ship/2/uwb/ranging/{tag}/{wire}',
                    {'tag_label': tag, 'anchor_label': wire, 'range_m': r,
                     'timestamp_sent': str(burst_id),
                     'timestamp_received': t0 + 0.002 * i}))
    return out


def solve(messages, tag='T04', tag_z=TAG_Z):
    solver = TagSolver(ytu_setup(tag, tag_z=tag_z))
    fixes = []
    for topic, payload in messages:
        fixes.extend(solver.accept(topic, payload,
                                   arrival=payload['timestamp_received']))
    fixes.extend(solver.flush(force=True))
    return fixes, solver


class SolveTests(unittest.TestCase):
    TRUTH = (16.0, 15.0)

    def test_clean_five_anchor_burst_returns_the_truth(self):
        fixes, _ = solve(burst(self.TRUTH, ['1', '2', '3', '4', '5'], 1000, 100.0))
        self.assertEqual(len(fixes), 1)
        self.assertAlmostEqual(fixes[0].x, self.TRUTH[0], places=6)
        self.assertAlmostEqual(fixes[0].y, self.TRUTH[1], places=6)

    def test_residuals_are_zero_on_exact_ranges(self):
        fixes, _ = solve(burst(self.TRUTH, ['1', '2', '3', '4', '5'], 1001, 200.0))
        self.assertLess(fixes[0].rmse, 1e-9)

    def test_three_anchors_are_enough_on_the_plane(self):
        fixes, _ = solve(burst(self.TRUTH, ['1', '4', '5'], 1002, 300.0))
        self.assertEqual(len(fixes), 1)
        self.assertLess(math.hypot(fixes[0].x - self.TRUTH[0],
                                   fixes[0].y - self.TRUTH[1]), 1e-5)

    def test_two_anchors_produce_no_fix(self):
        """Dropped, never guessed: an underdetermined burst has no position."""
        fixes, _ = solve(burst(self.TRUTH, ['1', '4'], 1003, 400.0))
        self.assertEqual(fixes, [])

    def test_a_failed_exchange_is_not_a_measurement(self):
        """The DW1000 reports a failed two-way ranging as a large negative."""
        msgs = burst(self.TRUTH, ['1', '2', '3', '4', '5'], 1004, 500.0)
        msgs[0][1]['range_m'] = -21474.836
        fixes, _ = solve(msgs)
        self.assertEqual(len(fixes), 1)
        self.assertEqual(len(fixes[0].anchor_ids), 4)
        self.assertLess(math.hypot(fixes[0].x - self.TRUTH[0],
                                   fixes[0].y - self.TRUTH[1]), 1e-6)

    def test_both_rig_anchor_sets_are_well_conditioned_at_the_centre(self):
        for ship, spec in RIG_TAGS.items():
            ids = [AMAP[a] for a in spec['anchors']]
            fixes, _ = solve(burst(self.TRUTH, ids, 2000 + ship, 600.0))
            self.assertEqual(len(fixes), 1, f'ship {ship} produced no fix')
            self.assertLess(fixes[0].cond, 10.0, f'ship {ship} ill-conditioned')


class PlaneTests(unittest.TestCase):
    def test_a_wrong_plane_moves_the_fix_horizontally(self):
        """Why the height is never defaulted: it is a constraint, not a label.

        Ship 1's anchors all sit in a narrow y band, so its z error leaks into
        y; ship 2's span x, so its leaks into x. Each must move in its own axis
        and stay put in the other.

        The leak axis depends on where the hull sits relative to its anchors,
        not on the anchor spread alone, so each ship is checked AT ITS MEASURED
        POSITION (2026-09-18). At the arena centre ship 2's x leak falls to
        0.78 m and at (16, 25) it nearly vanishes -- a real property of the
        geometry, not a weaker version of the same effect.
        """
        for ship, truth, leaks, holds in ((1, (16.7, 25.3), 'y', 'x'),
                                          (2, (25.9, 12.6), 'x', 'y')):
            ids = [AMAP[a] for a in RIG_TAGS[ship]['anchors']]
            low, _ = solve(burst(truth, ids, 3000 + ship, 700.0, tag_z=1.2),
                           tag_z=1.2)
            high, _ = solve(burst(truth, ids, 3100 + ship, 800.0, tag_z=1.2),
                            tag_z=3.0)
            self.assertEqual(len(low), 1)
            self.assertEqual(len(high), 1)
            moved = {'x': abs(high[0].x - low[0].x),
                     'y': abs(high[0].y - low[0].y)}
            self.assertGreater(moved[leaks], 0.5,
                               f'ship {ship}: a 1.8 m plane error should move {leaks}')
            self.assertLess(moved[holds], moved[leaks],
                            f'ship {ship}: {holds} should move less than {leaks}')

    def test_an_unknown_tag_refuses_to_guess_a_height(self):
        with self.assertRaises(ValueError):
            ytu_setup('T99')


class HullTrackerTests(unittest.TestCase):
    def test_a_pair_gives_the_midpoint_and_the_baseline_bearing(self):
        tracker = HullTracker(2, 'T03', 'T04')
        bow, _ = solve(burst((16.5, 15.0), ['1', '3', '7', '8'], 4000, 900.0,
                             tag='T03', tag_z=1.2), tag='T03', tag_z=1.2)
        stern, _ = solve(burst((15.5, 15.0), ['1', '3', '7', '8'], 4001, 900.01,
                               tag='T04', tag_z=1.2), tag='T04', tag_z=1.2)
        self.assertIsNone(tracker.add(stern[0]))     # nothing to pair with yet
        hull = tracker.add(bow[0])
        self.assertIsNotNone(hull)
        self.assertAlmostEqual(hull.x, 16.0, places=3)
        self.assertAlmostEqual(hull.y, 15.0, places=3)
        self.assertAlmostEqual(hull.baseline_m, 1.0, places=3)
        self.assertAlmostEqual(math.degrees(hull.bearing_rad), 0.0, places=2)

    def test_a_pair_straddling_a_gap_is_refused(self):
        """A pair spanning real motion turns translation into false rotation."""
        tracker = HullTracker(2, 'T03', 'T04', max_pair_dt_s=0.06)
        bow, _ = solve(burst((16.5, 15.0), ['1', '3', '7', '8'], 4002, 1000.0,
                             tag='T03', tag_z=1.2), tag='T03', tag_z=1.2)
        stern, _ = solve(burst((15.5, 15.0), ['1', '3', '7', '8'], 4003, 1005.0,
                               tag='T04', tag_z=1.2), tag='T04', tag_z=1.2)
        tracker.add(stern[0])
        self.assertIsNone(tracker.add(bow[0]))
        self.assertEqual(tracker.n_paired, 0)


if __name__ == '__main__':
    unittest.main()


class SensitivityTests(unittest.TestCase):
    def test_the_plane_moves_the_fix_about_one_for_one(self):
        """Measured 0.77-1.12 on the rigs, so a 0.3 m plane error costs ~0.3 m."""
        from boatdet.uwb import plane_sensitivity
        setup = ytu_setup('T03', tag_z=1.2)
        fixes, _ = solve(burst((25.9, 12.6), ['1', '3', '7', '8'], 5000, 1100.0,
                               tag='T03', tag_z=1.2), tag='T03', tag_z=1.2)
        sensitivity = plane_sensitivity(fixes[0], setup)
        self.assertIsNotNone(sensitivity)
        self.assertGreater(sensitivity, 0.5)
        self.assertLess(sensitivity, 2.0)
