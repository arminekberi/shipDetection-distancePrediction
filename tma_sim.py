"""Measure the bearings-only estimator against known truth, before any video.

Detection and calibration errors cannot be separated from estimator errors on a
recording, so the estimator is checked here first, where the target state is
known exactly and the only error injected is bearing noise. The scenarios differ
in own-ship geometry alone: what they measure is how much range information the
manoeuvre carries, which is the property that decides whether the method works
on the rig at all.

    venv/bin/python tma_sim.py --scenario all
"""
import argparse
import csv
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from boatdet.config import TmaConfig
from boatdet.tma import (BearingObservation, OwnShipState, RangeHypothesisTma,
                         compass_from_course, course_from_compass, wrap_angle)


@dataclass(frozen=True)
class Leg:
    """Own-ship course and speed held until `until_s`."""
    until_s: float
    course_deg: float
    speed_mps: float


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    legs: tuple
    target_start: tuple          # (x, y) meters
    target_course_deg: float
    target_speed_mps: float
    rate_hz: float = None        # measurement rate this scenario is meant to run at
    duration_s: float = None
    overrides: dict = None       # TmaConfig fields this scale needs; defaults are basin-scale

    def configured(self, config):
        """The estimator configuration for this scenario's scale."""
        return replace(config, **self.overrides) if self.overrides else config


# The rig records one scene only: a covered test basin, tens of meters across, at
# roughly 4 fps. The defaults in boatdet.config are scaled for it. The open-water
# scenarios carry their own overrides so the method stays exercised at sea scale,
# where it is much harder: the same manoeuvre buys a hundredth of the parallax.
OPEN_WATER = {'min_range_m': 30.0, 'max_range_m': 3000.0, 'max_target_speed_mps': 15.0,
              'speed_prior_slack_mps': 3.0, 'maneuver_window_s': 120.0,
              'weight_forget_s': 60.0, 'max_dt_s': 5.0, 'pixel_sigma_px': 4.0}

SCENARIOS = {
    'basin-straight': Scenario(
        'basin-straight', 'basin: own boat runs the tank on one heading, range not observable',
        (Leg(60.0, 90.0, 0.8),), (55.0, 6.0), 90.0, 0.4, rate_hz=4.0, duration_s=60.0),
    'basin-weave': Scenario(
        'basin-weave', 'basin: an 8 degree weave, the largest manoeuvre the tank width allows',
        (Leg(15.0, 90.0, 0.8), Leg(30.0, 82.0, 0.8), Leg(45.0, 98.0, 0.8), Leg(60.0, 82.0, 0.8)),
        (55.0, 6.0), 90.0, 0.4, rate_hz=4.0, duration_s=60.0),
    'basin-crossing': Scenario(
        'basin-crossing', 'basin: target crossing ahead, own boat straight, still ambiguous',
        (Leg(60.0, 90.0, 0.8),), (45.0, -5.0), 0.0, 0.25, rate_hz=4.0, duration_s=60.0),
    'straight': Scenario(
        'straight', 'open water: own ship holds one course, range is not observable',
        (Leg(240.0, 90.0, 5.0),), (600.0, 900.0), 225.0, 4.0, overrides=OPEN_WATER),
    'maneuver': Scenario(
        'maneuver', 'open water: own ship changes course once, across the line of sight',
        (Leg(90.0, 90.0, 5.0), Leg(240.0, 20.0, 5.0)), (600.0, 900.0), 225.0, 4.0,
        overrides=OPEN_WATER),
    'zigzag': Scenario(
        'zigzag', 'open water: repeated leg changes, parallax accumulates continuously',
        (Leg(60.0, 70.0, 5.0), Leg(120.0, 110.0, 5.0),
         Leg(180.0, 70.0, 5.0), Leg(240.0, 110.0, 5.0)), (600.0, 900.0), 225.0, 4.0,
        overrides=OPEN_WATER),
    'stationary-target': Scenario(
        'stationary-target', 'open water: stopped target, straight own ship, still ambiguous',
        (Leg(240.0, 90.0, 5.0),), (400.0, 700.0), 0.0, 0.0, overrides=OPEN_WATER),
    'closing': Scenario(
        'closing', 'open water: target closing head-on while the own ship doglegs',
        (Leg(80.0, 45.0, 6.0), Leg(240.0, 0.0, 6.0)), (1200.0, 1200.0), 225.0, 6.0,
        overrides=OPEN_WATER),
}


def own_ship_at(scenario, timestamp):
    """Own-ship state produced by integrating the legs up to `timestamp`."""
    x = y = 0.0
    previous = 0.0
    course_deg, speed = scenario.legs[0].course_deg, scenario.legs[0].speed_mps
    for leg in scenario.legs:
        span = min(timestamp, leg.until_s) - previous
        course_deg, speed = leg.course_deg, leg.speed_mps
        if span > 0:
            course = course_from_compass(leg.course_deg)
            x += speed * span * math.cos(course)
            y += speed * span * math.sin(course)
            previous = min(timestamp, leg.until_s)
        if timestamp <= leg.until_s:
            break
    course = course_from_compass(course_deg)
    return OwnShipState(timestamp=timestamp, x_m=x, y_m=y, heading_rad=course,
                        speed_mps=speed, course_rad=course)


def target_at(scenario, timestamp):
    """True target position at `timestamp`; the target runs a constant course."""
    course = course_from_compass(scenario.target_course_deg)
    return (scenario.target_start[0] + scenario.target_speed_mps * timestamp * math.cos(course),
            scenario.target_start[1] + scenario.target_speed_mps * timestamp * math.sin(course))


def run(scenario, config, noise_deg, rate_hz, duration_s, seed):
    """Feed noisy bearings to the estimator; one row per update."""
    rng = np.random.default_rng(seed)
    sigma = math.radians(noise_deg)
    rate_hz = scenario.rate_hz or rate_hz
    duration_s = scenario.duration_s or duration_s
    estimator = RangeHypothesisTma(scenario.configured(config))
    rows = []
    for step in range(int(duration_s * rate_hz) + 1):
        timestamp = step / rate_hz
        own = own_ship_at(scenario, timestamp)
        target = target_at(scenario, timestamp)
        truth_bearing = math.atan2(target[1] - own.y_m, target[0] - own.x_m)
        observation = BearingObservation(timestamp=timestamp,
                                         bearing_rad=wrap_angle(truth_bearing + rng.normal(0.0, sigma)),
                                         sigma_rad=sigma, sensor_x_m=own.x_m, sensor_y_m=own.y_m)
        estimate = estimator.update(observation)
        if estimate is None:
            continue
        true_range = math.hypot(target[0] - own.x_m, target[1] - own.y_m)
        rows.append({
            'scenario': scenario.name, 't_s': round(timestamp, 2),
            'true_range_m': round(true_range, 1), 'range_m': round(estimate.range_m, 1),
            'range_error_m': round(estimate.range_m - true_range, 1),
            'range_low_m': round(estimate.range_low_m, 1), 'range_high_m': round(estimate.range_high_m, 1),
            'range_covered': bool(estimate.range_low_m <= true_range <= estimate.range_high_m),
            'true_speed_mps': scenario.target_speed_mps, 'speed_mps': round(estimate.speed_mps, 2),
            'speed_error_mps': round(estimate.speed_mps - scenario.target_speed_mps, 2),
            'true_course_deg': scenario.target_course_deg,
            'course_deg': round(estimate.course_deg, 1),
            'course_error_deg': round(math.degrees(wrap_angle(
                estimate.course_rad - course_from_compass(scenario.target_course_deg))), 1),
            'position_sigma_m': round(estimate.position_sigma_m, 1),
            'effective_hypotheses': round(estimate.effective_hypotheses, 2),
            'parallax_ratio': round(estimate.parallax_ratio, 5),
            'status': estimate.status})
    return rows


def summarize(rows):
    """Final state of a run plus the coverage of the reported interval over it."""
    last = rows[-1]
    converged = [row for row in rows if row['status'] == 'converged']
    covered = sum(row['range_covered'] for row in rows) / len(rows)
    first_converged = converged[0]['t_s'] if converged else None
    return {'scenario': last['scenario'], 'final_status': last['status'],
            'time_to_converged_s': first_converged,
            'final_range_error_m': last['range_error_m'],
            'final_range_error_pct': round(100 * last['range_error_m'] / max(last['true_range_m'], 1e-6), 1),
            'final_speed_error_mps': last['speed_error_mps'],
            'final_course_error_deg': last['course_error_deg'],
            'final_interval_m': f"{last['range_low_m']:.0f}-{last['range_high_m']:.0f}",
            'interval_covers_truth_frac': round(covered, 3),
            'final_effective_hypotheses': last['effective_hypotheses']}


def print_table(rows, every_s):
    columns = ('t_s', 'true_range_m', 'range_m', 'range_low_m', 'range_high_m',
               'speed_mps', 'course_deg', 'effective_hypotheses', 'parallax_ratio', 'status')
    print('  '.join(f'{name:>12}' for name in columns))
    for row in rows:
        if row['t_s'] % every_s > 1e-9:
            continue
        print('  '.join(f'{row[name]!s:>12}' for name in columns))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scenario', default='all', choices=('all', *SCENARIOS),
                        help='own-ship and target geometry to measure')
    parser.add_argument('--noise-deg', type=float, default=0.5, help='bearing noise, one sigma')
    parser.add_argument('--rate-hz', type=float, default=1.0,
                        help='bearings per second, for scenarios that do not set their own')
    parser.add_argument('--duration-s', type=float, default=240.0,
                        help='length of the run, for scenarios that do not set their own')
    parser.add_argument('--seed', type=int, default=0, help='noise seed')
    parser.add_argument('--print-every-s', type=float, default=10.0, help='row interval in the table')
    parser.add_argument('--csv', help='write every row of every scenario here')
    parser.add_argument('--hypotheses', type=int, default=TmaConfig.hypotheses)
    parser.add_argument('--min-range-m', type=float, default=TmaConfig.min_range_m)
    parser.add_argument('--max-range-m', type=float, default=TmaConfig.max_range_m)
    args = parser.parse_args()

    config = TmaConfig(hypotheses=args.hypotheses, min_range_m=args.min_range_m,
                       max_range_m=args.max_range_m)
    names = list(SCENARIOS) if args.scenario == 'all' else [args.scenario]
    all_rows, summaries = [], []
    for name in names:
        scenario = SCENARIOS[name]
        rows = run(scenario, config, args.noise_deg, args.rate_hz, args.duration_s, args.seed)
        all_rows += rows
        summaries.append(summarize(rows))
        print(f'\n=== {scenario.name}: {scenario.description} ===')
        print_table(rows, args.print_every_s)

    print('\n=== summary ===')
    header = list(summaries[0])
    print('  '.join(f'{name:>26}' if name == 'scenario' else f'{name:>22}' for name in header))
    for summary in summaries:
        print('  '.join(f'{summary[name]!s:>26}' if name == 'scenario' else f'{summary[name]!s:>22}'
                        for name in header))
    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f'\nwrote {len(all_rows)} rows to {path}')


if __name__ == '__main__':
    main()
