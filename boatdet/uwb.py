"""Plane-constrained UWB trilateration from the raw `uwb/ranging` feed.

The rigs publish two things per tag: `ship/<id>/uwb/position/<tag>`, already
solved by the DWM firmware, and `ship/<id>/uwb/ranging/<tag>/<anchor>`, the
individual anchor ranges. This module solves the second, and the reason is
measured rather than aesthetic.

Each tag ranges to exactly FOUR anchors, so a free 3D solve has four ranges for
three unknowns and lets z float. Ship 1's four anchors (A02, A04, A08, A09) all
sit in a narrow y band, 21.69-28.85 m, with the tag at y~25.3, so its z is
nearly unobservable and the error leaks almost entirely into y: moving the
assumed plane from 1.2 m to 3.0 m moves y by 1.78 m and x by 0.011 m. The
firmware duly reports ship 1's tags at z = 2.94 m and 4.41 m -- impossible for a
hull-mounted tag -- and its y is wrong by 1.3-2.3 m. Holding z FIXED spends the
fourth range on the horizontal solution instead, which is the only place it is
worth anything. Ship 2's anchors span x instead, so at its operating
position the same error leaks into x (1.54 m for the same 1.8 m of plane), but
its z solves sanely and both methods agree there to ~0.2 m. Which axis a plane
error lands in depends on where the hull sits relative to its four anchors, so
it is a property of the run, not a constant of the rig.

Measured 2026-09-18, 25 s, both boats stationary; see docs for the z sweep.

PROVENANCE. `solve_plane_position` and `UWBBurstAssembler` are Bilal's, from
`uwb_tag_localizer_standalone.py` (sent 2026-09-18), which in turn copied them
from `pool_digital_twin` @ codex/cosmetic-up-camera-rays with these hashes:

    estimation/state_estimator/trilaterate.py
        eff758b27cd43b48636e80ccfceb603f2c8274e86f32be2b9ef1f658e6b178be
    estimation/state_estimator/uwb_bursts.py
        3954165cb3a6b8403bb0a274213d63864e04186922ddb7e87ef55f5e890a4301
    rl_controller/target_source.py
        608ec6733ac377f350c082e51d7be8b53e1b199d938f26ac1916aa3a0210fc73

That makes this a copy of a copy, which is the exact failure mode Bilal's own
docstring warns about: when this disagrees with the boat, suspect this file
first. `tests/test_uwb.py` re-runs his self-test against the YTU survey, which
catches a botched copy but NOT a stale one -- only a `pool_digital_twin`
checkout can catch that, and we have none.

No numpy: every step is 2x2 or reduces to it, so the analytic forms are exact.
"""
import bisect
import math
import sys
import time
from collections import deque

VENUES = {
    "ytu": {
        "anchors": {
            "1": [24.35, 9.74, 7.0], "2": [24.35, 21.69, 7.0],
            "3": [18.3, 14.0, 7.0], "4": [12.32, 21.74, 7.0],
            "5": [12.32, 9.49, 7.0], "6": [4.41, 0.0, 2.14],
            "7": [28.19, 0.0, 2.16], "8": [28.99, 28.85, 2.17],
            "9": [4.42, 28.85, 2.16],
        },
        "anchor_map": {"A01": "1", "A02": "2", "A03": "3", "A04": "4",
                       "A05": "5", "A06": "6", "A07": "7", "A08": "8",
                       "A09": "9"},
        "target_uwb": {"tag_label": "T04", "tag_z": 1.1, "expected_anchors": [],
                       "range_bias": {}, "sigma_range_m": None,
                       "calibration_id": "ytu-target-2026-09-16"},
    },
}
DEFAULTS = {"broker_host": "192.168.1.110", "broker_port": 1883,
            "topic": "buoy/uwb/ranging/#", "sigma_range_m": 0.10,
            "min_anchors": 3, "burst_timeout_s": 0.25, "max_cond": 100.0}
MAX_PENDING_TAGS = 32
MAX_RANGES_PER_BURST = 64
SOLVE_TOO_FEW = "too_few_anchors"; SOLVE_BAD_INPUT = "bad_input"
SOLVE_DUPLICATE = "duplicate_anchors"; SOLVE_IMPOSSIBLE = "impossible_geometry"
SOLVE_DEGENERATE = "degenerate_geometry"; SOLVE_NO_CONVERGE = "no_convergence"
SOLVE_BAD_COV = "bad_covariance"


def _sym_eig2(a, b, c):
    half_tr = 0.5 * (a + c)
    root = math.hypot(0.5 * (a - c), b)
    l1, l2 = half_tr + root, half_tr - root
    if b != 0.0:
        vx, vy = l1 - c, b
    elif a >= c:
        vx, vy = 1.0, 0.0
    else:
        vx, vy = 0.0, 1.0
    norm = math.hypot(vx, vy) or 1.0
    v1 = (vx / norm, vy / norm)
    return (l1, l2), (v1, (-v1[1], v1[0]))


def _solve_sym2(a, b, c, r0, r1):
    det = a * c - b * b
    scale = max(abs(a), abs(b), abs(c), 1e-300)
    if not math.isfinite(det) or abs(det) <= 1e-15 * scale * scale:
        return None
    return ((c * r0 - b * r1) / det, (a * r1 - b * r0) / det)


def _inv_sym2(a, b, c):
    det = a * c - b * b
    scale = max(abs(a), abs(b), abs(c), 1e-300)
    if not math.isfinite(det) or abs(det) <= 1e-15 * scale * scale:
        return None
    return (c / det, -b / det, a / det)


def _lstsq2(rows, rhs):
    n = len(rows)
    if n < 1:
        return None
    gxx = gxy = gyy = bx = by = 0.0
    for (mx, my), value in zip(rows, rhs):
        gxx += mx * mx; gxy += mx * my; gyy += my * my
        bx += mx * value; by += my * value
    (l1, l2), (v1, v2) = _sym_eig2(gxx, gxy, gyy)
    s1 = math.sqrt(l1) if l1 > 0.0 else 0.0
    s2 = math.sqrt(l2) if l2 > 0.0 else 0.0
    if s1 <= 0.0:
        return None
    cutoff = sys.float_info.epsilon * max(n, 2) * s1
    x = y = 0.0
    for (sv, lam, (vx, vy)) in ((s1, l1, v1), (s2, l2, v2)):
        if sv <= cutoff:
            continue
        coeff = (vx * bx + vy * by) / lam
        x += coeff * vx; y += coeff * vy
    return (x, y)


class PositionFix:
    __slots__ = ("tag", "x", "y", "z", "cov_xy", "anchor_ids", "ranges",
                 "residuals", "rmse", "cond", "rank", "t", "t_span",
                 "burst_id", "n_iter")

    def __init__(self, tag, x, y, z, cov_xy, anchor_ids, ranges, residuals,
                 rmse, cond, rank, t, t_span=0.0, burst_id=None, n_iter=0):
        self.tag = tag
        self.x, self.y, self.z = float(x), float(y), float(z)
        self.cov_xy = cov_xy
        self.anchor_ids = list(anchor_ids)
        self.ranges = [float(r) for r in ranges]
        self.residuals = [float(r) for r in residuals]
        self.rmse = float(rmse); self.cond = float(cond); self.rank = int(rank)
        self.t = float(t); self.t_span = float(t_span)
        self.burst_id = burst_id; self.n_iter = int(n_iter)

    @property
    def sigma_xy(self):
        return (math.sqrt(max(self.cov_xy[0], 0.0)),
                math.sqrt(max(self.cov_xy[2], 0.0)))


def _trilaterate_plane(anchors, ranges, z):
    if len(ranges) < 3:
        return None
    rho2 = []
    for (ax, ay, az), r in zip(anchors, ranges):
        dz = z - az
        rho2.append(max(r * r - dz * dz, 1e-6))
    x0, y0 = anchors[0][0], anchors[0][1]
    ref = x0 * x0 + y0 * y0
    rows, rhs = [], []
    for (ax, ay, _az), q in zip(anchors[1:], rho2[1:]):
        rows.append((2.0 * (ax - x0), 2.0 * (ay - y0)))
        rhs.append(ax * ax + ay * ay - ref - q + rho2[0])
    return _lstsq2(rows, rhs)


def solve_plane_position(anchors, ranges, z, sigma_range_m=0.10, *,
                         anchor_ids=None, tag=None, t=None, t_span=0.0,
                         burst_id=None, vertical_tol_m=None, min_anchors=3,
                         max_cond=1e8, max_iter=25, tol=1e-10):
    n = len(ranges)
    if len(anchors) != n or n == 0:
        return None, SOLVE_BAD_INPUT
    ids = ([str(a) for a in anchor_ids] if anchor_ids is not None
           else [str(i) for i in range(n)])
    if len(ids) != n:
        return None, SOLVE_BAD_INPUT
    z = float(z)
    if not math.isfinite(z):
        return None, SOLVE_BAD_INPUT
    sig = [float(sigma_range_m)] * n if not isinstance(
        sigma_range_m, (list, tuple)) else [float(s) for s in sigma_range_m]
    if len(sig) != n or any(not math.isfinite(s) or s <= 0.0 for s in sig):
        return None, SOLVE_BAD_INPUT
    good = [math.isfinite(r) and r > 0.0 and all(math.isfinite(v) for v in a)
            for a, r in zip(anchors, ranges)]
    if not any(good):
        return None, SOLVE_BAD_INPUT
    vtol = 3.0 * max(sig) if vertical_tol_m is None else float(vertical_tol_m)
    keep = [g and abs(z - a[2]) <= (r + vtol)
            for g, a, r in zip(good, anchors, ranges)]
    if sum(keep) < int(min_anchors):
        return None, (SOLVE_IMPOSSIBLE if sum(good) >= int(min_anchors)
                      else SOLVE_TOO_FEW)
    A = [(float(a[0]), float(a[1]), float(a[2]))
         for a, k in zip(anchors, keep) if k]
    r = [float(v) for v, k in zip(ranges, keep) if k]
    sig = [s for s, k in zip(sig, keep) if k]
    ids = [i for i, k in zip(ids, keep) if k]
    dz = [z - a[2] for a in A]
    if len({(round(a[0], 6), round(a[1], 6), round(a[2], 6))
            for a in A}) < int(min_anchors):
        return None, SOLVE_DUPLICATE
    seed = _trilaterate_plane(A, r, z)
    if seed is None or not all(math.isfinite(v) for v in seed):
        x = sum(a[0] for a in A) / len(A)
        y = sum(a[1] for a in A) / len(A)
    else:
        x, y = seed
    w = [1.0 / (s * s) for s in sig]
    n_iter = 0; converged = False
    for n_iter in range(1, int(max_iter) + 1):
        nxx = nxy = nyy = gx = gy = 0.0
        for (ax, ay, _az), meas, d_z, weight in zip(A, r, dz, w):
            ex, ey, ez = x - ax, y - ay, -d_z
            rho = math.sqrt(ex * ex + ey * ey + ez * ez)
            if rho < 1e-9:
                return None, SOLVE_DEGENERATE
            jx, jy = ex / rho, ey / rho
            res = meas - rho
            nxx += weight * jx * jx; nxy += weight * jx * jy
            nyy += weight * jy * jy; gx += weight * jx * res
            gy += weight * jy * res
        if not all(math.isfinite(v) for v in (nxx, nxy, nyy, gx, gy)):
            return None, SOLVE_NO_CONVERGE
        delta = _solve_sym2(nxx, nxy, nyy, gx, gy)
        if delta is None:
            return None, SOLVE_DEGENERATE
        if not all(math.isfinite(v) for v in delta):
            return None, SOLVE_NO_CONVERGE
        step = math.hypot(delta[0], delta[1])
        if step > 1e4:
            return None, SOLVE_NO_CONVERGE
        x, y = x + delta[0], y + delta[1]
        if step < tol:
            converged = True
            break
    if not converged:
        return None, SOLVE_NO_CONVERGE
    if not (math.isfinite(x) and math.isfinite(y)):
        return None, SOLVE_NO_CONVERGE
    nxx = nxy = nyy = 0.0
    residuals, sq = [], 0.0
    for (ax, ay, _az), meas, d_z, weight in zip(A, r, dz, w):
        ex, ey, ez = x - ax, y - ay, -d_z
        rho = math.sqrt(ex * ex + ey * ey + ez * ez)
        if rho < 1e-9:
            return None, SOLVE_DEGENERATE
        jx, jy = ex / rho, ey / rho
        res = meas - rho
        residuals.append(res); sq += res * res
        nxx += weight * jx * jx; nxy += weight * jx * jy
        nyy += weight * jy * jy
    sv, _ = _sym_eig2(nxx, nxy, nyy)
    sv = (abs(sv[0]), abs(sv[1]))
    rank = sum(1 for s in sv if s > sv[0] * 1e-12) if sv[0] > 0 else 0
    cond = (sv[0] / sv[1]) if sv[1] > 0 else math.inf
    if rank < 2 or not math.isfinite(cond) or cond > float(max_cond):
        return None, SOLVE_DEGENERATE
    cov = _inv_sym2(nxx, nxy, nyy)
    if cov is None or not all(math.isfinite(v) for v in cov):
        return None, SOLVE_BAD_COV
    ev, _ = _sym_eig2(cov[0], cov[1], cov[2])
    if min(ev) <= 0.0:
        return None, SOLVE_BAD_COV
    return PositionFix(tag=tag, x=x, y=y, z=z, cov_xy=cov, anchor_ids=ids,
                       ranges=r, residuals=residuals,
                       rmse=math.sqrt(sq / len(r)), cond=cond, rank=rank,
                       t=(0.0 if t is None else float(t)), t_span=t_span,
                       burst_id=burst_id, n_iter=n_iter), None


class _Pending:
    __slots__ = ("burst_id", "ranges", "t_first", "t_last", "arrival_first",
                 "arrival_last", "n_seen")

    def __init__(self, burst_id, t, arrival):
        self.burst_id = burst_id
        self.ranges = {}
        self.t_first = float(t); self.t_last = float(t)
        self.arrival_first = arrival; self.arrival_last = arrival
        self.n_seen = 0


class UWBBurstAssembler:
    def __init__(self, anchors, uwb_z, sigma_range_m, *, bias=None,
                 anchor_map=None, exclude_anchors=(), min_anchors=3,
                 timeout_s=DEFAULTS["burst_timeout_s"],
                 max_pending_tags=MAX_PENDING_TAGS, max_cond=None):
        self.anchors = {str(k): (float(v[0]), float(v[1]), float(v[2]))
                        for k, v in (anchors or {}).items()}
        self.uwb_z = float(uwb_z)
        self.sigma_range_m = float(sigma_range_m)
        self.bias = {str(k): float(v) for k, v in (bias or {}).items()}
        self.anchor_map = {str(k): str(v) for k, v in (anchor_map or {}).items()}
        self.exclude = {str(a) for a in (exclude_anchors or ())}
        self.min_anchors = max(3, int(min_anchors))
        self.timeout_s = float(timeout_s)
        self.max_pending_tags = int(max_pending_tags)
        self.max_cond = max_cond
        self._pending = {}
        self.reset_counters()

    def reset_counters(self):
        self.n_ranges_in = 0; self.n_ranges_valid = 0
        self.n_invalid_range = 0; self.n_unknown_anchor = 0
        self.n_excluded = 0; self.n_duplicate_anchor = 0
        self.n_overflow = 0; self.n_bursts = 0
        self.n_bursts_timeout = 0; self.n_bursts_incomplete = 0
        self.n_fixes = 0; self.n_evicted = 0; self.n_solve_fail = 0
        self.fail_reasons = {}
        self._ranges_per_burst = deque(maxlen=500)
        self._rmse = deque(maxlen=500); self._cond = deque(maxlen=500)
        self._span = deque(maxlen=500); self._latency = deque(maxlen=500)

    def add(self, tag, anchor, range_m, t, burst_id=None, arrival=None):
        tag = str(tag)
        self.n_ranges_in += 1
        out = self._flush_stale(t, except_tag=tag)
        raw = self._as_range(range_m)
        if raw is None:
            self.n_invalid_range += 1
            return out
        a = self.anchor_map.get(str(anchor), str(anchor))
        if a in self.exclude:
            self.n_excluded += 1
            return out
        if a not in self.anchors:
            self.n_unknown_anchor += 1
            return out
        corrected = raw - self.bias.get(a, 0.0)
        pend = self._pending.get(tag)
        if pend is not None and self._is_boundary(pend, a, t, burst_id):
            fix = self._complete(tag, timed_out=(t - pend.t_last) > self.timeout_s)
            if fix is not None:
                out.append(fix)
            pend = None
        if pend is None:
            self._evict_if_needed()
            pend = _Pending(burst_id, t, arrival)
            self._pending[tag] = pend
        elif pend.burst_id is None and burst_id is not None:
            pend.burst_id = burst_id
        pend.n_seen += 1
        if a in pend.ranges:
            self.n_duplicate_anchor += 1
            return out
        if len(pend.ranges) >= MAX_RANGES_PER_BURST:
            self.n_overflow += 1
            return out
        pend.ranges[a] = (float(t), float(corrected), float(raw))
        pend.t_last = max(pend.t_last, float(t))
        if arrival is not None:
            if pend.arrival_first is None:
                pend.arrival_first = arrival
            pend.arrival_last = arrival
        self.n_ranges_valid += 1
        return out

    def pending_anchors(self, tag):
        pend = self._pending.get(str(tag))
        return frozenset() if pend is None else frozenset(pend.ranges)

    def complete(self, tag):
        fix = self._complete(str(tag))
        return [] if fix is None else [fix]

    def flush(self, now=None, force=False):
        out = []
        for tag in list(self._pending):
            pend = self._pending.get(tag)
            if pend is None:
                continue
            stale = (now is not None
                     and (float(now) - pend.t_last) > self.timeout_s)
            if force or stale:
                fix = self._complete(tag, timed_out=stale and not force)
                if fix is not None:
                    out.append(fix)
        return out

    @staticmethod
    def _as_range(range_m):
        try:
            r = float(range_m)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(r) or r <= 0.0:
            return None
        return r

    def _is_boundary(self, pend, anchor, t, burst_id):
        if burst_id is not None and pend.burst_id is not None:
            return str(burst_id) != str(pend.burst_id)
        if (float(t) - pend.t_last) > self.timeout_s:
            return True
        return anchor in pend.ranges

    def _evict_if_needed(self):
        if len(self._pending) < self.max_pending_tags:
            return
        oldest = min(self._pending, key=lambda k: self._pending[k].t_last)
        del self._pending[oldest]
        self.n_evicted += 1

    def _complete(self, tag, timed_out=False):
        pend = self._pending.pop(tag, None)
        if pend is None:
            return None
        self.n_bursts += 1
        if timed_out:
            self.n_bursts_timeout += 1
        n = len(pend.ranges)
        self._ranges_per_burst.append(n)
        span = float(pend.t_last - pend.t_first)
        self._span.append(span)
        if n < self.min_anchors:
            self.n_bursts_incomplete += 1
            return None
        ids = sorted(pend.ranges)
        kw = {} if self.max_cond is None else {"max_cond": self.max_cond}
        fix, why = solve_plane_position(
            [self.anchors[a] for a in ids],
            [pend.ranges[a][1] for a in ids],
            self.uwb_z, self.sigma_range_m, anchor_ids=ids, tag=tag,
            t=pend.t_last, t_span=span, burst_id=pend.burst_id,
            min_anchors=self.min_anchors, **kw)
        if fix is None:
            self.n_solve_fail += 1
            self.fail_reasons[why] = self.fail_reasons.get(why, 0) + 1
            return None
        self.n_fixes += 1
        self._rmse.append(fix.rmse); self._cond.append(fix.cond)
        if pend.arrival_first is not None and pend.arrival_last is not None:
            self._latency.append(max(0.0, float(pend.arrival_last)
                                     - float(pend.arrival_first)))
        return fix

    def _flush_stale(self, now, except_tag=None):
        out = []
        for tag in list(self._pending):
            if tag == except_tag:
                continue
            pend = self._pending.get(tag)
            if pend is not None and (float(now) - pend.t_last) > self.timeout_s:
                fix = self._complete(tag, timed_out=True)
                if fix is not None:
                    out.append(fix)
        return out

    def stats(self):
        def _mean(dq, nd=3):
            return round(sum(dq) / len(dq), nd) if dq else None

        def _median(dq, nd=1):
            if not dq:
                return None
            ordered = sorted(dq); mid = len(ordered) // 2
            value = (ordered[mid] if len(ordered) % 2
                     else 0.5 * (ordered[mid - 1] + ordered[mid]))
            return round(value, nd)
        return {
            "n_ranges": int(self.n_ranges_in),
            "n_ranges_valid": int(self.n_ranges_valid),
            "n_invalid_range": int(self.n_invalid_range),
            "n_unknown_anchor": int(self.n_unknown_anchor),
            "n_excluded": int(self.n_excluded),
            "n_duplicate_anchor": int(self.n_duplicate_anchor),
            "n_bursts": int(self.n_bursts),
            "n_bursts_timeout": int(self.n_bursts_timeout),
            "n_bursts_incomplete": int(self.n_bursts_incomplete),
            "n_fixes": int(self.n_fixes),
            "n_solve_fail": int(self.n_solve_fail),
            "n_degenerate": int(self.fail_reasons.get("degenerate_geometry", 0)),
            "n_evicted": int(self.n_evicted),
            "fail_reasons": dict(self.fail_reasons) or None,
            "n_pending": len(self._pending),
            "ranges_per_burst": _mean(self._ranges_per_burst, 2),
            "burst_span_ms": (round(sum(self._span) / len(self._span) * 1e3, 1)
                              if self._span else None),
            "assembly_latency_ms": (
                round(sum(self._latency) / len(self._latency) * 1e3, 1)
                if self._latency else None),
            "solve_rmse_m": _mean(self._rmse),
            "solve_cond": _median(self._cond, 1),
        }


def tag_from_topic(topic):
    levels = [level for level in str(topic or "").split("/") if level]
    if not levels:
        return None

    def concrete(value):
        return None if value in (None, "#", "+") or "#" in value \
            or "+" in value else value
    if "ranging" in levels:
        index = levels.index("ranging") + 1
        return concrete(levels[index]) if index < len(levels) else None
    if len(levels) >= 2:
        return concrete(levels[-2])
    return None


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def parse_range(topic, payload, *, arrival=None, allow_arrival_time=False):
    if not isinstance(payload, dict):
        raise ValueError("range payload is not an object")
    parts = [p for p in str(topic or "").split("/") if p]
    topic_anchor = parts[-1] if parts else None
    topic_tag = parts[-2] if len(parts) >= 2 else None
    anchor = payload.get("anchor_label", payload.get("anchor"))
    anchor = str(anchor) if anchor not in (None, "") else None
    if anchor is None:
        anchor = topic_anchor
    elif topic_anchor is not None and topic_anchor != anchor:
        raise ValueError(
            f"topic anchor {topic_anchor!r} disagrees with the payload's "
            f"anchor label {anchor!r}")
    tag = payload.get("tag_label", payload.get("tag"))
    tag = str(tag) if tag not in (None, "") else None
    if tag is None:
        tag = topic_tag
    if not tag:
        raise ValueError("range carries no tag label, and the topic has none")
    if not anchor:
        raise ValueError("range carries no anchor label, and the topic has none")
    range_m = _finite(payload.get("range_m"))
    if range_m is None:
        raise ValueError(f"range_m is missing or not finite: "
                         f"{payload.get('range_m')!r}")
    burst = payload.get("timestamp_sent")
    burst_id = None if burst in (None, "") else str(burst)
    t_meas = _finite(payload.get("timestamp_received"))
    time_source = "publisher"
    if t_meas is None:
        if not allow_arrival_time:
            raise ValueError("timestamp_received is missing or not finite")
        t_meas = time.time() if arrival is None else float(arrival)
        time_source = "receiver-arrival"
    return {"tag": tag, "anchor": anchor, "range_m": range_m,
            "t_meas": t_meas, "burst_id": burst_id, "time_source": time_source,
            "arrival": time.time() if arrival is None else float(arrival)}


class Setup:
    def __init__(self, venue, anchors, anchor_map, tag_label, tag_z,
                 expected_anchors=(), range_bias=None, sigma_range_m=0.10,
                 calibration_id="", tag_source="venue"):
        self.venue = str(venue or "")
        self.anchors = {str(k): (float(v[0]), float(v[1]), float(v[2]))
                        for k, v in anchors.items()}
        self.anchor_map = dict(anchor_map or {})
        self.tag_label = str(tag_label); self.tag_source = str(tag_source)
        self.tag_z = float(tag_z)
        self.expected_anchors = frozenset(str(a) for a in expected_anchors)
        self.range_bias = {str(k): float(v) for k, v in (range_bias or {}).items()}
        self.sigma_range_m = float(sigma_range_m)
        self.calibration_id = str(calibration_id or "")




class TagSolver:
    """One tag's raw ranges in, solved plane fixes out. No transport, no threads.

    The MQTT half of Bilal's `TargetSource` is deliberately not here: the
    producer owns the client, so this stays replayable against a JSONL capture.
    A malformed message is COUNTED and dropped rather than raised, because a
    real capture has them and one bad range must not end a run.
    """

    def __init__(self, setup, *, min_anchors=3,
                 timeout_s=DEFAULTS["burst_timeout_s"],
                 max_cond=DEFAULTS["max_cond"], allow_arrival_time=False):
        self.setup = setup
        self._allow_arrival = bool(allow_arrival_time)
        self.assembler = UWBBurstAssembler(
            setup.anchors, setup.tag_z, setup.sigma_range_m,
            bias=setup.range_bias, anchor_map=setup.anchor_map,
            min_anchors=int(min_anchors), timeout_s=float(timeout_s),
            max_cond=float(max_cond))
        self.counters = {"messages": 0, "retained": 0, "malformed": 0,
                         "other_tag": 0}

    def accept(self, topic, payload, *, retained=False, arrival=None):
        """Offer one `.../uwb/ranging/...` message. Returns completed fixes."""
        self.counters["messages"] += 1
        if retained:
            # The broker's memory of an old measurement, delivered at subscribe
            # time with nothing on it to say so: fresh by arrival, oldest
            # possible by measurement time.
            self.counters["retained"] += 1
            return []
        try:
            row = parse_range(topic, payload, arrival=arrival,
                              allow_arrival_time=self._allow_arrival)
        except ValueError:
            self.counters["malformed"] += 1
            return []
        if row["tag"] != self.setup.tag_label:
            self.counters["other_tag"] += 1
            return []
        fixes = self.assembler.add(row["tag"], row["anchor"], row["range_m"],
                                   row["t_meas"], burst_id=row["burst_id"],
                                   arrival=row["arrival"])
        expected = self.setup.expected_anchors
        if expected and expected <= self.assembler.pending_anchors(row["tag"]):
            fixes = list(fixes) + self.assembler.complete(row["tag"])
        return list(fixes)

    def flush(self, now=None, force=False):
        return self.assembler.flush(now=now, force=force)

    def stats(self):
        out = dict(self.counters)
        out.update(self.assembler.stats())
        return out


class HullFix:
    """Both of one hull's tags solved at one instant.

    `bearing_rad` is the direction from the stern tag to the bow tag in the
    anchor frame, NOT the hull heading: the rotation between the tag baseline
    and the bow is a mounting constant nobody has surveyed. `HullTracker`
    applies it.
    """

    __slots__ = ("ship", "t", "x", "y", "bearing_rad", "baseline_m", "dt_pair",
                 "bow", "stern")

    def __init__(self, ship, t, x, y, bearing_rad, baseline_m, dt_pair,
                 bow, stern):
        self.ship = ship
        self.t = float(t)
        self.x, self.y = float(x), float(y)
        self.bearing_rad = bearing_rad
        self.baseline_m = baseline_m
        self.dt_pair = dt_pair
        self.bow, self.stern = bow, stern


class HullTracker:
    """Pairs a hull's two tags in time and reports one position and bearing.

    The tags range independently at ~10 Hz, so their bursts never share a
    timestamp. Each bow fix is matched to the NEAREST stern fix in measurement
    time and rejected beyond `max_pair_dt_s`, because a pair straddling real
    motion turns hull translation into a false rotation.

    Position is the midpoint of the two tags, which is closer to the hull
    centre than either tag and averages down the per-tag noise.
    """

    def __init__(self, ship, bow_tag, stern_tag, *, max_pair_dt_s=0.06,
                 history=400):
        self.ship = int(ship)
        self.bow_tag, self.stern_tag = str(bow_tag), str(stern_tag)
        self.max_pair_dt_s = float(max_pair_dt_s)
        self._fixes = {self.bow_tag: deque(maxlen=history),
                       self.stern_tag: deque(maxlen=history)}
        self._times = {self.bow_tag: deque(maxlen=history),
                       self.stern_tag: deque(maxlen=history)}
        self.n_paired = 0
        self.n_unpaired = 0

    def add(self, fix):
        """Feed one tag fix. Returns a HullFix when it completes a pair."""
        tag = str(fix.tag)
        if tag not in self._fixes:
            return None
        self._fixes[tag].append(fix)
        self._times[tag].append(fix.t)
        other = self.stern_tag if tag == self.bow_tag else self.bow_tag
        mate = self._nearest(other, fix.t)
        if mate is None:
            self.n_unpaired += 1
            return None
        bow, stern = ((fix, mate) if tag == self.bow_tag else (mate, fix))
        self.n_paired += 1
        dx, dy = bow.x - stern.x, bow.y - stern.y
        return HullFix(
            ship=self.ship, t=max(bow.t, stern.t),
            x=0.5 * (bow.x + stern.x), y=0.5 * (bow.y + stern.y),
            bearing_rad=math.atan2(dy, dx), baseline_m=math.hypot(dx, dy),
            dt_pair=abs(bow.t - stern.t), bow=bow, stern=stern)

    def _nearest(self, tag, t):
        times = self._times[tag]
        if not times:
            return None
        i = bisect.bisect_left(times, t)
        best, best_dt = None, None
        for j in (i - 1, i):
            if 0 <= j < len(times):
                dt = abs(times[j] - t)
                if best_dt is None or dt < best_dt:
                    best, best_dt = j, dt
        if best is None or best_dt > self.max_pair_dt_s:
            return None
        return self._fixes[tag][best]


#: What the YTU rigs actually put on the wire, measured 2026-09-18 over 25 s with
#: both boats stationary. Bilal's `VENUES` survey above declares a `target_uwb`
#: block for T04 only, at z = 1.1 m; every tag's ranges prefer 1.2-1.5 m, so the
#: plane is recorded per tag here rather than inherited.
#:
#: `bow`/`stern` are UNSURVEYED. Which tag is forward, and the angle between the
#: tag baseline and the bow, is a mounting fact nobody has measured -- so the
#: bearing this module derives is the baseline's, and `heading_offset_deg` below
#: is a placeholder, not a calibration. Ship 2's baseline gives 5.7 deg of
#: heading noise (0.496 m +/- 0.032 m apart); ship 1's gives 22.7 deg, because
#: its weak y axis dominates a 0.776 m baseline. Only ship 2's is a heading
#: source -- which is the boat that needs one, since the pursuer reads the
#: target's px/py and never its heading.
RIG_TAGS = {
    1: {"tags": {"T01": 1.2, "T02": 1.4}, "bow": "T01", "stern": "T02",
        "anchors": ("A02", "A04", "A08", "A09"), "heading_offset_deg": None,
        "heading_sd_deg": 22.7},
    2: {"tags": {"T03": 1.2, "T04": 1.5}, "bow": "T03", "stern": "T04",
        "anchors": ("A01", "A03", "A07", "A08"), "heading_offset_deg": None,
        "heading_sd_deg": 5.7},
}


def ytu_setup(tag, tag_z=None, sigma_range_m=None):
    """A `Setup` for one YTU tag, with the measured plane unless overridden."""
    venue = VENUES["ytu"]
    if tag_z is None:
        for spec in RIG_TAGS.values():
            if tag in spec["tags"]:
                tag_z = spec["tags"][tag]
                break
    if tag_z is None:
        raise ValueError(
            f"tag {tag!r} has no measured plane height and none was given. The "
            f"height is a plane CONSTRAINT: a wrong one moves the fix in x and "
            f"y, so it is not defaulted.")
    return Setup("ytu", venue["anchors"], venue["anchor_map"], tag,
                 float(tag_z),
                 sigma_range_m=(DEFAULTS["sigma_range_m"]
                                if sigma_range_m is None else sigma_range_m))


def plane_sensitivity(fix, setup, delta_m=0.1):
    """Metres the horizontal fix moves per metre of error in the assumed plane.

    This is what makes the height a CONSTRAINT rather than a label: measured on
    the rigs it runs 0.77-1.12, so a plane wrong by 0.3 m puts the position out
    by roughly 0.3 m. Multiply by the plane's own uncertainty to get metres.

    It is NOT a geometry-quality gate, and it was tried as one and failed: the
    degenerate configuration described in `rmse_gate_note` below scored 0.77,
    LOWER than the healthy sets at 0.85-1.12. Use RMSE for that.

    Returns `None` if the perturbed solve fails.
    """
    moved, _ = solve_plane_position(
        [setup.anchors[a] for a in fix.anchor_ids], fix.ranges,
        fix.z + float(delta_m), setup.sigma_range_m,
        anchor_ids=fix.anchor_ids, tag=fix.tag, max_cond=1e12)
    if moved is None:
        return None
    return math.hypot(moved.x - fix.x, moved.y - fix.y) / abs(float(delta_m))


#: WHY A FIX IS GATED ON RMSE AND NOT ON `cond`.
#:
#: The firmware chooses which anchors a tag ranges to, and it CHANGES THAT SET
#: AS THE BOAT MOVES -- mid-run, with nothing announcing it. Measured
#: 2026-09-18: ship 2's tags ranged to A01, A03, A07, A08 (two of them low, at
#: z~2.16 m) and solved to 0.035 m of RMSE; after the boats moved a few metres
#: the same tags were ranging to A01, A02, A03, A05, all four on the ceiling at
#: z = 7.00 m with one nearly overhead, and RMSE became 0.42-0.55 m. Dropping
#: any single anchor left 0.18-0.37 m, so it is not one bad anchor; the set is
#: mutually inconsistent, and positions from different anchor TRIPLES of it
#: disagree by up to 12 m. The tag baseline, which is rigid, apparently grew
#: from 0.496 m to 0.726 m.
#:
#: `cond` reported 1.5 throughout that -- an excellent number. It is the
#: conditioning of the horizontal information matrix for a KNOWN height, so it
#: cannot see this at all. RMSE separates the two states cleanly, with nothing
#: in between, which is the only reason it is the gate.
#:
#: Root cause is NOT established. Candidates are the survey of A05 (unused by
#: the earlier capture, so untested by it), the near-overhead geometry of A01,
#: and NLOS on those steep paths. It wants a rig-side answer, not a solver one.
RMSE_GATE_M = 0.25


def anchor_heights(fix, setup):
    """The surveyed heights of the anchors that actually solved this burst."""
    return [setup.anchors[a][2] for a in fix.anchor_ids]
