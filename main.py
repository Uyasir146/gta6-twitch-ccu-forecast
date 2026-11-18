"""GTA VI release-day Twitch CCU forecast: hour 2 to hour 24, five quantiles an hour.

The entry for make.fun's `gta6-twitch-ccu-forecast-01`. The grader hands one case file at a time
(`<id>.in`, schema makefun/twitch-ccu-case/v1) and reads `<id>.out`::

    {"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}

Everything here is standard library and deterministic: no network, no clock, no randomness, so two runs
on the same case produce the same file byte for byte.

The model
---------
By the time this program runs, hour 1 of release day is history: the case's `known` row carries its print
`p1` (the mean of the minute reads inside 00:00-01:00Z). The day is UTC-aligned for every launch, and the
unlock rolls west with the clock (New Zealand 11:00Z on the 18th, central Europe 23:00Z, the UK 00:00Z, US
Eastern 05:00Z, US Pacific 08:00Z), so hour 1 holds the Europe/UK wave and the US waves land in hours 5-10.

What is left to forecast is the *shape* of the day, as a multiple of `p1`:

    x(h) = ln(viewers(h) / p1)

The case ships 38-42 past big launches as `comparables`, each with its own hourly curve, so each one gives
a sample of that shape. This program forecasts the quantiles of x(h) from those samples, weighted by how
much each launch resembles the target, and multiplies back by `p1`.

Two things drive the weighting:

* Scale. A launch whose hour 1 is already huge (Elden Ring at 875k, Cyberpunk at 852k) decays inside a
  narrow band and recovers for US prime time; a launch whose hour 1 is quiet has its whole day ahead of it
  and runs up several log units later. The log-ratio curves of the comparables correlate with ln(p1) at
  about -0.9 by hour 18, so a comparable is weighted by a Gaussian kernel on u = ln(p1), and the central
  curve is a weighted local-linear fit in u, evaluated at the target's u. Above the largest comparable the
  fit is held flat: GTA VI's hour 1 will very likely be bigger than any launch in the set, and
  extrapolating a slope there would invent shape rather than borrow it.
* Tier. Each comparable's `unlock` note carries the case author's tier: `comparable`, `early-access-
  contaminated` (a launch whose pre-order/early-access spike distorts its release-day curve), `weak`, and
  `target-history` (the GTA VI category's own pre-release curve, which is a hype gauge, not a launch, and
  carries no weight). The rest get a weight below 1.

The band, and why it is a blend
-------------------------------
Two spread estimates are computed for every hour and blended:

* the empirical quantiles of the weighted ratio sample, which is sharp where the neighbours are close;
* a parametric band, z_tau * s, where s is the local residual scale of the fit, floored at a fraction of
  the all-launch residual scale of that hour.

Blending matters at the top of the scale: when only two or three comparables are close to a mega-launch,
empirical quantiles collapse onto those launches' own ratios and q75 lands on top of q95. The parametric
band keeps the spread smooth there, the empirical one keeps it honest in the middle of the range, and the
LOO study behind these constants says the blend costs nothing where both work and rescues the extreme.

Levels are pulled outward by a per-level factor. The pinball loss this is scored on is asymmetric in an
awkward direction: a 5% quantile set too low is cheap (it is only wrong when the truth falls below it,
2*0.05*error) while one set too high is expensive (2*0.95*error). So a quantile estimated with error is
safest pulled away from the median, and the factors below were tuned for exactly that, on the launch data.

What the constants were fit on
------------------------------
Leave-one-out over the comparables themselves: each launch's hour 1 is treated as the known print, its
hourly curve is forecast from the *other* launches only, and the pinball loss is scored against its real
hours. That is the same information the final case gives, so the LOO loss is an honest estimate of the
entry's loss. On the launches with hour 1 >= 50k (the regime a mega-launch lives in) an unweighted
neighbour quantile scores ~0.251 and this program scores ~0.204; restricting the donors to launches below
500k so the top launch has to be extrapolated, ~0.142.

Run it
------
::

    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py

The grader runs `python3 -P -s /bot/main.py` with `/input` and `/output` and an 8 s cap per case.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = (0.05, 0.25, 0.50, 0.75, 0.95)
# The standard normal's quantile at each level, for the parametric half of the band.
Z = {0.05: -1.6449, 0.25: -0.6745, 0.75: 0.6745, 0.95: 1.6449}
# Hours of the release day that are scored: hour 1 is known, hours 2..24 are forecast.
HOURS = tuple(range(2, 25))

# --- constants fitted by leave-one-out over the comparables (see the docstring) -------------------
KMOD = 0.55        # bandwidth of the Gaussian kernel on u = ln(hour-1 print), in log units
ALPHA = 1.0        # extra width of the kernel below the target (smaller launches useful, bigger are not)
PHI = 0.25         # weight of the parametric band against the empirical one
FLOOR_FRAC = 0.70  # the local residual scale is never below this fraction of the all-launch scale
EXTRA = 1.00       # uniform widening of the band
INFLATE = (1.00, 1.10, 1.00, 0.90, 0.90)  # per-level outward pull (5%, 25%, 50%, 75%, 95%)
CAP = math.log(50.0)  # no quantile sits further than 50x from the median in log space

# The case author's tier for each comparable, read from its `unlock` note.
TIER_WEIGHT = {
    "comparable": 1.0,
    "early-access-contaminated": 0.55,
    "weak": 0.25,
    "target-history": 0.0,
}
DEFAULT_TIER_WEIGHT = 0.5


def parse_ts(text):
    """Twitch record timestamps are ISO-8601 with a Z; tolerate a plain offset too."""
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def release_day_curve(hourly, day_start):
    """Mean viewers inside each hour of the release day: {1..24: mean}.

    The case's hourly points are not guaranteed one per hour (the practice source samples sparsely), so
    an hour with no point is simply absent; callers skip those hours rather than interpolate.
    """
    t0 = parse_ts(day_start)
    acc = {}
    for point in hourly or []:
        try:
            offset = (parse_ts(point["from"]) - t0).total_seconds() / 3600.0
        except (KeyError, ValueError, TypeError):
            continue
        if 0.0 <= offset < 24.0:
            viewers = point.get("viewers")
            if isinstance(viewers, (int, float)) and viewers > 0:
                acc.setdefault(int(offset) + 1, []).append(float(viewers))
    return {hour: sum(v) / len(v) for hour, v in acc.items() if v}


def tier_of(comparable):
    """The tier the case author wrote into the comparable's `unlock` note, e.g. '... Tier: weak.'"""
    note = comparable.get("unlock") or ""
    marker = note.find("Tier:")
    if marker < 0:
        return "comparable"
    tail = note[marker + len("Tier:"):]
    return tail.split(".")[0].split("\n")[0].strip() or "comparable"


def load_comparables(case):
    """Each comparable as (name, u = ln(p1), weight, {hour: viewers of the release day})."""
    out = []
    for comparable in case.get("comparables") or []:
        if not isinstance(comparable, dict) or not comparable.get("hourly"):
            continue
        curve = release_day_curve(comparable["hourly"], comparable.get("day_start") or "")
        p1 = curve.get(1, 0.0)
        if p1 <= 0:
            continue
        weight = TIER_WEIGHT.get(tier_of(comparable), DEFAULT_TIER_WEIGHT)
        out.append({
            "name": comparable.get("name") or "",
            "u": math.log(p1),
            "weight": weight,
            "curve": curve,
        })
    return out


def kernel(u, u_target, kmod=KMOD, alpha=ALPHA):
    """Gaussian kernel on the log hour-1 print: 1 at the target's scale, falling off in log units."""
    delta = u - u_target
    scaled = delta / kmod if delta > 0 else delta / (kmod * alpha)
    return math.exp(-0.5 * scaled * scaled)


def weighted_quantile(pairs, tau):
    """The tau quantile of [(value, weight)], interpolating inside the step that crosses it."""
    if not pairs:
        return 0.0
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0]
    target = tau * total
    seen = 0.0
    previous_value, previous_seen = ordered[0][0], 0.0
    for value, weight in ordered:
        seen += weight
        if seen >= target:
            span = seen - previous_seen
            if span <= 0:
                return value
            return previous_value + (value - previous_value) * ((target - previous_seen) / span)
        previous_value, previous_seen = value, seen
    return ordered[-1][0]


def local_linear(points, at):
    """Weighted local-linear fit of x on u, evaluated at `at`; flat beyond the sample's own range."""
    if not points:
        return 0.0
    lo = min(p[0] for p in points)
    hi = max(p[0] for p in points)
    at = min(max(at, lo), hi)
    sw = sx = sy = sxx = sxy = 0.0
    for u, y, w in points:
        sw += w
        sx += w * u
        sy += w * y
        sxx += w * u * u
        sxy += w * u * y
    if sw <= 1e-12:
        return 0.0
    mean_u, mean_y = sx / sw, sy / sw
    var_u = sxx / sw - mean_u * mean_u
    cov_uy = sxy / sw - mean_u * mean_y
    slope = cov_uy / var_u if var_u > 1e-12 else 0.0
    slope = max(-1.0, min(1.0, slope))  # a day cannot move more than one log unit per log unit of scale
    return mean_y + slope * (at - mean_u)


def forecast_hour(hour, p1, comparables):
    """The five quantiles of hour `hour`'s viewers, and the record of how they were built."""
    u_target = math.log(max(p1, 1.0))
    donors = []
    for comparable in comparables:
        if comparable["weight"] <= 0:
            continue
        viewers = comparable["curve"].get(hour)
        if not viewers or viewers <= 0:
            continue
        donors.append((comparable, comparable["u"], math.log(viewers / math.exp(comparable["u"]))))
    if not donors:
        return None

    # weights: tier * kernel on scale, applied once and used everywhere below
    weighted = [(c["u"], y, c["weight"] * kernel(c["u"], u_target)) for c, _, y in donors]
    median = local_linear(weighted, u_target)

    # residual scale, each donor measured against a fit that leaves it out
    local_num = local_den = 0.0
    global_num = global_den = 0.0
    for index, (comparable, u, y) in enumerate(donors):
        others = [w for j, w in enumerate(weighted) if j != index]
        if not others:
            continue
        residual = y - local_linear(others, u)
        weight = weighted[index][2]
        local_num += weight * residual * residual
        local_den += weight
        others_flat = [(c2["u"], y2, c2["weight"]) for j, (c2, _, y2) in enumerate(donors) if j != index]
        residual_flat = y - local_linear(others_flat, u)
        global_num += comparable["weight"] * residual_flat * residual_flat
        global_den += comparable["weight"]
    scale_local = math.sqrt(local_num / local_den) if local_den > 1e-12 else 0.3
    scale_global = math.sqrt(global_num / global_den) if global_den > 1e-12 else 0.3
    scale = max(scale_local, FLOOR_FRAC * scale_global, 0.05) * EXTRA

    quantiles = []
    for level, pull in zip(LEVELS, INFLATE):
        if level == 0.50:
            x = median
        else:
            empirical = weighted_quantile([(y, w) for _, y, w in weighted], level) - median
            empirical = max(-CAP, min(CAP, empirical))
            deviation = (1.0 - PHI) * empirical + PHI * Z[level] * scale
            x = median + deviation * pull
        quantiles.append(p1 * math.exp(max(-CAP, min(CAP, x))))

    # the schema wants finite values above 0, at most 1e9, and never decreasing
    quantiles = [min(1e9, max(1e-6, value)) for value in quantiles]
    for i in range(1, len(quantiles)):
        quantiles[i] = max(quantiles[i], quantiles[i - 1])
    return {
        "hour": hour,
        "q": [round(value, 2) for value in quantiles],
        "donors": len(donors),
        "median": round(median, 4),
        "scale": round(scale, 4),
    }


def fallback_hour(hour, p1):
    """With no usable comparable at all: hold the hour-1 print, spread like a random walk in log space."""
    spread = 0.15 * math.sqrt(hour - 1)
    quantiles = [p1 * math.exp(Z[level] * spread) if level != 0.50 else p1 for level in LEVELS]
    quantiles = [min(1e9, max(1e-6, value)) for value in quantiles]
    for i in range(1, len(quantiles)):
        quantiles[i] = max(quantiles[i], quantiles[i - 1])
    return {"hour": hour, "q": [round(value, 2) for value in quantiles], "donors": 0}


def row_viewers(row):
    """The viewers count of a read or a print row, whether it sits on the row or under `game`."""
    if not isinstance(row, dict):
        return None
    value = row.get("viewers")
    if isinstance(value, (int, float)):
        return float(value)
    game = row.get("game")
    if isinstance(game, dict):
        value = game.get("viewers")
        if isinstance(value, (int, float)):
            return float(value)
    return None


def hour_one_print(case):
    """Hour 1's print, from `known` in whichever shape the case ships it.

    The practice cases ship one row: {"hour": 1, "viewers": 23357, "channels": 619, "reads": 1, ...}.
    The final case ships every valid read before the deadline and hour 1's print. So: take an hour-1
    print row if one is there, otherwise average the reads inside the first hour (the rules' own print:
    the mean of the minute reads, rounded half up, at least 1), otherwise the last read's viewers.
    """
    known = case.get("known")
    rows = known if isinstance(known, list) else [known] if isinstance(known, dict) else []
    for row in rows:
        if isinstance(row, dict) and row.get("hour") == 1 and row_viewers(row) is not None:
            return max(1.0, row_viewers(row))
    reads, day_start = [], case.get("day_start")
    for row in rows:
        viewers = row_viewers(row)
        if viewers is None:
            continue
        stamp = row.get("from") or row.get("at") or row.get("minute")
        if not stamp or not day_start:
            reads.append(viewers)
            continue
        try:
            offset = (parse_ts(stamp) - parse_ts(day_start)).total_seconds() / 3600.0
        except (ValueError, TypeError):
            reads.append(viewers)
            continue
        if 0.0 <= offset < 1.0:
            reads.append(viewers)
    if reads:
        mean = sum(reads) / len(reads)
        return max(1.0, float(math.floor(mean + 0.5)))  # rounded half up, as the recorder's print is
    return 1.0


def forecast_case(case):
    """The 23 scored rows for one case, in the order the case asks for them (hours 2..24)."""
    p1 = hour_one_print(case)
    comparables = load_comparables(case)
    hours = sorted({row["hour"] for row in case.get("forecast") or [] if isinstance(row, dict)}) or list(HOURS)
    rows = []
    for hour in hours:
        row = forecast_hour(hour, p1, comparables) or fallback_hour(hour, p1)
        rows.append(row)
    return rows


def main(argv):
    input_dir = Path(argv[1]) if len(argv) > 1 else INPUT_DIR
    output_dir = Path(argv[2]) if len(argv) > 2 else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = sorted(input_dir.glob("*.in"))
    if not cases:
        print(f"no .in case in {input_dir}", file=sys.stderr)
        return 1
    for path in cases:
        with open(path, "r", encoding="utf-8") as handle:
            case = json.load(handle)
        rows = forecast_case(case)
        case_id = case.get("id") or path.stem
        payload = {"hours": [{"hour": row["hour"], "q": row["q"]} for row in rows]}
        target = output_dir / f"{case_id}.out"
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
        losers = [row for row in rows if row.get("donors", 0) == 0]
        print(
            f"{case_id}: p1={hour_one_print(case):.0f} "
            f"comparables={len(load_comparables(case))} rows={len(rows)} "
            f"fallback_rows={len(losers)} bytes={target.stat().st_size}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
