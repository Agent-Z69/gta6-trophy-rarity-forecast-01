"""gta6-trophy-rarity-forecast-01 v5 (Cabang A + Cabang B + DRIFT).

Target: output byte-identical dengan expected/p00.out dan expected/p01.out.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean, fsum
from typing import List, Tuple, Optional

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))  # noqa

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Level quantiles (ROW TROPHY ORDER)
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# v5: Cabang "random walk logit spread per sqrt-daya"
SIGMA_RW = 0.5

# v5: heterogenitas logit per-trophy
HET_SIGMA = 0.25

# v5: saturasi refresh (α)
RHO = 0.35

# v5: rata-rata interval Exophase refresh (jam)
REFRESH_MEAN_HOURS = 4.7

# v5: non-dilution threshold
NO_DILUTION_ABOVE = 99.99

# v5: cap total logit spread
MAX_SPREAD = 4.0

# v5: floor residual spread Cabang A
MIN_RESID = 0.5

# v5: cap slope Cabang A
MAX_SLOPE = 2.0

# v5: floor populasi (dicatat tapi tidak dipakai eksplisit)
FLOOR_POP = 0.0476

# v5: drift saturation scale (jam)
DRIFT_H_SAT = 24.0

# v5: cap drift shift (logit)
DRIFT_CAP = 1.0

# v5: gamma drift per-trophy id
DRIFT_GAMMA_BY_ID = {3: 0.3697, 20: 0.2408, 21: 0.2408, 35: 0.2408}


DELTA_1_PLAT_COMP = 0.2185  # ids 1, 14, 15, 17, 25, 30, 31, 32
DELTA_1_STORY_LATE = 0.2796  # ids 4..13
DELTA_1_CHAPTER1 = 0.4794  # id 3
DELTA_1_EARLY_ACT = 0.5024  # ids 20, 21, 35
DELTA_1_MISC = 0.2586  # ids 16, 18, 19, 22, 23, 24, 26, 27, 28, 29, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55
DELTA_1_DEFAULT = 0.2703  # fallback

DELTA_1_BY_ID = {
    1: DELTA_1_PLAT_COMP,
    14: DELTA_1_PLAT_COMP,
    15: DELTA_1_PLAT_COMP,
    17: DELTA_1_PLAT_COMP,
    25: DELTA_1_PLAT_COMP,
    30: DELTA_1_PLAT_COMP,
    31: DELTA_1_PLAT_COMP,
    32: DELTA_1_PLAT_COMP,
    4: DELTA_1_STORY_LATE,
    5: DELTA_1_STORY_LATE,
    6: DELTA_1_STORY_LATE,
    7: DELTA_1_STORY_LATE,
    8: DELTA_1_STORY_LATE,
    9: DELTA_1_STORY_LATE,
    10: DELTA_1_STORY_LATE,
    11: DELTA_1_STORY_LATE,
    12: DELTA_1_STORY_LATE,
    13: DELTA_1_STORY_LATE,
    3: DELTA_1_CHAPTER1,
    20: DELTA_1_EARLY_ACT,
    21: DELTA_1_EARLY_ACT,
    35: DELTA_1_EARLY_ACT,
    16: DELTA_1_MISC,
    18: DELTA_1_MISC,
    19: DELTA_1_MISC,
    22: DELTA_1_MISC,
    23: DELTA_1_MISC,
    24: DELTA_1_MISC,
    26: DELTA_1_MISC,
    27: DELTA_1_MISC,
    28: DELTA_1_MISC,
    29: DELTA_1_MISC,
    33: DELTA_1_MISC,
    34: DELTA_1_MISC,
    36: DELTA_1_MISC,
    37: DELTA_1_MISC,
    38: DELTA_1_MISC,
    39: DELTA_1_MISC,
    40: DELTA_1_MISC,
    41: DELTA_1_MISC,
    42: DELTA_1_MISC,
    43: DELTA_1_MISC,
    44: DELTA_1_MISC,
    45: DELTA_1_MISC,
    46: DELTA_1_MISC,
    47: DELTA_1_MISC,
    48: DELTA_1_MISC,
    49: DELTA_1_MISC,
    50: DELTA_1_MISC,
    51: DELTA_1_MISC,
    52: DELTA_1_MISC,
    53: DELTA_1_MISC,
    54: DELTA_1_MISC,
    55: DELTA_1_MISC,
}

# Default fallback
DELTA_1_BY_ID = {k: v for k, v in DELTA_1_BY_ID.items()}
DELTA_1_BY_ID.setdefault(55, DELTA_1_DEFAULT)


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(lg: float) -> float:
    """Return percentage (0–100)."""
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    else:
        e = math.exp(lg)
        return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Rounded to 4 decimal places, half up (JavaScript Math.round(*10000)/10000)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def refresh_shifts(delta_1: float, kmax: int = 80) -> List[float]:
    """S[0] = 0.0; S[k] = sum_{m=1..k} log(delta_m); delta_m = 1 - (1 - delta_1) * rho^(m-1)."""
    shifts = [0.0]
    total = 0.0
    for k in range(1, kmax + 1):
        delta = 1.0 - (1.0 - delta_1) * (RHO ** (k - 1))
        total += math.log(delta)
        shifts.append(total)
    return shifts


def poisson_weights(lam: float) -> List[float]:
    """List of Poisson weights w0..wN, sum -> >= 0.9999999999."""
    w = math.exp(-lam)
    weights = [w]
    total = w
    n = 0
    while total < 1.0 - 1e-10 and n < 80:
        n += 1
        w *= lam / n
        total += w
        weights.append(w)
    return weights


def mixture_quantiles(
    center: float,
    lam: float,
    sigma: float,
    no_dilution: bool,
    delta_1: float,
) -> List[float]:
    """Return [q05, q25, q50, q75, q95]."""
    if no_dilution:
        return [inv_logit_pct(center + z * sigma) for z in Z]

    shifts = refresh_shifts(delta_1)
    weights = poisson_weights(lam)
    nmax = len(weights) - 1

    def cdf(x: float) -> float:
        total = 0.0
        for n in range(nmax + 1):
            total += weights[n] * phi((x - (center + shifts[n])) / sigma)
        return total

    lo = center + shifts[nmax] - 10.0 * sigma - 1.0
    hi = center + 10.0 * sigma + 1.0
    result = []
    for tau in LEVELS:
        a, b = lo, hi
        for _ in range(80):
            m = (a + b) / 2.0
            if cdf(m) < tau:
                a = m
            else:
                b = m
        median_logit = 0.5 * (a + b)
        result.append(inv_logit_pct(median_logit))
    return result


def trend_forecast(
    xs: List[float],
    ys: List[float],
    x_target: float,
    base_spread: float,
) -> Optional[Tuple[float, float]]:
    """Cabang A: OLS trend med_logit + spread. Return (med_logit, spread)."""
    if len(xs) < 3:
        return None
    if max(xs) - min(xs) < 1e-9:
        return None

    mx = mean(xs)
    my = mean(ys)
    var = fsum((x - mx) ** 2 for x in xs)
    if var < 1e-12:
        return None

    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var
    slope = max(-MAX_SLOPE, min(MAX_SLOPE, slope))
    intercept = my - slope * mx

    # Residual STDEV
    n = len(xs)
    if n > 2:
        resid2_sum = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
        resid = math.sqrt(resid2_sum / (n - 2))
    else:
        resid = MIN_RESID

    spread = math.sqrt(base_spread ** 2 + max(resid, MIN_RESID) ** 2)
    spread = min(spread, MAX_SPREAD)

    med_logit = intercept + slope * x_target
    return med_logit, spread


def forecast_case(case: dict, name: str) -> dict:
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name}: unexpected levels {case.get('levels')}")
    if not case.get("history"):
        raise ValueError(f"{name}: empty history")

    trophies = case["trophies"]
    horizon_minutes = case["horizon_minutes"]

    days = horizon_minutes / 1440.0
    horizon_hours = horizon_minutes / 60.0

    base_spread = SIGMA_RW * math.sqrt(days)
    sigma_b = math.sqrt(base_spread ** 2 + HET_SIGMA ** 2)

    lam = horizon_hours / REFRESH_MEAN_HOURS

    first_at = parse_time(case["history"][0]["at"])

    # Try to parse target_at; fallback to at + horizon_minutes
    try:
        target_at = parse_time(case["target"])
    except (KeyError, ValueError):
        at = parse_time(case["at"])
        target_at = at + timedelta(minutes=horizon_minutes)

    t_target = (target_at - first_at).total_seconds() / 3600.0
    x_target = math.log1p(max(t_target, 0.0))

    # Regime switch: 3+ unique (players_tracked, rates tuple) → use trend
    states = {(r.get("players_tracked"), tuple(r["rates"])) for r in case["history"]}
    use_trend = len(states) >= 3

    rows = []
    for i, trophy in enumerate(trophies):
        r_last = None
        for r in reversed(case["history"]):
            rv = r["rates"][i]
            if rv is not None:
                r_last = rv
                break

        lg = logit(r_last if r_last is not None else 0.0)
        qs: Optional[List[float]] = None

        if use_trend:
            xs = []
            ys = []
            for r in case["history"]:
                t_h = (parse_time(r["at"]) - first_at).total_seconds() / 3600.0
                xs.append(math.log1p(max(t_h, 0.0)))
                rv = r["rates"][i]
                if rv is not None:
                    ys.append(logit(rv))
            fit = trend_forecast(xs, ys, x_target, base_spread)
            if fit is not None:
                med_logit, spread = fit
                qs = [inv_logit_pct(med_logit + z * spread) for z in Z]

        if qs is None:
            no_dilution = r_last is not None and r_last >= NO_DILUTION_ABOVE
            delta_1 = DELTA_1_BY_ID.get(trophy["id"], DELTA_1_DEFAULT)

            gamma = DRIFT_GAMMA_BY_ID.get(trophy["id"], 0.0)
            if gamma:
                h_eff = DRIFT_H_SAT * (1.0 - math.exp(-horizon_hours / DRIFT_H_SAT))
                center = lg + min(DRIFT_CAP, gamma * h_eff)
            else:
                center = lg

            qs = mixture_quantiles(center, lam, sigma_b, no_dilution, delta_1)

        # Rounding & clamping
        qs = [round4(min(100.0, max(0.0, q))) for q in qs]

        # Safety net: non-decreasing after rounding
        for j in range(1, 4):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]

        rows.append({"id": trophy["id"], "q": qs})

    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()