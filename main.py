#!/usr/bin/env python3
"""GTA VI Trophy Rarity Forecast (V7-D Tahap-2 / V2)."""
from __future__ import annotations

import json
import math
import os
import sys
import shutil
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path for imports (like v5)
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 2. Konstanta
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
SIGMA_RW = 0.5
HET_SIGMA = 0.25
RHO = 0.35
REFRESH_MEAN_HOURS = 4.7
NO_DILUTION_ABOVE = 99.99
MAX_SPREAD = 4.0
MIN_RESID = 0.5
MAX_SLOPE = 2.0
DRIFT_H_SAT = 24.0
DRIFT_CAP = 1.0
DRIFT_GAMMA_BY_ID = {3: 0.1898, 4: 0.1902, 8: 0.2200, 20: 0.1296, 21: 0.1296, 35: 0.1296,
                     46: 0.2107, 50: 0.1662}

# 3. delta_1 per kelompok
DELTA_1_PLAT_COMP = 0.2185  # ids 1, 14, 15, 17, 25, 30, 31, 32
DELTA_1_STORY_LATE = 0.2796  # ids 4..13
DELTA_1_CHAPTER1 = 0.4794  # id 3
DELTA_1_EARLY_ACT = 0.5024  # ids 20, 21, 35
DELTA_1_MISC = 0.2586  # ids 16, 18, 19, 22, 23, 24, 26, 27, 28, 29, 33, 34,
                       # 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                       # 50, 51, 52, 53, 54, 55
DELTA_1_DEFAULT = 0.2703

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
# Font fallback untuk id lain
for trophy_id in range(1, 56):
    if trophy_id not in DELTA_1_BY_ID:
        DELTA_1_BY_ID[trophy_id] = DELTA_1_DEFAULT

# 4. Fungsi bantu
def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(lg: float) -> float:
    if lg >= 0.0:
        return 100.0 / (1.0 + math.exp(-lg))
    else:
        e = math.exp(lg)
        return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    if abs(value - round(value)) < 1e-12:
        return int(round(value))
    return value


def parse_time(s: str) -> datetime:
    """ISO8601 to datetime, handle Z."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def phi(z: float) -> float:
    """Standard normal CDF using erf."""
    return 0.5 * (1.0 + erf(z / math.sqrt(2.0)))


def erf(x: float) -> float:
    """Error function from standard library."""
    # Using math.erf directly
    return math.erf(x)


def rho_effective(horizon_hours: float) -> float:
    """Rho sadar-horizon."""
    if horizon_hours <= 12.0:
        return 0.35
    elif horizon_hours >= 24.0:
        return 0.77
    else:
        f = math.log(horizon_hours / 12.0) / math.log(2.0)
        return 0.35 + (0.77 - 0.35) * f


def refresh_shifts(delta_1: float, rho: float, kmax: int = 80) -> list[float]:
    """Poisson refresh shifts per delta_1."""
    shifts = [0.0]
    total = 0.0
    for k in range(1, kmax + 1):
        delta = 1.0 - (1.0 - delta_1) * (rho ** (k - 1))
        total += math.log(delta)
        shifts.append(total)
    return shifts


def poisson_weights(lam: float) -> list[float]:
    """Poisson weights until cumulative >= 0.9999999999 or n >= 80."""
    weights = []
    w = math.exp(-lam)
    total = 0.0
    n = 0
    while total < 1.0 - 1e-10 and n < 80:
        weights.append(w)
        total += w
        n += 1
        w *= lam / n
    return weights


def adaptive_lam_multiplier(history: list[dict], trophies: list[dict]) -> float:
    """Lambda adaptive dua-rezim (v7-b)."""
    if len(history) < 2:
        return 1.0

    states = {(r.get("players_tracked"), tuple(r["rates"])) for r in history}
    if len(states) < 2:
        return 0.0

    if len(history) == 2:
        return 1.0

    # >=3 read
    first = history[-2]
    last = history[-1]
    w_h = (parse_time(last["at"]) - parse_time(first["at"])).total_seconds() / 3600.0
    if w_h <= 0:
        return 1.0

    ks = []
    for i, trophy in enumerate(trophies):
        if trophy["id"] == 2:
            continue
        rf = first["rates"][i]
        rl = last["rates"][i]
        if rf is None or rl is None or rf <= 0.0 or rl <= 0.0:
            continue
        d1 = DELTA_1_BY_ID.get(trophy["id"], DELTA_1_DEFAULT)
        frac = (logit(rf) - logit(rl)) / (-math.log(d1))
        ks.append(max(0.0, frac))

    if not ks:
        return 1.0

    ks.sort()
    if len(ks) % 2 == 1:
        k_hat = ks[len(ks) // 2]
    else:
        k_hat = 0.5 * (ks[len(ks) // 2 - 1] + ks[len(ks) // 2])

    m_hist = k_hat / (w_h / REFRESH_MEAN_HOURS)
    return max(0.0, min(2.5, m_hist))


def mixture_quantiles(center: float, lam: float, sigma: float,
                       no_dilution: bool, delta_1: float,
                       rho: float = RHO) -> list[float]:
    """Cabang B: campuran Poisson-refresh."""
    if no_dilution:
        return [inv_logit_pct(center + z * sigma) for z in Z]

    shifts = refresh_shifts(delta_1, rho=rho)
    weights = poisson_weights(lam)
    nmax = len(weights) - 1

    def cdf(x: float) -> float:
        total = 0.0
        for n, w in enumerate(weights):
            total += w * phi((x - (center + shifts[n])) / sigma)
        return total

    lo = center + shifts[nmax] - 10.0 * sigma - 1.0
    hi = center + 10.0 * sigma + 1.0

    out = []
    for tau in LEVELS:
        a, b = lo, hi
        for _ in range(80):
            m = 0.5 * (a + b)
            if cdf(m) < tau:
                a = m
            else:
                b = m
        # CRITICAL: Append INV_LOGIT_PCT of midpoint (not raw logit)
        out.append(inv_logit_pct(0.5 * (a + b)))
    return out


def trend_forecast(xs: list[float], ys: list[float],
                    x_target: float, base_spread: float) -> tuple[float, float] | None:
    """Cabang A: tren OLS."""
    if len(xs) < 3:
        return None
    if max(xs) - min(xs) < 1e-9:
        return None

    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    var = sum((x - mx) ** 2 for x in xs)
    if var < 1e-12:
        return None

    n = len(xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var
    slope = max(-MAX_SLOPE, min(MAX_SLOPE, slope))

    intercept = my - slope * mx

    resid2 = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    resid = math.sqrt(resid2 / (n - 2)) if n > 2 else MIN_RESID

    spread = math.sqrt(base_spread ** 2 + max(resid, MIN_RESID) ** 2)
    spread = min(MAX_SPREAD, spread)

    return (intercept + slope * x_target, spread)


def forecast_case(case: dict) -> dict:
    """Alur utama satu case."""
    # Validasi
    if case["levels"] != LEVELS:
        raise ValueError("levels mismatch")
    history = case.get("history", [])
    if not history:
        raise ValueError("history empty")

    trophies = case["trophies"]
    horizon_minutes = case["horizon_minutes"]

    days = horizon_minutes / 1440.0
    horizon_hours = horizon_minutes / 60.0

    base_spread = SIGMA_RW * math.sqrt(days)

    # HET_SIGMA sadar-horizon
    het_eff = HET_SIGMA * (1.0 - math.exp(-horizon_hours / REFRESH_MEAN_HOURS))
    sigma_b = math.sqrt(base_spread ** 2 + het_eff ** 2)

    # Lam adaptif
    m_hist = adaptive_lam_multiplier(history, trophies)
    if m_hist < 1.0:
        m_h = m_hist + (1.0 - m_hist) * (1.0 - math.exp(-horizon_hours / DRIFT_H_SAT))
    else:
        m_h = m_hist
    lam = horizon_hours / REFRESH_MEAN_HOURS * m_h

    first_at = parse_time(history[0]["at"])

    # Target parsing
    target_str = case.get("target")
    try:
        target_at = parse_time(target_str)
    except (KeyError, ValueError, TypeError):
        checkpoint_at = parse_time(case["at"])
        target_at = checkpoint_at + timedelta(minutes=horizon_minutes)

    t_target = (target_at - first_at).total_seconds() / 3600.0
    x_target = math.log1p(max(t_target, 0.0))

    # Regime switch
    states = {(r.get("players_tracked"), tuple(r["rates"])) for r in history}
    use_trend = len(states) >= 3

    results = []
    for i, trophy in enumerate(trophies):
        r_last = None
        # Cari rate non-null TERAKHIR
        for r in reversed(history):
            if r.get("rates") and len(r["rates"]) > i:
                val = r["rates"][i]
                if val is not None:
                    r_last = val
                    break

        center = logit(r_last if r_last is not None else 0.0)
        qs = None

        # Cabang A (trend)
        if use_trend:
            xs = []
            ys = []
            for r in history:
                if r.get("rates") and len(r["rates"]) > i:
                    r_days = r["rates"][i]
                    if r_days is not None:
                        t_h = (parse_time(r["at"]) - first_at).total_seconds() / 3600.0
                        xs.append(math.log1p(max(t_h, 0.0)))
                        ys.append(logit(r_days))

            if len(xs) >= 3:
                fit = trend_forecast(xs, ys, x_target, base_spread)
                if fit is not None:
                    med_logit, spread = fit
                    qs = [inv_logit_pct(med_logit + z * spread) for z in Z]

        # Cabang B (mixture)
        if qs is None:
            no_dilution = (r_last is not None and r_last >= NO_DILUTION_ABOVE)
            delta_1 = DELTA_1_BY_ID.get(trophy["id"], DELTA_1_DEFAULT)

            # DRIFT onset menjenuh (Cabang B saja)
            gamma = DRIFT_GAMMA_BY_ID.get(trophy["id"], 0.0)
            if gamma:
                s_g = gamma / 0.2200
                center = center + DRIFT_CAP * s_g * (1.0 - math.exp(-horizon_hours / DRIFT_H_SAT))

            qs = mixture_quantiles(center, lam, sigma_b, no_dilution, delta_1,
                                   rho=rho_effective(horizon_hours))

        # Safety net: non-decreasing after rounding
        qs = [round4(min(100.0, max(0.0, q))) for q in qs]
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]

        results.append({"id": trophy["id"], "q": qs})

    return {"trophies": results}


def main() -> None:
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

    output_dir.mkdir(parents=True, exist_ok=True)

    # Baca semua *.in, terurut
    for in_path in sorted(input_dir.glob("*.in")):
        stem = in_path.stem
        with in_path.open("r", encoding="utf-8") as f:
            case = json.load(f)

        forecast = forecast_case(case)

        out_path = output_dir / f"{stem}.out"
        out_path.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()