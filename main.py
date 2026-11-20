# v5 (V5-C) — entry v5, DIPROMOSIKAN 20 Nov 2026 (ADOPTION_RULES.md §4/§5).
# Dibangun mekanis dari entry/main.py v2 (sha f993dfef) + DRIFT berbatas
# (Cabang B saja). Perubahan vs v2 HANYA yang tertulis di
# v5-scratch/ADOPTION_RULES.md §2. Lolos G1-G5; v5 = v2 + drift.
"""gta6-trophy-rarity-forecast-01 entry v5: per-group dilution / saturation model (Python 3.13 stdlib only).

Model summary (full derivation + calibration in ../BUILD_LOG.md, sections v1/v2):
  Earn rate = earners / players_tracked. Exophase refreshes its counts in jumps:
  new tracked players arrive mostly without the trophies yet, so every rate
  except a ~100% one FALLS at a refresh (dilution), then the population
  saturates. Two regimes, chosen from the case's own history:

  Branch A -- history shows movement (>= 3 distinct read states): per trophy,
  least-squares fit of logit(rate) against ln(1 + hours since first read)
  (saturating curve, never a linear extrapolation in time), read at the
  target. Spread = random-walk base combined with the fit residual
  (floored at 0.5 logit), capped at 4 logit. UNCHANGED from v1 (the per-trophy
  OLS is already adaptive per trophy; HET_SIGMA does not affect it).

  Branch B -- history is flat (< 3 distinct states, e.g. the practice cases):
  no trend is estimable, so the forecast is a Poisson-refresh mixture.
  Refreshes arrive with mean interval REFRESH_MEAN_HOURS; the k-th refresh
  multiplies a non-100% rate by delta_k = 1 - (1 - delta_1) * RHO**(k-1)
  (later refreshes bite less: the tracked population saturates). Each
  mixture component is a normal in logit space with sigma = random-walk base
  combined with HET_SIGMA (per-trophy dilution heterogeneity). Quantiles are
  found by bisection on the mixture CDF. A trophy at ~100% does not dilute.

  v2 changes vs v1 (see BUILD_LOG.md section v2 for the numbers):
  1. Per-group delta_1 instead of one uniform 0.22067. Each group's delta_1
     is backed out of its observed 12:22 UTC live ratio R (rate_1222 /
     rate_launch) under the assumption R = delta_1 * delta_2 with RHO = 0.35,
     i.e. R = d * (0.65 + 0.35*d), so d = (-0.65 + sqrt(0.4225 + 1.4*R)) / 0.7:
       platinum + completionist (ids 1,14,15,17,25,30,31,32): R 0.1587 -> 0.2185
       story lanjut            (ids 4..13):                    R 0.2091 -> 0.2796
       Chapter 1               (id 3):                         R 0.3920 -> 0.4794
       aktivitas awal          (ids 20,21,35):                 R 0.4149 -> 0.5024
         (median of fishing 0.5686 / arcade 0.4149 / vehicles 0.3120)
       misc                    (32 ids, explicit list below): R 0.1915 -> 0.2586
       fallback global (any id not listed):                    R 0.2013 -> 0.2703
     Group membership is explicit id lists, never name heuristics. The
     fallback keeps the program generic in the number of trophies.
     CAVEAT: the ratios come from ONE Exophase snapshot (crawl 12:22 UTC
     19 Nov 2026, re-served twice by the tool cache -- no second independent
     time point, so drift is unmeasured and NO drift term is added), and the
     "exactly 2 refreshes" back-out assumption is itself unproven (about one
     effective extra refresh between 05:00 and 12:22, possibly offset by real
     earning drift). The delta_1 values are therefore calibrations of the
     observed ratios, not measured per-refresh factors.
  2. REFRESH_MEAN_HOURS 8 -> 4.7 h (P(refresh, 90 min) = 0.273) and
     HET_SIGMA 0.7 -> 0.25 logit. Calibration round (v2-scratch/
     calibration.md): (mu=4.7, rho=0.35, het=0.25) was the only cell beating
     v1 on L_refresh AND mu-consistent expected loss AND final loss, with
     the uniform-delta model; het=0.25 matches the observed SD of ln(ratio)
     (0.253), and after per-group deltas the within-group residual SD is
     0.117 -- so 0.25 stays conservative, not overconfident. RHO stays 0.35.
     Mu=16 was NOT adopted although its mu-consistent E was lowest: that
     ranking is mechanical (it bets refresh is rare, P=0.089) while refresh
     in practice ALREADY happened, and its L_refresh is 16% WORSE than v1.
  3. No drift term: drift could not be measured this round (single cached
     snapshot), so none is added.

  Null rates: last non-null read is used; a trophy never observed gets the
  0.01 floor. The program is generic in the number of trophies (no
  hard-coded 55) and in the horizon. Deterministic, offline, stdlib only.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# --- v2 calibration constants (see BUILD_LOG.md section v2) ---
SIGMA_RW = 0.5        # random-walk logit spread per sqrt-day (as the starter)
HET_SIGMA = 0.25      # per-trophy dilution heterogeneity, logit (v1: 0.7)
RHO = 0.35            # saturation: later refreshes keep (1-DELTA_1)*RHO**(k-1) of the bite
REFRESH_MEAN_HOURS = 4.7   # mean interval between Exophase refreshes (v1: 8.0)
NO_DILUTION_ABOVE = 99.99  # trophies at ~100% do not dilute
MAX_SPREAD = 4.0      # cap on any total logit spread
MIN_RESID = 0.5       # floor on the Branch A residual spread, logit
MAX_SLOPE = 2.0       # clamp on the Branch A logit-vs-log-time slope

# --- v5 constants (ADOPTION_RULES.md §1/§2; ditulis SEBELUM pengujian) ---
FLOOR_POP = 0.0476   # D_plateau populasi: median rasio plat+completionist
                     # vs launch di 15:26 UTC 19 Nov (D(t): 0,1587 -> 0,0476
                     # -> 0,0476 BEKU). Floor median: rate tidak boleh jatuh
                     # di bawah launch_rate x FLOOR_POP (G=1, konservatif).
DRIFT_H_SAT = 24.0   # skala waktu penjenuhan drift (jam)
DRIFT_CAP = 1.0      # total shift drift maksimum (logit) — pelajaran v4
                     # C8/C9: drift tak berbatas membuat q50=100% di final.
# gamma drift (rumus v4 S3(b), hanya kelompok dgn median rasio interval > 1):
# Chapter 1 (id 3) = 0,3697; aktivitas awal (20,21,35) = 0,2408; lainnya 0.
# gamma>0 pada plat/story/misc adalah artefak model dilusi, BUKAN earning.
DRIFT_GAMMA_BY_ID = {3: 0.3697, 20: 0.2408, 21: 0.2408, 35: 0.2408}

# --- v2 per-group delta_1 (backed out of 12:22 UTC live ratios, rho=0.35;
# see module docstring and BUILD_LOG.md section v2). Explicit id lists. ---
DELTA_1_PLAT_COMP = 0.2185   # ids 1,14,15,17,25,30,31,32 (R 0.1587)
DELTA_1_STORY_LATE = 0.2796  # ids 4..13 (R 0.2091)
DELTA_1_CHAPTER1 = 0.4794    # id 3 (R 0.3920)
DELTA_1_EARLY_ACT = 0.5024   # ids 20,21,35 (R median 0.4149)
DELTA_1_MISC = 0.2586        # 32 misc ids below (R median 0.1915)
DELTA_1_DEFAULT = 0.2703     # fallback global (R median non-100% 0.2013)

DELTA_1_BY_ID = {}
for _i in (1, 14, 15, 17, 25, 30, 31, 32):
    DELTA_1_BY_ID[_i] = DELTA_1_PLAT_COMP
for _i in range(4, 14):
    DELTA_1_BY_ID[_i] = DELTA_1_STORY_LATE
DELTA_1_BY_ID[3] = DELTA_1_CHAPTER1
for _i in (20, 21, 35):
    DELTA_1_BY_ID[_i] = DELTA_1_EARLY_ACT
for _i in (16, 18, 19, 22, 23, 24, 26, 27, 28, 29, 33, 34, 36, 37, 38, 39,
           40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55):
    DELTA_1_BY_ID[_i] = DELTA_1_MISC
del _i


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up (same as the starter's JS-style rounding)."""
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


def refresh_shifts(delta_1: float, kmax: int = 80) -> list:
    """S[N] = sum of log delta_k for k = 1..N (S[0] = 0). Saturating: later
    refreshes multiply the rate by a factor closer to 1.
    delta_k = 1 - (1 - delta_1) * RHO**(k-1)."""
    shifts = [0.0]
    total = 0.0
    for k in range(1, kmax + 1):
        delta = 1.0 - (1.0 - delta_1) * (RHO ** (k - 1))
        total += math.log(delta)
        shifts.append(total)
    return shifts


def poisson_weights(lam: float) -> list:
    """Poisson pmf weights until the cumulative mass exceeds 1 - 1e-10."""
    weights = []
    w = math.exp(-lam)
    total = 0.0
    n = 0
    while True:
        weights.append(w)
        total += w
        if total >= 1.0 - 1e-10 or n >= 80:
            break
        n += 1
        w *= lam / n
    return weights


def mixture_quantiles(center: float, lam: float, sigma: float,
                      no_dilution: bool, delta_1: float) -> list:
    """The 5 quantiles (in percent) of the Branch B predictive distribution:
    a Poisson mixture over refresh counts of normals in logit space, with
    this trophy's own group delta_1."""
    if no_dilution:
        return [inv_logit_pct(center + z * sigma) for z in Z]
    shifts = refresh_shifts(delta_1)
    weights = poisson_weights(lam)
    nmax = len(weights) - 1

    def cdf(x: float) -> float:
        s = 0.0
        for n in range(nmax + 1):
            s += weights[n] * phi((x - (center + shifts[n])) / sigma)
        return s

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
        out.append(inv_logit_pct(0.5 * (a + b)))
    return out


def trend_forecast(xs: list, ys: list, x_target: float, base_spread: float):
    """Branch A: OLS of logit(rate) on ln(1+hours). Returns (median_logit,
    spread) or None when the fit is not usable (< 3 points or no x variance)."""
    n = len(xs)
    if n < 3:
        return None
    if max(xs) - min(xs) < 1e-9:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    var = sum((x - mx) ** 2 for x in xs)
    if var < 1e-12:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var
    slope = max(-MAX_SLOPE, min(MAX_SLOPE, slope))
    intercept = my - slope * mx
    resid2 = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    resid = math.sqrt(resid2 / (n - 2)) if n > 2 else MIN_RESID
    spread = math.sqrt(base_spread ** 2 + max(resid, MIN_RESID) ** 2)
    spread = min(spread, MAX_SPREAD)
    return (intercept + slope * x_target, spread)


def last_observed_rate(history: list, i: int):
    """Last non-null rate of trophy i over the history (None if never seen)."""
    for read in reversed(history):
        r = read["rates"][i]
        if r is not None:
            return r
    return None


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    horizon_minutes = case["horizon_minutes"]
    days = horizon_minutes / 1440.0
    horizon_hours = horizon_minutes / 60.0
    base_spread = SIGMA_RW * math.sqrt(days)
    sigma_b = math.sqrt(base_spread ** 2 + HET_SIGMA ** 2)
    lam = horizon_hours / REFRESH_MEAN_HOURS

    first_at = parse_time(history[0]["at"])
    try:
        target_at = parse_time(case["target"])
    except (KeyError, ValueError):
        target_at = parse_time(case["at"])  # fallback, should not happen
        from datetime import timedelta
        target_at = target_at + timedelta(minutes=horizon_minutes)
    t_target = (target_at - first_at).total_seconds() / 3600.0
    x_target = math.log1p(max(t_target, 0.0))

    # Regime switch: how many distinct read states does the history hold?
    states = {(r.get("players_tracked"), tuple(r["rates"])) for r in history}
    use_trend = len(states) >= 3

    rows = []
    for i, trophy in enumerate(trophies):
        r_last = last_observed_rate(history, i)
        center = logit(r_last if r_last is not None else 0.0)
        qs = None
        if use_trend:
            xs, ys = [], []
            for read in history:
                r = read["rates"][i]
                if r is None:
                    continue
                t_h = (parse_time(read["at"]) - first_at).total_seconds() / 3600.0
                xs.append(math.log1p(max(t_h, 0.0)))
                ys.append(logit(r))
            fit = trend_forecast(xs, ys, x_target, base_spread)
            if fit is not None:
                med_logit, spread = fit
                qs = [inv_logit_pct(med_logit + z * spread) for z in Z]
        if qs is None:
            # Branch B: Poisson-refresh mixture around the last observed rate,
            # with this trophy's group delta_1.
            no_dilution = (r_last is not None and r_last >= NO_DILUTION_ABOVE)
            delta_1 = DELTA_1_BY_ID.get(trophy["id"], DELTA_1_DEFAULT)
            # v5 DRIFT berbatas, Cabang B SAJA (ADOPTION_RULES.md §2):
            # shift pusat +gamma*h_eff, h_eff = 24*(1-exp(-h/24)),
            # total <= +1,0 logit. Cabang A tidak disentuh (isolasi).
            gamma = DRIFT_GAMMA_BY_ID.get(trophy["id"], 0.0)
            if gamma:
                h_eff = DRIFT_H_SAT * (1.0 - math.exp(-horizon_hours / DRIFT_H_SAT))
                center = center + min(DRIFT_CAP, gamma * h_eff)
            qs = mixture_quantiles(center, lam, sigma_b, no_dilution, delta_1)
        qs = [round4(min(100.0, max(0.0, q))) for q in qs]
        # Safety net: rounding must not break the non-decreasing order.
        for j in range(1, 5):
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
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
