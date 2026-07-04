"""Generate 180-day daily data and one detailed "today" day (multi-session
1-second pressure timeseries + 1440-minute pH timeseries) for a single
MoralGuard patient ("민재").

Scoring methodology (session-first, distinct from the OralGuard cohort
pipeline's day-only formulas — see the design discussion in the session
history: MoralGuard needs session-level ARS/BES so the app can show a
per-brushing-session score, which the original day-averaged OralGuard
model has no concept of):

  - Daily_DRS: diet only, computed once per day by simulating that day's
    Stephan-curve pH trajectory (sugar_freq_day events of the day's food
    type) and summing actual minutes below the critical pH 5.5 threshold,
    normalized against a 200-minute ceiling. Brushing timing does not
    affect DRS at all — the natural pH recovery runs to completion
    regardless of when you brush; brush_delay_min's effect on risk lives
    entirely in Session_ARS's erosion-abrasion penalty below.
  - Session_ARS, Session_BES: computed per individual brushing session
    from that session's own measured pressure/duration/coverage.
  - Daily_ARS / Daily_BES: mean of that day's Session_ARS / Session_BES
    values (a day with brush_freq=2 has 2 sessions averaged, etc).
  - DOHI = 100 - (0.45*Daily_DRS + 0.25*Daily_ARS + 0.30*Daily_BES),
    clipped to [0, 100]. Note BES is now a badness/inefficiency score
    (higher = worse), unlike the OralGuard cohort model where BES was an
    efficiency score (higher = better) — this makes all three
    sub-scores uniformly "higher = worse", which is why DOHI no longer
    needs the (100 - BES) inversion the OralGuard formula used.

References: Stephan & Miller (1943) / Dimensions of Dental Hygiene (2024)
for demineralization exposure timing; Wiegand (2013), Hamza (2023), Senna
(2008) for the pressure risk curve; Attin et al., Caries Res (2001) for
the erosion-abrasion synergy penalty; Creeth et al., J Dent Hyg (2009)
for the time-vs-plaque-removal curve; Wang et al., Biosensors (2025) and
Parkinson et al. (2022) for the 18-zone coverage weights; Brusius et al.,
Braz Oral Res (2023) for the brushing-frequency risk factor; Bratthall &
Petersson, Community Dent Oral Epidemiol (2005) and Featherstone, Oral
Health Prev Dent (2004) for the weighted-sum composite index design.

Writes moralguard-app/data/patient_data.json.
"""
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

SEED = 7
N_DAYS = 180
TODAY = datetime(2026, 7, 3)

FOOD_TYPES = ["high_acid", "medium", "low"]
FOOD_PROBS = [0.30, 0.50, 0.20]

RECOVERY_TIME = {"high_acid": 40, "medium": 25, "low": 10}  # minutes
DRS_NORMALIZATION_MIN = 200  # 5 exposures/day x 40 min (worst-case recovery) cap
TIME_POINTS = [0, 30, 45, 60, 120, 180, 240]
PLAQUE_POINTS = [0, 12, 16, 20, 41, 50, 100]
COVERAGE_WEIGHTS = {"buccal": (8, 1.0), "occlusal": (4, 1.2), "lingual": (6, 1.5)}
FREQ_FACTOR_BES = {1: 1.30, 2: 1.00, 3: 0.87}

SESSION_FORCE_JITTER_SD = 0.25
SESSION_TIME_JITTER_SD = 15
SESSION_COVERAGE_JITTER_SD = 6

PH_BASELINE = 7.0
CRITICAL_PH = 5.5
DROP_DURATION_MIN = 5
TROUGH_BY_TYPE = {"high_acid": 4.0, "medium": 5.3, "low": 6.3}
WAKING_WINDOW_MIN = 900  # ~15h waking day used for the generic per-day exposure sim


def assign_grade_scalar(dohi):
    if dohi >= 90:
        return "Excellent"
    if dohi >= 75:
        return "Good"
    if dohi >= 60:
        return "Fair"
    return "Poor"


# ---------------------------------------------------------------- scoring
def simulate_day_exposure_minutes(food_type, sugar_freq_day, rng=None):
    """Simulate that day's Stephan-curve pH trajectory (sugar_freq_day
    eating events of the day's dominant food_type, spread across the waking
    day) and return the actual total minutes spent below the critical pH
    5.5 threshold. This is diet/timing only — no brushing interaction is
    modeled here, matching the "양치와 무관, 순수하게 식이 행동만 반영"
    (independent of brushing, purely diet) design: the natural Stephan
    recovery curve runs to completion regardless of when the patient
    brushes, so brush_delay_min does not enter DRS at all (its effect on
    the model lives entirely in Session_ARS's erosion-abrasion penalty,
    which IS specifically about brushing too soon after acid exposure).

    Pass `rng` for a stochastic draw (used for the 180-day history, for
    realistic day-to-day variation); omit it for a deterministic, evenly-
    spaced simulation (used for intervention what-if scoring, where we want
    a reproducible number rather than fresh randomness on every call).
    """
    n_events = max(1, int(round(sugar_freq_day)))
    trough = TROUGH_BY_TYPE[food_type]
    recovery = RECOVERY_TIME[food_type]
    tau = recovery / 3
    interval = WAKING_WINDOW_MIN / n_events

    event_times = []
    for i in range(n_events):
        base_t = i * interval
        if rng is not None:
            base_t += rng.uniform(-interval * 0.2, interval * 0.2)
        event_times.append(max(0, base_t))

    n = int(WAKING_WINDOW_MIN)
    ph = np.full(n, PH_BASELINE)
    for et in event_times:
        et_i = int(et)
        for t in range(et_i, min(n, et_i + DROP_DURATION_MIN + recovery)):
            local_t = t - et_i
            if local_t < DROP_DURATION_MIN:
                frac = local_t / DROP_DURATION_MIN
                candidate = PH_BASELINE + (trough - PH_BASELINE) * frac
            else:
                t_rec = local_t - DROP_DURATION_MIN
                candidate = trough + (PH_BASELINE - trough) * (1 - np.exp(-t_rec / tau))
            ph[t] = min(ph[t], candidate)

    return int(np.sum(ph < CRITICAL_PH))


def daily_drs_from_minutes(total_dem_minutes):
    """Normalize total demineralization-exposure minutes against a
    worst-case ceiling of 5 sugar exposures/day at the longest (high-acid)
    40-minute natural recovery time = 200 minutes."""
    return min(100.0, total_dem_minutes / DRS_NORMALIZATION_MIN * 100)


def p_risk(force_n):
    """Piecewise pressure-risk curve: near-flat below 3.0N (clinically
    non-significant abrasion), steepening 3.0-3.9N, capped past that."""
    if force_n <= 3.0:
        return (force_n / 3.0) * 25
    if force_n <= 3.9:
        return 25 + (force_n - 3.0) / 0.9 * 45
    return min(100.0, 70 + (force_n - 3.9) * 15)


def session_ars(force_n, duration_sec, food_type, brush_delay_min):
    """Session-level abrasion risk: pressure risk, scaled by brush duration,
    with an erosion-abrasion synergy penalty when brushing too soon after
    an acidic food (Attin et al. 2001)."""
    time_factor = duration_sec / 120
    if food_type == "high_acid" and brush_delay_min < 30:
        penalty = 1.0 + 1.5 * (1 - brush_delay_min / 30)
    else:
        penalty = 1.0
    return min(100.0, p_risk(force_n) * time_factor * penalty)


def t_inefficiency(duration_sec):
    removed_pct = float(np.interp(duration_sec, TIME_POINTS, PLAQUE_POINTS))
    return 100.0 - removed_pct


def c_inefficiency(buccal_pct, occlusal_pct, lingual_pct):
    (bn, bw), (on, ow), (ln, lw) = COVERAGE_WEIGHTS["buccal"], COVERAGE_WEIGHTS["occlusal"], COVERAGE_WEIGHTS["lingual"]
    total_weight = bn * bw + on * ow + ln * lw
    weighted_coverage = (buccal_pct * bn * bw + occlusal_pct * on * ow + lingual_pct * ln * lw) / total_weight
    return 100.0 - weighted_coverage


def session_bes(duration_sec, buccal_pct, occlusal_pct, lingual_pct, daily_brush_freq):
    """Session-level brushing-quality badness: residual plaque + coverage
    shortfall, scaled by how many times per day the patient brushes at all
    (Brusius et al. 2023 frequency risk factor)."""
    freq_factor = FREQ_FACTOR_BES[min(int(daily_brush_freq), 3)]
    ti = t_inefficiency(duration_sec)
    ci = c_inefficiency(buccal_pct, occlusal_pct, lingual_pct)
    return min(100.0, freq_factor * (0.4 * ti + 0.6 * ci))


def compute_dohi(drs, ars, bes):
    dohi = 100 - (0.45 * drs + 0.25 * ars + 0.30 * bes)
    return max(0.0, min(100.0, dohi))


def simulate_daily_ars_bes(rng, day_row):
    """Simulate that day's individual brushing sessions (one per brush_freq)
    by jittering session-level force/duration/coverage around the day's
    behavioral mean, then average their Session_ARS/Session_BES into the
    day's aggregate — this is what makes Daily_ARS/BES a genuine average of
    real per-session values rather than a single day-level formula."""
    n_sessions = int(day_row["brush_freq"])
    ars_vals, bes_vals = [], []
    for _ in range(n_sessions):
        force = float(np.clip(rng.normal(day_row["brush_force_N"], SESSION_FORCE_JITTER_SD), 0.1, 5.0))
        duration = float(np.clip(rng.normal(day_row["brush_time_sec"], SESSION_TIME_JITTER_SD), 20, 240))
        buccal = float(np.clip(rng.normal(day_row["buccal_coverage_pct"], SESSION_COVERAGE_JITTER_SD), 0, 100))
        occlusal = float(np.clip(rng.normal(day_row["occlusal_coverage_pct"], SESSION_COVERAGE_JITTER_SD), 0, 100))
        lingual = float(np.clip(rng.normal(day_row["lingual_coverage_pct"], SESSION_COVERAGE_JITTER_SD), 0, 100))
        ars_vals.append(session_ars(force, duration, day_row["food_type"], day_row["brush_delay_min"]))
        bes_vals.append(session_bes(duration, buccal, occlusal, lingual, n_sessions))
    return float(np.mean(ars_vals)), float(np.mean(bes_vals))


# ---------------------------------------------------------------- generation
def ar1_series(rng, n, mu, sigma, phi, lo, hi, mid_bump=None):
    """Bounded AR(1) process around a mean, optionally with a temporary
    mid-series excursion (mid_bump = (start_day, end_day, extra_mean))."""
    e = np.zeros(n)
    z = rng.normal(0, 1, n)
    e[0] = sigma * z[0]
    for t in range(1, n):
        e[t] = phi * e[t - 1] + sigma * z[t]
    series = mu + e
    if mid_bump is not None:
        start, end, extra = mid_bump
        ramp = np.zeros(n)
        span = end - start
        for t in range(start, min(end, n)):
            # smooth up-then-down bump (triangular) over [start, end]
            frac = (t - start) / span
            ramp[t] = extra * (1 - abs(2 * frac - 1))
        series = series + ramp
    return np.clip(series, lo, hi)


def recent_improvement_ramp(n, start_day, shift):
    """Linear ramp from 0 to `shift`, applied from `start_day` to the last
    day — models a patient who recently improved a habit, so "today" lands
    in a solid Good range rather than an arbitrary AR(1) draw."""
    ramp = np.zeros(n)
    span = max(1, n - 1 - start_day)
    for t in range(start_day, n):
        ramp[t] = shift * min(1.0, (t - start_day) / span)
    return ramp


def build_daily_series(rng):
    """Generate each day's behavioral MEANS (not scores yet) — these are the
    center each day's individual brushing sessions jitter around."""
    brush_time_sec = ar1_series(rng, N_DAYS, 100, 20, 0.7, 30, 240)
    # Overpressure phase within the last 30 days (peaks ~day 161), recovered
    # by "today" (day 180) so the intervention baseline picker (worst day in
    # last 30 days) can find a meaningful high-force day to anchor on.
    brush_force_N = ar1_series(rng, N_DAYS, 1.3, 0.35, 0.7, 0.3, 5.0,
                                mid_bump=(150, 172, 1.4))
    brush_freq = np.clip(np.round(ar1_series(rng, N_DAYS, 2.0, 0.4, 0.5, 1, 3)), 1, 3).astype(int)
    buccal_cov = ar1_series(rng, N_DAYS, 78, 8, 0.6, 0, 100)
    occlusal_cov = ar1_series(rng, N_DAYS, 62, 10, 0.6, 0, 100)
    lingual_cov = ar1_series(rng, N_DAYS, 48, 10, 0.6, 0, 100)
    sugar_freq_day = ar1_series(rng, N_DAYS, 4.0, 1.0, 0.6, 1, 10)
    brush_delay_min = ar1_series(rng, N_DAYS, 40, 15, 0.6, 0, 180)
    food_type = rng.choice(FOOD_TYPES, size=N_DAYS, p=FOOD_PROBS)

    # Recent habit improvement narrative: over the last 20 days, brush delay
    # rises (this no longer affects DRS, which is diet-only, but it *does*
    # lower Session_ARS's erosion-abrasion penalty for high-acid foods),
    # coverage rises (lowers BES), and sugar_freq_day falls (lowers DRS via
    # fewer daily demineralization events) — so "today" (day 180) reads as
    # a solid Good day while the calendar/trajectory history still shows
    # the earlier rough patch (overpressure bump, variable DRS/BES) for a
    # realistic story.
    improve_start = N_DAYS - 20
    brush_delay_min = np.clip(
        brush_delay_min + recent_improvement_ramp(N_DAYS, improve_start, 22), 0, 180
    )
    buccal_cov = np.clip(buccal_cov + recent_improvement_ramp(N_DAYS, improve_start, 12), 0, 100)
    occlusal_cov = np.clip(occlusal_cov + recent_improvement_ramp(N_DAYS, improve_start, 15), 0, 100)
    lingual_cov = np.clip(lingual_cov + recent_improvement_ramp(N_DAYS, improve_start, 18), 0, 100)
    sugar_freq_day = np.clip(
        sugar_freq_day - recent_improvement_ramp(N_DAYS, improve_start, 1.2), 1, 10
    )

    df = pd.DataFrame({
        "brush_time_sec": brush_time_sec,
        "brush_force_N": brush_force_N,
        "brush_freq": brush_freq,
        "buccal_coverage_pct": buccal_cov,
        "occlusal_coverage_pct": occlusal_cov,
        "lingual_coverage_pct": lingual_cov,
        "sugar_freq_day": sugar_freq_day,
        "food_type": food_type,
        "brush_delay_min": brush_delay_min,
    })

    dates = [(TODAY - timedelta(days=N_DAYS - 1 - i)).strftime("%Y-%m-%d") for i in range(N_DAYS)]
    df["day"] = np.arange(1, N_DAYS + 1)
    df["date"] = dates
    return df


def score_daily_series(df, rng_ars_bes, rng_ph):
    """Compute Daily_DRS by simulating that day's pH exposure curve, and
    Daily_ARS/Daily_BES by simulating and averaging that day's individual
    brushing sessions."""
    df = df.copy()

    exposure_min = np.zeros(len(df), dtype=int)
    for i, (_, row) in enumerate(df.iterrows()):
        exposure_min[i] = simulate_day_exposure_minutes(row["food_type"], row["sugar_freq_day"], rng_ph)
    df["dem_exposure_min"] = exposure_min
    df["DRS"] = [daily_drs_from_minutes(m) for m in exposure_min]

    daily_ars = np.zeros(len(df))
    daily_bes = np.zeros(len(df))
    for i, (_, row) in enumerate(df.iterrows()):
        a, b = simulate_daily_ars_bes(rng_ars_bes, row)
        daily_ars[i] = a
        daily_bes[i] = b
    df["ARS"] = daily_ars
    df["BES"] = daily_bes

    df["DOHI"] = [compute_dohi(d, a, b) for d, a, b in zip(df["DRS"], df["ARS"], df["BES"])]
    df["grade"] = df["DOHI"].apply(assign_grade_scalar)
    return df


def build_pressure_session(rng, mean_force, duration_sec, overpressure=False):
    """1-second resolution pressure timeseries for one brushing session."""
    t = np.arange(duration_sec)
    base = ar1_series(rng, duration_sec, mean_force, 0.15, 0.6, 0.1, 5.0)
    if overpressure:
        # inject a clear overpressure excursion above 3.0N for a few seconds
        start = rng.integers(int(duration_sec * 0.3), int(duration_sec * 0.6))
        length = rng.integers(6, 14)
        bump = np.zeros(duration_sec)
        peak = rng.uniform(3.3, 4.2)
        for i in range(length):
            frac = i / length
            bump[start + i] = (peak - mean_force) * (1 - abs(2 * frac - 1)) if start + i < duration_sec else 0
        base = base + bump
    base = np.clip(base, 0.1, 5.0)
    return t.tolist(), base.round(2).tolist()


def build_zone_coverage(rng, buccal_agg, occlusal_agg, lingual_agg):
    jitter = rng.uniform(-8, 8, size=18)
    buccal = np.clip(buccal_agg + jitter[0:8], 0, 100)
    occlusal = np.clip(occlusal_agg + jitter[8:12], 0, 100)
    lingual = np.clip(lingual_agg + jitter[12:18], 0, 100)
    return {
        "buccal": buccal.round(1).tolist(),
        "occlusal": occlusal.round(1).tolist(),
        "lingual": lingual.round(1).tolist(),
    }


def avg_arr(arr):
    return sum(arr) / len(arr)


def build_today_sessions(rng, today_row):
    """Two named demo sessions for "today" with full 1-second pressure
    timeseries + jittered 18-zone coverage (used for the Trends tab's
    detailed per-session views). Each session's own Session_ARS/Session_BES
    is attached so the frontend and this script compute identical numbers
    from the same measured values."""
    sessions = []
    session_defs = [
        {"time": "07:12", "overpressure": False},
        {"time": "22:41", "overpressure": True},
    ]
    for i, sdef in enumerate(session_defs):
        duration = int(np.clip(rng.normal(today_row["brush_time_sec"], 15), 45, 200))
        t, force = build_pressure_session(
            rng, today_row["brush_force_N"], duration, overpressure=sdef["overpressure"]
        )
        max_force = max(force)
        min_force = min(force)
        overpressure_sec = sum(1 for f in force if f > 3.0)
        zone_cov = build_zone_coverage(
            rng,
            today_row["buccal_coverage_pct"],
            today_row["occlusal_coverage_pct"],
            today_row["lingual_coverage_pct"],
        )
        avg_force = float(np.mean(force))
        buccal_pct = avg_arr(zone_cov["buccal"])
        occlusal_pct = avg_arr(zone_cov["occlusal"])
        lingual_pct = avg_arr(zone_cov["lingual"])
        s_ars = session_ars(avg_force, duration, today_row["food_type"], today_row["brush_delay_min"])
        s_bes = session_bes(duration, buccal_pct, occlusal_pct, lingual_pct, len(session_defs))
        overall_cov = (sum(zone_cov["buccal"]) + sum(zone_cov["occlusal"]) + sum(zone_cov["lingual"])) / 18
        sessions.append({
            "session_id": i,
            "time": sdef["time"],
            "duration_sec": duration,
            "avg_force_N": round(avg_force, 2),
            "max_force_N": round(float(max_force), 2),
            "min_force_N": round(float(min_force), 2),
            "overpressure_sec": overpressure_sec,
            "pressure_timeseries": {"t": t, "force": force},
            "zone_coverage": zone_cov,
            "overall_coverage_pct": round(overall_cov, 1),
            "session_ars": round(s_ars, 2),
            "session_bes": round(s_bes, 2),
        })
    return sessions


def simulate_day_ph(rng):
    n = 1440
    ph = np.full(n, PH_BASELINE)

    event_defs = [
        (7 * 60 + int(rng.integers(-15, 15)), "breakfast", "medium"),
        (12 * 60 + int(rng.integers(-20, 20)), "lunch", rng.choice(["high_acid", "medium"])),
        (15 * 60 + int(rng.integers(-30, 30)), "snack", "high_acid"),
        (18 * 60 + int(rng.integers(-20, 20)), "dinner", rng.choice(["medium", "low"])),
        (21 * 60 + int(rng.integers(-30, 30)), "drink", rng.choice(["high_acid", "low"])),
    ]

    events = []
    for minute, label, food_type in event_defs:
        minute = int(np.clip(minute, 0, n - 1))
        trough = TROUGH_BY_TYPE[food_type]
        recovery = RECOVERY_TIME[food_type]
        for t in range(minute, min(n, minute + DROP_DURATION_MIN + recovery)):
            if t < minute + DROP_DURATION_MIN:
                frac = (t - minute) / DROP_DURATION_MIN
                candidate = PH_BASELINE + (trough - PH_BASELINE) * frac
            else:
                t_rec = t - (minute + DROP_DURATION_MIN)
                tau = recovery / 3
                candidate = trough + (PH_BASELINE - trough) * (1 - np.exp(-t_rec / tau))
            ph[t] = min(ph[t], candidate)
        events.append({"minute": minute, "label": label, "food_type": food_type})

    noise = rng.normal(0, 0.05, n)
    ph = np.clip(ph + noise, 3.5, 7.3)
    return {
        "minute": list(range(n)),
        "ph": ph.round(3).tolist(),
        "events": events,
    }


def _row_to_features(row):
    return {
        "brush_time_sec": row["brush_time_sec"],
        "brush_force_N": row["brush_force_N"],
        "brush_freq": int(row["brush_freq"]),
        "buccal_coverage_pct": row["buccal_coverage_pct"],
        "occlusal_coverage_pct": row["occlusal_coverage_pct"],
        "lingual_coverage_pct": row["lingual_coverage_pct"],
        "sugar_freq_day": row["sugar_freq_day"],
        "food_type": row["food_type"],
        "brush_delay_min": row["brush_delay_min"],
    }


def score_scenario(feat):
    # Deterministic (rng=None) exposure sim: none of the 3 intervention
    # levers touch food_type/sugar_freq_day, so before/after DRS is always
    # identical within a scenario — a fixed, reproducible number here is
    # more appropriate than fresh stochastic noise on every call.
    exposure_min = simulate_day_exposure_minutes(feat["food_type"], feat["sugar_freq_day"])
    drs = daily_drs_from_minutes(exposure_min)
    ars = session_ars(feat["brush_force_N"], feat["brush_time_sec"], feat["food_type"], feat["brush_delay_min"])
    bes = session_bes(feat["brush_time_sec"], feat["buccal_coverage_pct"], feat["occlusal_coverage_pct"],
                       feat["lingual_coverage_pct"], feat["brush_freq"])
    dohi = compute_dohi(drs, ars, bes)
    return {"DRS": round(drs, 2), "ARS": round(ars, 2), "BES": round(bes, 2), "DOHI": round(dohi, 2)}


def build_interventions(df):
    """Each of the 3 demo scenarios is anchored on its OWN worst real day in
    the last 30 days for the specific lever being tested (e.g. "reduce
    force" is measured against the day force actually spiked) — a single
    shared baseline breaks down because Session_ARS depends on BOTH
    brush_time_sec AND brush_force_N jointly (and Session_BES also depends
    on brush_time_sec), so forcing every scenario onto one baseline lets an
    unrelated bad value contaminate a scenario that isn't about it. Every
    scenario reports all of DRS/ARS/BES/DOHI (not just its "primary" score)
    since the formulas are interconnected enough that a lever can move more
    than one sub-score.

    The combined "fix everything" projection (used for the single
    motivational delta) is evaluated separately on ONE real day — whichever
    day is worst specifically on the two levers these fixes can move
    (see worst_fixable_row below) — with all 3 fixes applied together, so
    it's an internally consistent single calculation rather than an invalid
    sum of 3 deltas measured from different starting points.
    """
    recent = df.iloc[-30:]
    worst_time_row = recent.loc[recent["brush_time_sec"].idxmin()]
    worst_force_row = recent.loc[recent["brush_force_N"].idxmax()]

    # The combined demo anchors on whichever day is worst specifically on
    # the two dimensions our 3 levers can actually move (ARS, BES) — not
    # raw worst-DOHI, since DRS (45% of DOHI) is diet-only now and none of
    # these interventions touch it, so a DRS-dominated "worst day" would
    # understate how much these particular fixes are worth.
    fixable_badness = 0.25 * recent["ARS"] + 0.30 * recent["BES"]
    worst_fixable_row = recent.loc[fixable_badness.idxmax()]

    # "Wait 30 min before brushing" only ever moves Session_ARS's erosion-
    # abrasion penalty now (DRS is diet-only and brushing-independent), so
    # its demo day must actually trigger that penalty: a high-acid day
    # brushed soon after eating. Fall back to the full 180-day history if
    # the last 30 days happen to have none.
    high_acid_recent = recent[recent["food_type"] == "high_acid"]
    if len(high_acid_recent) > 0:
        worst_penalty_row = high_acid_recent.loc[high_acid_recent["brush_delay_min"].idxmin()]
    else:
        high_acid_all = df[df["food_type"] == "high_acid"]
        worst_penalty_row = high_acid_all.loc[high_acid_all["brush_delay_min"].idxmin()]

    # Each override is a floor/ceiling ("ensure at least" / "cap at most"),
    # not a flat replacement — a flat override can *backfire* on a base row
    # that already satisfies the target (e.g. forcing brush_force_N to a
    # flat 1.5N on a day whose real force was already 1.2N would raise it).
    def apply_targets(base_row, targets):
        row = _row_to_features(base_row)
        for field, (direction, value) in targets.items():
            current = row[field]
            row[field] = max(current, value) if direction == "at_least" else min(current, value)
        return row

    scenario_defs = {
        "wait_30min": {
            "label": "식후 30분 대기 양치", "base_row": worst_penalty_row,
            "targets": {"brush_delay_min": ("at_least", 30)},
        },
        "brush_2min": {
            "label": "양치 시간 2분 충족", "base_row": worst_time_row,
            "targets": {"brush_time_sec": ("at_least", 120)},
        },
        "reduce_force": {
            "label": "압력 1.5N으로 감소", "base_row": worst_force_row,
            "targets": {"brush_force_N": ("at_most", 1.5)},
        },
    }

    results = {}
    for key, s in scenario_defs.items():
        before = score_scenario(_row_to_features(s["base_row"]))
        after = score_scenario(apply_targets(s["base_row"], s["targets"]))
        results[key] = {
            "label": s["label"],
            "affects": ["DRS", "ARS", "BES", "DOHI"],
            "reference_date": s["base_row"]["date"],
            "before": before,
            "after": after,
            "delta_dohi": round(after["DOHI"] - before["DOHI"], 2),
        }

    combined_targets = {}
    for s in scenario_defs.values():
        combined_targets.update(s["targets"])
    before_combined = score_scenario(_row_to_features(worst_fixable_row))
    after_combined = score_scenario(apply_targets(worst_fixable_row, combined_targets))
    results["combined"] = {
        "label": "습관 3가지 모두 교정",
        "affects": ["DRS", "ARS", "BES", "DOHI"],
        "reference_date": worst_fixable_row["date"],
        "before": before_combined,
        "after": after_combined,
        "delta_dohi": round(after_combined["DOHI"] - before_combined["DOHI"], 2),
    }
    return results


def main():
    rng = np.random.default_rng(SEED)
    df = build_daily_series(rng)

    rng_sessions = np.random.default_rng(SEED + 500)
    rng_ph_exposure = np.random.default_rng(SEED + 600)
    df = score_daily_series(df, rng_sessions, rng_ph_exposure)

    today_row = df.iloc[-1]
    sessions = build_today_sessions(np.random.default_rng(SEED + 1), today_row)
    ph_data = simulate_day_ph(np.random.default_rng(SEED + 2))

    # Override "today"'s Daily_DRS/ARS/BES/DOHI/grade so the Home tab's
    # headline number is consistent with what's actually visualized
    # elsewhere in the app:
    #  - DRS: real total minutes below critical pH from TODAY's own richly
    #    simulated 1440-minute curve (ph_data), not the generic per-day sim
    #    used for the other 179 days — reuses the exact curve the Trends
    #    tab's pH view already shows.
    #  - ARS/BES: average of the 2 *named* demo sessions actually shown in
    #    the Trends tab (rather than the generic simulated sessions from
    #    score_daily_series).
    today_exposure_min = int(sum(1 for p in ph_data["ph"] if p < CRITICAL_PH))
    today_drs = daily_drs_from_minutes(today_exposure_min)
    today_ars = float(np.mean([s["session_ars"] for s in sessions]))
    today_bes = float(np.mean([s["session_bes"] for s in sessions]))
    today_dohi = compute_dohi(today_drs, today_ars, today_bes)
    today_grade = assign_grade_scalar(today_dohi)

    last_idx = df.index[-1]
    df.loc[last_idx, "dem_exposure_min"] = today_exposure_min
    df.loc[last_idx, "DRS"] = today_drs
    df.loc[last_idx, "ARS"] = today_ars
    df.loc[last_idx, "BES"] = today_bes
    df.loc[last_idx, "DOHI"] = today_dohi
    df.loc[last_idx, "grade"] = today_grade

    interventions = build_interventions(df)

    daily_series = df[[
        "day", "date", "DRS", "ARS", "BES", "DOHI", "grade", "dem_exposure_min",
        "brush_time_sec", "brush_force_N", "brush_freq",
        "buccal_coverage_pct", "occlusal_coverage_pct", "lingual_coverage_pct",
        "sugar_freq_day", "food_type", "brush_delay_min",
    ]].round(2).to_dict(orient="records")

    result = {
        "patient": {"name": "민재", "id": "MJ-001", "today_date": TODAY.strftime("%Y-%m-%d")},
        "daily_series": daily_series,
        "today": {
            "date": TODAY.strftime("%Y-%m-%d"),
            "DRS": round(today_drs, 2),
            "ARS": round(today_ars, 2),
            "BES": round(today_bes, 2),
            "DOHI": round(today_dohi, 2),
            "grade": today_grade,
            "sessions": sessions,
            "ph_timeseries": ph_data,
        },
        "interventions": interventions,
    }

    # Sanity checks
    assert len(daily_series) == N_DAYS
    assert len(sessions) == 2
    assert len(ph_data["ph"]) == 1440
    for s in sessions:
        assert len(s["pressure_timeseries"]["force"]) == s["duration_sec"]
        assert min(s["pressure_timeseries"]["force"]) >= 0
    assert all(0 <= r["DOHI"] <= 100 for r in daily_series)
    assert all(0 <= r["ARS"] <= 100 for r in daily_series)
    assert all(0 <= r["BES"] <= 100 for r in daily_series)

    out_path = "data/patient_data.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path}")
    print("Today:", result["today"]["DOHI"], result["today"]["grade"],
          "(DRS", result["today"]["DRS"], "ARS", result["today"]["ARS"], "BES", result["today"]["BES"], ")")
    print("Sessions:", [(s["time"], s["avg_force_N"], s["max_force_N"], s["overpressure_sec"], s["session_ars"], s["session_bes"]) for s in sessions])
    print("Interventions:", {k: v["delta_dohi"] for k, v in interventions.items()})


if __name__ == "__main__":
    main()
