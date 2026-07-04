"""Compute DRS/ARS/BES/DOHI scores and 180-day ICDAS progression outcome
for the synthetic cohort. Reads data/cohort_raw.csv, writes
data/cohort_scored.csv.

Scoring methodology ported 1:1 from moralguard-app/generate_data.py's
session-first model (see that file for the full citation list):
  - DRS: diet-only, from a simulated Stephan-curve pH trajectory
    (sugar_freq_day exposures of the person's food_type spread across the
    waking day), normalized against a 200-minute worst-case ceiling.
  - ARS/BES: each person's brush_freq individual brushing sessions are
    simulated by jittering force/duration/coverage around that person's
    behavioral mean, scored session-by-session, then averaged — a genuine
    average of simulated per-session values, not a single day-level
    formula. BES is a badness/inefficiency score (higher = worse), same
    direction as DRS/ARS, so DOHI needs no (100-BES) inversion.
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"

PROGRESSION_SEED = 142  # distinct from generation seed (42) to avoid stream overlap
DRS_SIM_SEED = 642      # pH-exposure simulation rng, distinct stream
SESSION_SIM_SEED = 542  # per-session ARS/BES jitter rng, distinct stream

RECOVERY_TIME = {"high_acid": 40, "medium": 25, "low": 10}  # minutes
GRADE_PROGRESSION_PROB = {"Excellent": 0.02, "Good": 0.08, "Fair": 0.20, "Poor": 0.45}

TIME_POINTS = [0, 30, 45, 60, 120, 180, 240]
PLAQUE_POINTS = [0, 12, 16, 20, 41, 50, 100]
COVERAGE_WEIGHTS = {"buccal": (8, 1.0), "occlusal": (4, 1.2), "lingual": (6, 1.5)}
TOTAL_COVERAGE_WEIGHT = sum(n * w for n, w in COVERAGE_WEIGHTS.values())
FREQ_FACTOR_BES = {1: 1.30, 2: 1.00, 3: 0.87}

SESSION_FORCE_JITTER_SD = 0.25
SESSION_TIME_JITTER_SD = 15
SESSION_COVERAGE_JITTER_SD = 6

PH_BASELINE = 7.0
CRITICAL_PH = 5.5
DROP_DURATION_MIN = 5
TROUGH_BY_TYPE = {"high_acid": 4.0, "medium": 5.3, "low": 6.3}
WAKING_WINDOW_MIN = 900
DRS_NORMALIZATION_MIN = 200  # 5 exposures/day x 40 min (worst-case recovery) cap


# ---------------------------------------------------------------- DRS (diet-only pH sim)
def simulate_day_exposure_minutes(food_type, sugar_freq_day, rng=None):
    """Simulate one day's Stephan-curve pH trajectory and return total
    minutes spent below the critical pH 5.5 threshold. Diet/timing only —
    no brushing interaction (brush_delay_min's effect lives entirely in
    session_ars's erosion-abrasion penalty). Pass `rng` for stochastic event
    timing (used for cohort scoring); omit for a deterministic, evenly-
    spaced simulation (used for intervention what-if scoring)."""
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


def compute_drs(df, rng=None):
    minutes = [
        simulate_day_exposure_minutes(row.food_type, row.sugar_freq_day, rng)
        for row in df.itertuples()
    ]
    drs = np.minimum(100.0, np.array(minutes) / DRS_NORMALIZATION_MIN * 100)
    return pd.Series(drs, index=df.index)


# ---------------------------------------------------------------- ARS / BES (session model)
def p_risk(force_n):
    """Piecewise pressure-risk curve: rises from 0N, steepens past 3.0N,
    capped past 3.9N."""
    if force_n <= 3.0:
        return (force_n / 3.0) * 25
    if force_n <= 3.9:
        return 25 + (force_n - 3.0) / 0.9 * 45
    return min(100.0, 70 + (force_n - 3.9) * 15)


def _p_risk_vec(force):
    force = np.asarray(force, dtype=float)
    return np.where(
        force <= 3.0, force / 3.0 * 25,
        np.where(force <= 3.9, 25 + (force - 3.0) / 0.9 * 45,
                 np.minimum(100.0, 70 + (force - 3.9) * 15)),
    )


def session_ars(force_n, duration_sec, food_type, brush_delay_min):
    """Session-level abrasion risk: pressure risk scaled by duration, with
    an erosion-abrasion synergy penalty when brushing too soon after an
    acidic food."""
    time_factor = duration_sec / 120
    if food_type == "high_acid" and brush_delay_min < 30:
        penalty = 1.0 + 1.5 * (1 - brush_delay_min / 30)
    else:
        penalty = 1.0
    return min(100.0, p_risk(force_n) * time_factor * penalty)


def compute_ars(df):
    """Deterministic, single-session evaluation (no jitter) — used for
    intervention what-if scoring, matching moralguard's score_scenario."""
    force = df["brush_force_N"].to_numpy()
    duration = df["brush_time_sec"].to_numpy()
    delay = df["brush_delay_min"].to_numpy()
    is_high_acid_early = (df["food_type"].to_numpy() == "high_acid") & (delay < 30)

    p = _p_risk_vec(force)
    time_factor = duration / 120
    penalty = np.where(is_high_acid_early, 1.0 + 1.5 * (1 - delay / 30), 1.0)
    ars = np.minimum(100.0, p * time_factor * penalty)
    return pd.Series(ars, index=df.index)


def _t_inefficiency(duration_sec):
    removed_pct = np.interp(duration_sec, TIME_POINTS, PLAQUE_POINTS)
    return 100.0 - removed_pct


def _c_inefficiency(buccal_pct, occlusal_pct, lingual_pct):
    (bn, bw), (on, ow), (ln, lw) = (
        COVERAGE_WEIGHTS["buccal"], COVERAGE_WEIGHTS["occlusal"], COVERAGE_WEIGHTS["lingual"]
    )
    weighted_coverage = (
        buccal_pct * bn * bw + occlusal_pct * on * ow + lingual_pct * ln * lw
    ) / TOTAL_COVERAGE_WEIGHT
    return 100.0 - weighted_coverage


def session_bes(duration_sec, buccal_pct, occlusal_pct, lingual_pct, daily_brush_freq):
    """Session-level brushing-quality badness: residual plaque + coverage
    shortfall, scaled by daily brushing frequency."""
    freq_factor = FREQ_FACTOR_BES[min(int(daily_brush_freq), 3)]
    ti = _t_inefficiency(duration_sec)
    ci = _c_inefficiency(buccal_pct, occlusal_pct, lingual_pct)
    return min(100.0, freq_factor * (0.4 * ti + 0.6 * ci))


def compute_bes(df):
    """Deterministic, single-session evaluation (no jitter) — used for
    intervention what-if scoring, matching moralguard's score_scenario."""
    duration = df["brush_time_sec"].to_numpy()
    freq_factor = df["brush_freq"].astype(int).map(FREQ_FACTOR_BES)
    assert freq_factor.isna().sum() == 0, "unexpected brush_freq value outside {1,2,3}"

    ti = _t_inefficiency(duration)
    ci = _c_inefficiency(
        df["buccal_coverage_pct"].to_numpy(),
        df["occlusal_coverage_pct"].to_numpy(),
        df["lingual_coverage_pct"].to_numpy(),
    )
    bes = np.minimum(100.0, freq_factor.to_numpy() * (0.4 * ti + 0.6 * ci))
    return pd.Series(bes, index=df.index)


def simulate_person_ars_bes(rng, row):
    """Simulate that person's individual brushing sessions (one per
    brush_freq) by jittering session-level force/duration/coverage around
    the person's behavioral mean, then average their session_ars/session_bes
    into the person's aggregate — a genuine average of real per-session
    values rather than a single day-level formula."""
    n_sessions = int(row["brush_freq"])
    ars_vals, bes_vals = [], []
    for _ in range(n_sessions):
        force = float(np.clip(rng.normal(row["brush_force_N"], SESSION_FORCE_JITTER_SD), 0.1, 5.0))
        duration = float(np.clip(rng.normal(row["brush_time_sec"], SESSION_TIME_JITTER_SD), 20, 240))
        buccal = float(np.clip(rng.normal(row["buccal_coverage_pct"], SESSION_COVERAGE_JITTER_SD), 0, 100))
        occlusal = float(np.clip(rng.normal(row["occlusal_coverage_pct"], SESSION_COVERAGE_JITTER_SD), 0, 100))
        lingual = float(np.clip(rng.normal(row["lingual_coverage_pct"], SESSION_COVERAGE_JITTER_SD), 0, 100))
        ars_vals.append(session_ars(force, duration, row["food_type"], row["brush_delay_min"]))
        bes_vals.append(session_bes(duration, buccal, occlusal, lingual, n_sessions))
    return float(np.mean(ars_vals)), float(np.mean(bes_vals))


def simulate_ars_bes(df, rng):
    """Stochastic cohort-scoring path: each person's ARS/BES is the average
    of `rng`-jittered simulated sessions (see simulate_person_ars_bes)."""
    ars = np.zeros(len(df))
    bes = np.zeros(len(df))
    for i, (_, row) in enumerate(df.iterrows()):
        a, b = simulate_person_ars_bes(rng, row)
        ars[i] = a
        bes[i] = b
    return ars, bes


# ---------------------------------------------------------------- DOHI / grade / progression
def compute_dohi(drs, ars, bes):
    """BES is a badness scale (higher = worse), same direction as DRS/ARS,
    so no inversion is needed here."""
    dohi = 100 - (0.45 * drs + 0.25 * ars + 0.30 * bes)
    return dohi.clip(0, 100)


def assign_grade(dohi):
    dohi_arr = dohi.to_numpy() if hasattr(dohi, "to_numpy") else np.asarray(dohi)
    conditions = [dohi_arr >= 90, dohi_arr >= 75, dohi_arr >= 60]
    choices = ["Excellent", "Good", "Fair"]
    return np.select(conditions, choices, default="Poor")


def simulate_progression(df, grade, seed=PROGRESSION_SEED):
    rng = np.random.default_rng(seed)
    prob = pd.Series(grade, index=df.index).map(GRADE_PROGRESSION_PROB).to_numpy()
    increment = rng.binomial(1, prob)
    final_icdas = np.minimum(df["baseline_icdas"].to_numpy() + increment, 6)
    progressed = final_icdas > df["baseline_icdas"].to_numpy()
    return final_icdas, progressed


def score_cohort(df):
    df = df.copy()

    rng_drs = np.random.default_rng(DRS_SIM_SEED)
    rng_sessions = np.random.default_rng(SESSION_SIM_SEED)

    df["DRS"] = compute_drs(df, rng_drs).round(2)
    ars, bes = simulate_ars_bes(df, rng_sessions)
    df["ARS"] = np.round(ars, 2)
    df["BES"] = np.round(bes, 2)
    df["DOHI"] = compute_dohi(df["DRS"], df["ARS"], df["BES"]).round(2)
    df["grade"] = assign_grade(df["DOHI"])

    final_icdas, progressed = simulate_progression(df, df["grade"])
    df["final_icdas"] = final_icdas
    df["progressed"] = progressed
    return df


def validate_scored(df):
    for col in ["DRS", "ARS", "BES", "DOHI"]:
        assert df[col].min() >= 0 and df[col].max() <= 100, f"{col} out of [0,100]"
    assert set(df["grade"].unique()) <= {"Excellent", "Good", "Fair", "Poor"}
    assert (df["final_icdas"] >= df["baseline_icdas"]).all()
    assert (df["final_icdas"] <= 6).all()
    assert (df["progressed"] == (df["final_icdas"] > df["baseline_icdas"])).all()
    assert df.isna().sum().sum() == 0, "NaNs present in scored cohort"


def main():
    df = pd.read_csv(DATA_DIR / "cohort_raw.csv")
    scored = score_cohort(df)
    validate_scored(scored)

    print("Scored cohort summary:")
    print(scored[["DRS", "ARS", "BES", "DOHI"]].describe())
    print()
    print("Grade distribution:")
    print(scored["grade"].value_counts())
    print()
    print("Progression rate:", scored["progressed"].mean())

    out_path = DATA_DIR / "cohort_scored.csv"
    scored.to_csv(out_path, index=False)
    print(f"\nWrote {len(scored)} rows to {out_path}")


if __name__ == "__main__":
    main()
