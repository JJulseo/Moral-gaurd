"""Generate 180-day daily data and one detailed "today" day (multi-session
1-second pressure timeseries + 1440-minute pH timeseries) for a single
MoralGuard patient ("민재"). Reuses the DRS/ARS/BES/DOHI scoring formulas
from the OralGuard pipeline (../scripts/scoring.py) so both apps share the
same methodology. Writes moralguard-app/data/patient_data.json.
"""
import json
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, "../scripts")
from scoring import (  # noqa: E402
    RECOVERY_TIME,
    compute_ars,
    compute_bes,
    compute_drs,
    compute_dohi,
)

SEED = 7
N_DAYS = 180
TODAY = datetime(2026, 7, 3)

FOOD_TYPES = ["high_acid", "medium", "low"]
FOOD_PROBS = [0.30, 0.50, 0.20]


def assign_grade_scalar(dohi):
    if dohi >= 90:
        return "Excellent"
    if dohi >= 75:
        return "Good"
    if dohi >= 60:
        return "Fair"
    return "Poor"


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

    # Recent habit improvement narrative: over the last 20 days, delay drops
    # and coverage rises, so "today" (day 180) reads as a solid Good day
    # while the calendar/trajectory history still shows the earlier rough
    # patch (overpressure bump, variable DRS/BES) for a realistic story.
    # Note: DRS falls as brush_delay_min *rises* toward recovery_time (waiting
    # for the acid to naturally neutralize before brushing lowers demineralization
    # exposure — see compute_drs), so the improvement here increases delay.
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

    df["DRS"] = compute_drs(df)
    df["ARS"] = compute_ars(df)
    df["BES"] = compute_bes(df)
    df["DOHI"] = compute_dohi(df["DRS"], df["ARS"], df["BES"])
    df["grade"] = df["DOHI"].apply(assign_grade_scalar)

    dates = [(TODAY - timedelta(days=N_DAYS - 1 - i)).strftime("%Y-%m-%d") for i in range(N_DAYS)]
    df["day"] = np.arange(1, N_DAYS + 1)
    df["date"] = dates
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


def build_today_sessions(rng, today_row):
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
        overall_cov = (
            sum(zone_cov["buccal"]) + sum(zone_cov["occlusal"]) + sum(zone_cov["lingual"])
        ) / 18
        sessions.append({
            "session_id": i,
            "time": sdef["time"],
            "duration_sec": duration,
            "avg_force_N": round(float(np.mean(force)), 2),
            "max_force_N": round(float(max_force), 2),
            "min_force_N": round(float(min_force), 2),
            "overpressure_sec": overpressure_sec,
            "pressure_timeseries": {"t": t, "force": force},
            "zone_coverage": zone_cov,
            "overall_coverage_pct": round(overall_cov, 1),
        })
    return sessions


PH_BASELINE = 7.0
DROP_DURATION_MIN = 5
TROUGH_BY_TYPE = {"high_acid": 4.0, "medium": 5.3, "low": 6.3}


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


def build_interventions(df):
    """Each scenario is anchored on its own 'worst day in the last 30 days'
    for the specific lever being tested, so every scenario shows a
    meaningful, non-degenerate improvement (see design discussion: using a
    single day like 'today' can make an already-resolved habit show a
    trivial 0-point delta)."""
    recent = df.iloc[-30:]

    def score(row_dict):
        row_df = pd.DataFrame([row_dict])
        drs = compute_drs(row_df).iloc[0]
        ars = compute_ars(row_df).iloc[0]
        bes = compute_bes(row_df).iloc[0]
        dohi = compute_dohi(pd.Series([drs]), pd.Series([ars]), pd.Series([bes])).iloc[0]
        return {
            "DRS": round(float(drs), 2), "ARS": round(float(ars), 2),
            "BES": round(float(bes), 2), "DOHI": round(float(dohi), 2),
        }

    def scenario(worst_row, overrides):
        row = _row_to_features(worst_row)
        row.update(overrides)
        return row

    worst_drs_row = recent.loc[recent["DRS"].idxmax()]
    worst_time_row = recent.loc[recent["brush_time_sec"].idxmin()]
    worst_force_row = recent.loc[recent["brush_force_N"].idxmax()]

    scenarios = {
        "wait_30min": {
            "label": "식후 30분 대기 양치",
            "before": scenario(worst_drs_row, {}),
            "after": scenario(worst_drs_row, {"brush_delay_min": 30}),
        },
        "brush_2min": {
            "label": "양치 시간 2분 충족",
            "before": scenario(worst_time_row, {}),
            "after": scenario(worst_time_row, {"brush_time_sec": 120}),
        },
        "reduce_force": {
            "label": "압력 1.5N으로 감소",
            "before": scenario(worst_force_row, {}),
            "after": scenario(worst_force_row, {"brush_force_N": 1.5}),
        },
    }

    results = {}
    for key, s in scenarios.items():
        before = score(s["before"])
        after = score(s["after"])
        results[key] = {
            "label": s["label"],
            "before": before,
            "after": after,
            "delta_dohi": round(after["DOHI"] - before["DOHI"], 2),
        }
    return results


def main():
    rng = np.random.default_rng(SEED)
    df = build_daily_series(rng)

    today_row = df.iloc[-1]
    sessions = build_today_sessions(np.random.default_rng(SEED + 1), today_row)
    ph_data = simulate_day_ph(np.random.default_rng(SEED + 2))
    interventions = build_interventions(df)

    daily_series = df[[
        "day", "date", "DRS", "ARS", "BES", "DOHI", "grade",
        "brush_time_sec", "brush_force_N", "brush_freq",
        "buccal_coverage_pct", "occlusal_coverage_pct", "lingual_coverage_pct",
        "sugar_freq_day", "food_type", "brush_delay_min",
    ]].round(2).to_dict(orient="records")

    result = {
        "patient": {"name": "민재", "id": "MJ-001", "today_date": TODAY.strftime("%Y-%m-%d")},
        "daily_series": daily_series,
        "today": {
            "date": TODAY.strftime("%Y-%m-%d"),
            "DRS": round(float(today_row["DRS"]), 2),
            "ARS": round(float(today_row["ARS"]), 2),
            "BES": round(float(today_row["BES"]), 2),
            "DOHI": round(float(today_row["DOHI"]), 2),
            "grade": today_row["grade"],
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

    out_path = "data/patient_data.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path}")
    print("Today:", result["today"]["DOHI"], result["today"]["grade"])
    print("Sessions:", [(s["time"], s["avg_force_N"], s["max_force_N"], s["overpressure_sec"]) for s in sessions])
    print("Interventions:", {k: v["delta_dohi"] for k, v in interventions.items()})


if __name__ == "__main__":
    main()
