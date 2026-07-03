"""Compute DRS/ARS/BES/DOHI scores and 180-day ICDAS progression outcome
for the synthetic cohort. Reads data/cohort_raw.csv, writes
data/cohort_scored.csv.
"""
import numpy as np
import pandas as pd

PROGRESSION_SEED = 142  # distinct from generation seed (42) to avoid stream overlap

RECOVERY_TIME = {"high_acid": 40, "medium": 25, "low": 10}  # minutes
FREQ_FACTOR = {1: 0.70, 2: 1.00, 3: 1.15}
GRADE_PROGRESSION_PROB = {"Excellent": 0.02, "Good": 0.08, "Fair": 0.20, "Poor": 0.45}

TIME_POINTS = [0, 30, 45, 60, 120, 180, 240]
PLAQUE_POINTS = [0, 12, 16, 20, 41, 50, 100]


def compute_drs(df):
    recovery_time = df["food_type"].map(RECOVERY_TIME)
    dem_per_exposure = (recovery_time - df["brush_delay_min"]).clip(lower=0)
    det = dem_per_exposure * df["sugar_freq_day"]
    return (det / 120 * 100).clip(upper=100)


def compute_ars(df):
    force = df["brush_force_N"].to_numpy()
    conditions = [force <= 2.5, force <= 3.0, force <= 3.9]
    choices = [
        np.zeros_like(force),
        (force - 2.5) / 0.5 * 30,
        30 + (force - 3.0) / 0.9 * 40,
    ]
    p_risk = np.select(conditions, choices, default=100.0)

    brush_freq = df["brush_freq"].astype(int).to_numpy()
    ars_raw = p_risk * (df["brush_time_sec"].to_numpy() / 120) * (brush_freq / 2)

    is_high_acid_early = (df["food_type"] == "high_acid") & (df["brush_delay_min"] < 30)
    penalty = np.where(
        is_high_acid_early,
        1.0 + 1.5 * (1 - df["brush_delay_min"].to_numpy() / 30),
        1.0,
    )
    return pd.Series(np.minimum(100, ars_raw * penalty), index=df.index)


def compute_bes(df):
    t_score = np.interp(df["brush_time_sec"].to_numpy(), TIME_POINTS, PLAQUE_POINTS)

    buccal_zones = df["buccal_coverage_pct"] * 8 * 1.0
    occlusal_zones = df["occlusal_coverage_pct"] * 4 * 1.2
    lingual_zones = df["lingual_coverage_pct"] * 6 * 1.5
    total_weight = 8 * 1.0 + 4 * 1.2 + 6 * 1.5
    c_score = (buccal_zones + occlusal_zones + lingual_zones) / total_weight

    freq_factor = df["brush_freq"].astype(int).map(FREQ_FACTOR)
    assert freq_factor.isna().sum() == 0, "unexpected brush_freq value outside {1,2,3}"

    bes = freq_factor * (0.4 * t_score + 0.6 * c_score)
    return np.minimum(100, bes)


def compute_dohi(drs, ars, bes):
    dohi = 100 - (0.45 * drs + 0.25 * ars + 0.30 * (100 - bes))
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
    df["DRS"] = compute_drs(df).round(2)
    df["ARS"] = compute_ars(df).round(2)
    df["BES"] = compute_bes(df).round(2)
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
    df = pd.read_csv("data/cohort_raw.csv")
    scored = score_cohort(df)
    validate_scored(scored)

    print("Scored cohort summary:")
    print(scored[["DRS", "ARS", "BES", "DOHI"]].describe())
    print()
    print("Grade distribution:")
    print(scored["grade"].value_counts())
    print()
    print("Progression rate:", scored["progressed"].mean())

    out_path = "data/cohort_scored.csv"
    scored.to_csv(out_path, index=False)
    print(f"\nWrote {len(scored)} rows to {out_path}")


if __name__ == "__main__":
    main()
