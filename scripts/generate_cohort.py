"""Generate a synthetic 300-person / 180-day oral health cohort.

Each row represents one person's 180-day average values (not daily rows).
Distributions are based on published dental/hygiene literature; see the
project design spec for citations per variable.
"""
import numpy as np
import pandas as pd

SEED = 42
N = 300

BRUSH_FREQ_CHOICES = [1, 2, 3]
BRUSH_FREQ_PROBS = [0.20, 0.65, 0.15]

FOOD_TYPE_CHOICES = ["high_acid", "medium", "low"]
FOOD_TYPE_PROBS = [0.30, 0.50, 0.20]

BASELINE_ICDAS_CHOICES = [0, 1, 2, 3, 4, 5]
BASELINE_ICDAS_PROBS = [0.45, 0.20, 0.15, 0.10, 0.05, 0.05]


def make_rng(seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def gen_brush_time_sec(rng, n):
    return np.clip(rng.normal(96.6, 36.0, n), 30, 240)


def gen_brush_force_N(rng, n):
    return np.clip(rng.normal(1.3, 0.5, n), 0.3, 5.0)


def gen_brush_freq(rng, n):
    return rng.choice(BRUSH_FREQ_CHOICES, size=n, p=BRUSH_FREQ_PROBS)


def gen_coverage_pct(rng, n, mean, std):
    return np.clip(rng.normal(mean, std, n), 0, 100)


def gen_sugar_freq_day(rng, n):
    return np.clip(rng.normal(4.0, 1.5, n), 1, 10)


def gen_food_type(rng, n):
    return rng.choice(FOOD_TYPE_CHOICES, size=n, p=FOOD_TYPE_PROBS)


def gen_brush_delay_min(rng, n):
    return np.clip(rng.normal(45, 30, n), 0, 180)


def gen_age(rng, n):
    return np.clip(rng.normal(35, 12, n), 18, 70)


def gen_baseline_icdas(rng, n):
    return rng.choice(BASELINE_ICDAS_CHOICES, size=n, p=BASELINE_ICDAS_PROBS)


def build_cohort(rng, n=N):
    # Draw order fixed and documented for reproducibility.
    brush_time_sec = gen_brush_time_sec(rng, n)
    brush_force_N = gen_brush_force_N(rng, n)
    brush_freq = gen_brush_freq(rng, n)
    buccal_coverage_pct = gen_coverage_pct(rng, n, 75, 15)
    occlusal_coverage_pct = gen_coverage_pct(rng, n, 60, 20)
    lingual_coverage_pct = gen_coverage_pct(rng, n, 45, 20)
    sugar_freq_day = gen_sugar_freq_day(rng, n)
    food_type = gen_food_type(rng, n)
    brush_delay_min = gen_brush_delay_min(rng, n)
    age = gen_age(rng, n)
    baseline_icdas = gen_baseline_icdas(rng, n)

    df = pd.DataFrame({
        "person_id": [f"P{i+1:04d}" for i in range(n)],
        "brush_time_sec": brush_time_sec.round(2),
        "brush_force_N": brush_force_N.round(2),
        "brush_freq": brush_freq.astype(int),
        "buccal_coverage_pct": buccal_coverage_pct.round(2),
        "occlusal_coverage_pct": occlusal_coverage_pct.round(2),
        "lingual_coverage_pct": lingual_coverage_pct.round(2),
        "sugar_freq_day": sugar_freq_day.round(2),
        "food_type": food_type,
        "brush_delay_min": brush_delay_min.round(2),
        "age": age.round(1),
        "baseline_icdas": baseline_icdas.astype(int),
    })
    return df


def _check_proportions(series, choices, target_probs, label, tol_pp=5):
    counts = series.value_counts(normalize=True)
    for choice, target in zip(choices, target_probs):
        actual = counts.get(choice, 0.0) * 100
        target_pct = target * 100
        if abs(actual - target_pct) > tol_pp:
            print(
                f"  [warn] {label}={choice}: actual {actual:.1f}% vs target "
                f"{target_pct:.1f}% (>{tol_pp}pp off, single finite sample)"
            )


def validate_cohort(df):
    assert df.isna().sum().sum() == 0, "NaNs present in generated cohort"
    assert len(df) == N, f"expected {N} rows, got {len(df)}"
    assert df["person_id"].is_unique, "duplicate person_id values"

    ranges = {
        "brush_time_sec": (30, 240),
        "brush_force_N": (0.3, 5.0),
        "buccal_coverage_pct": (0, 100),
        "occlusal_coverage_pct": (0, 100),
        "lingual_coverage_pct": (0, 100),
        "sugar_freq_day": (1, 10),
        "brush_delay_min": (0, 180),
        "age": (18, 70),
    }
    for col, (lo, hi) in ranges.items():
        assert df[col].min() >= lo and df[col].max() <= hi, (
            f"{col} out of clip range [{lo},{hi}]: "
            f"got [{df[col].min()},{df[col].max()}]"
        )

    assert set(df["brush_freq"].unique()) <= set(BRUSH_FREQ_CHOICES)
    assert set(df["food_type"].unique()) <= set(FOOD_TYPE_CHOICES)
    assert set(df["baseline_icdas"].unique()) <= set(BASELINE_ICDAS_CHOICES)

    print("Validation checks passed (hard asserts). Soft proportion checks:")
    _check_proportions(df["brush_freq"], BRUSH_FREQ_CHOICES, BRUSH_FREQ_PROBS, "brush_freq")
    _check_proportions(df["food_type"], FOOD_TYPE_CHOICES, FOOD_TYPE_PROBS, "food_type")
    _check_proportions(df["baseline_icdas"], BASELINE_ICDAS_CHOICES, BASELINE_ICDAS_PROBS, "baseline_icdas")


def main():
    rng = make_rng(SEED)
    df = build_cohort(rng, N)
    validate_cohort(df)

    print("\ncohort_raw.csv summary:")
    print(df.describe(include="all"))

    out_path = "data/cohort_raw.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
