"""Analyze the scored cohort: distributions, correlations, coverage
heatmap, k-means personas, ROC-AUC, intervention simulations, and
180-day persona trajectories. Writes figures/*.png and
data/cohort_results.json (the data contract consumed by index.html).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# NanumGothic is only guaranteed to exist on machines that installed it
# (e.g. via `apt install fonts-nanum`); pick whichever Korean-capable font
# is actually available so this doesn't silently render tofu boxes on a
# grader's Windows (Malgun Gothic) or Mac (Apple SD Gothic Neo) machine.
_KOREAN_FONT_CANDIDATES = ["NanumGothic", "Malgun Gothic", "Apple SD Gothic Neo", "AppleGothic", "NanumBarunGothic"]
_installed_fonts = {f.name for f in fm.fontManager.ttflist}
_korean_font = next((f for f in _KOREAN_FONT_CANDIDATES if f in _installed_fonts), None)
if _korean_font:
    plt.rcParams["font.family"] = _korean_font
else:
    print("[warning] 한글 폰트를 찾지 못했습니다 — 그래프의 한글 라벨이 깨져 보일 수 있습니다 "
          "(나눔고딕 설치 권장).")
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, roc_curve

from scoring import compute_ars, compute_bes, compute_drs, compute_dohi

ARCHIVE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = ARCHIVE_DIR / "figures"
DATA_DIR = REPO_ROOT / "data"

ACCENTS = {"sky": "#38bdf8", "mint": "#34d399", "red": "#f87171", "amber": "#fbbf24"}


def load_scored():
    return pd.read_csv(DATA_DIR / "cohort_scored.csv")


# ---------------------------------------------------------------- (1)
def analyze_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(df["DOHI"], bins=20, color=ACCENTS["sky"], edgecolor="white")
    axes[0].set_title("DOHI Distribution")
    axes[0].set_xlabel("DOHI")
    axes[0].set_ylabel("Count")

    grade_counts = df["grade"].value_counts().reindex(
        ["Excellent", "Good", "Fair", "Poor"]
    ).fillna(0)
    colors = [ACCENTS["mint"], ACCENTS["sky"], "#fbbf24", ACCENTS["red"]]
    axes[1].pie(grade_counts, labels=grade_counts.index, autopct="%1.1f%%", colors=colors)
    axes[1].set_title("Grade Distribution")

    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/dohi_distribution.png", dpi=120)
    plt.close(fig)

    return {
        "summary": {
            "n": len(df),
            "dohi_mean": round(df["DOHI"].mean(), 2),
            "dohi_std": round(df["DOHI"].std(), 2),
            "grade_distribution": {k: int(v) for k, v in grade_counts.items()},
        },
        "distributions": {
            "dohi": df["DOHI"].round(2).tolist(),
            "drs": df["DRS"].round(2).tolist(),
            "ars": df["ARS"].round(2).tolist(),
            "bes": df["BES"].round(2).tolist(),
        },
    }


# ---------------------------------------------------------------- (2)
def analyze_correlation(df):
    cols = ["DRS", "ARS", "BES", "DOHI", "brush_time_sec", "brush_force_N"]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation Matrix")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/correlation_heatmap.png", dpi=120)
    plt.close(fig)

    return {
        "correlations": {
            "matrix": corr.round(3).values.tolist(),
            "labels": cols,
        }
    }


# ---------------------------------------------------------------- (3)
def analyze_coverage_heatmap(df, seed=342):
    rng = np.random.default_rng(seed)
    n = len(df)
    jitter = rng.uniform(-8, 8, size=(n, 18))  # cols 0-7 buccal, 8-11 occlusal, 12-17 lingual

    buccal_agg = df["buccal_coverage_pct"].to_numpy()[:, None]
    occlusal_agg = df["occlusal_coverage_pct"].to_numpy()[:, None]
    lingual_agg = df["lingual_coverage_pct"].to_numpy()[:, None]

    buccal_zones = np.clip(buccal_agg + jitter[:, 0:8], 0, 100)
    occlusal_zones = np.clip(occlusal_agg + jitter[:, 8:12], 0, 100)
    lingual_zones = np.clip(lingual_agg + jitter[:, 12:18], 0, 100)

    buccal_mean = buccal_zones.mean(axis=0)
    occlusal_mean = occlusal_zones.mean(axis=0)
    lingual_mean = lingual_zones.mean(axis=0)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    rows = [
        np.pad(buccal_mean, (0, 10), constant_values=np.nan),
        np.pad(occlusal_mean, (0, 14), constant_values=np.nan),
        np.pad(lingual_mean, (0, 12), constant_values=np.nan),
    ]
    sns.heatmap(rows, annot=False, cmap="RdYlGn", vmin=0, vmax=100,
                yticklabels=["Buccal", "Occlusal", "Lingual"], ax=ax, cbar_kws={"label": "Coverage %"})
    ax.set_title("18-Zone Coverage Heatmap (cohort mean, jittered)")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/coverage_heatmap.png", dpi=120)
    plt.close(fig)

    return {
        "coverage_heatmap": {
            "buccal": buccal_mean.round(2).tolist(),
            "occlusal": occlusal_mean.round(2).tolist(),
            "lingual": lingual_mean.round(2).tolist(),
        }
    }


# ---------------------------------------------------------------- (4)
PERSONA_NAME_MAP = {"drs": "당과다형", "ars": "과압마모형", "bes_inv": "저위생형"}


def _name_persona(z_drs, z_ars, z_bes_inv, used_names):
    scores = {"drs": z_drs, "ars": z_ars, "bes_inv": z_bes_inv}
    dominant = max(scores, key=scores.get)
    if scores[dominant] > 0.3:
        name = PERSONA_NAME_MAP[dominant]
    else:
        name = "균형형"
    if name in used_names:
        suffix = chr(ord("A") + used_names[name])
        used_names[name] += 1
        return f"{name} {suffix}"
    used_names[name] = 1
    return name


def analyze_personas(df, seed=42, n_clusters=4):
    # Rank-based (percentile) features for clustering only, not for scoring.
    # DRS/ARS/BES formulas/distributions are untouched; this only prevents a
    # single extreme outlier (e.g. a near-constant-zero ARS with one large
    # value) from dominating standardized-distance clustering and forming a
    # degenerate n=1 cluster.
    #
    # n_clusters=4 (not 3): at k=3 every cluster ends up dominated by one of
    # DRS/ARS/BES, so there is no low-risk-on-all-axes "균형형" persona to
    # contrast against — k=4 surfaces one (confirmed via silhouette score,
    # which is also slightly higher at k=4 than k=3), giving the comparison
    # a genuine good-outcome example alongside the three risk personas.
    x_scaled = df[["DRS", "ARS", "BES"]].rank(pct=True).to_numpy()

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_labels = km.fit_predict(x_scaled)

    pop_mean = df[["DRS", "ARS", "BES"]].mean()
    pop_std = df[["DRS", "ARS", "BES"]].std()

    used_names = {}
    personas = []
    for c in range(n_clusters):
        mask = cluster_labels == c
        cluster_mean = df.loc[mask, ["DRS", "ARS", "BES"]].mean()

        z_drs = (cluster_mean["DRS"] - pop_mean["DRS"]) / pop_std["DRS"]
        z_ars = (cluster_mean["ARS"] - pop_mean["ARS"]) / pop_std["ARS"]
        # BES is a badness scale (higher = worse), same direction as DRS/ARS,
        # so "저위생형" (poor-hygiene persona) is now high-BES, not low-BES.
        z_bes_inv = (cluster_mean["BES"] - pop_mean["BES"]) / pop_std["BES"]

        name = _name_persona(z_drs, z_ars, z_bes_inv, used_names)

        personas.append({
            "name": name,
            "n": int(mask.sum()),
            "dohi_mean": round(df.loc[mask, "DOHI"].mean(), 2),
            "drs_mean": round(cluster_mean["DRS"], 2),
            "ars_mean": round(cluster_mean["ARS"], 2),
            "bes_mean": round(cluster_mean["BES"], 2),
            "_cluster_id": c,
        })

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors_list = [ACCENTS["sky"], ACCENTS["mint"], ACCENTS["red"], ACCENTS["amber"]]
    for c in range(n_clusters):
        mask = cluster_labels == c
        axes[0].scatter(df.loc[mask, "DRS"], df.loc[mask, "ARS"],
                         s=df.loc[mask, "BES"], alpha=0.6, color=colors_list[c],
                         label=personas[c]["name"])
    axes[0].set_xlabel("DRS")
    axes[0].set_ylabel("ARS")
    axes[0].set_title("Personas (DRS vs ARS, size=BES)")
    axes[0].legend()

    axes[1].bar([p["name"] for p in personas], [p["dohi_mean"] for p in personas],
                color=colors_list)
    axes[1].set_title("Mean DOHI by Persona")
    axes[1].set_ylabel("DOHI")

    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/persona_clusters.png", dpi=120)
    plt.close(fig)

    return personas, cluster_labels, km, x_scaled


# ---------------------------------------------------------------- (5)
def analyze_roc(df):
    y_true = df["progressed"].astype(int)

    drs_auc = roc_auc_score(y_true, df["DRS"])
    bes_auc = roc_auc_score(y_true, df["BES"])
    dohi_auc = roc_auc_score(y_true, 100 - df["DOHI"])

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for label, score, color in [
        ("DRS", df["DRS"], ACCENTS["red"]),
        ("BES", df["BES"], "#fbbf24"),
        ("100-DOHI", 100 - df["DOHI"], ACCENTS["sky"]),
    ]:
        fpr, tpr, _ = roc_curve(y_true, score)
        auc = roc_auc_score(y_true, score)
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})", color=color)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC: Predicting ICDAS Progression")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/roc_curves.png", dpi=120)
    plt.close(fig)

    return {
        "roc": {
            "drs_auc": round(drs_auc, 4),
            "bes_auc": round(bes_auc, 4),
            "dohi_auc": round(dohi_auc, 4),
        }
    }


# ---------------------------------------------------------------- (6)
def _score_row(row_dict):
    row_df = pd.DataFrame([row_dict])
    drs = compute_drs(row_df).iloc[0]
    ars = compute_ars(row_df).iloc[0]
    bes = compute_bes(row_df).iloc[0]
    dohi = compute_dohi(pd.Series([drs]), pd.Series([ars]), pd.Series([bes])).iloc[0]
    return {
        "DRS": round(float(drs), 2),
        "ARS": round(float(ars), 2),
        "BES": round(float(bes), 2),
        "DOHI": round(float(dohi), 2),
    }


def analyze_interventions(df):
    baseline = {
        "brush_time_sec": df["brush_time_sec"].mean(),
        "brush_force_N": df["brush_force_N"].mean(),
        "brush_freq": int(df["brush_freq"].mode()[0]),
        "buccal_coverage_pct": df["buccal_coverage_pct"].mean(),
        "occlusal_coverage_pct": df["occlusal_coverage_pct"].mean(),
        "lingual_coverage_pct": df["lingual_coverage_pct"].mean(),
        "sugar_freq_day": df["sugar_freq_day"].mean(),
        "food_type": df["food_type"].mode()[0],
        "brush_delay_min": df["brush_delay_min"].mean(),
    }

    def make_scenario(overrides):
        row = dict(baseline)
        row.update(overrides)
        return row

    scenarios = {
        "S1": {
            "before": make_scenario({"food_type": "high_acid", "brush_delay_min": 0}),
            "after": make_scenario({"food_type": "high_acid", "brush_delay_min": 30}),
        },
        "S2": {
            "before": make_scenario({"brush_time_sec": 45}),
            "after": make_scenario({"brush_time_sec": 120}),
        },
        "S3": {
            "before": make_scenario({"brush_force_N": 3.0}),
            "after": make_scenario({"brush_force_N": 1.5}),
        },
    }

    results = {}
    for key, scen in scenarios.items():
        before = _score_row(scen["before"])
        after = _score_row(scen["after"])
        results[key] = {
            "before": before,
            "after": after,
            "delta_dohi": round(after["DOHI"] - before["DOHI"], 2),
        }

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    metric_map = {"S1": ["DRS", "ARS", "DOHI"], "S2": ["BES", "DOHI"], "S3": ["ARS", "DOHI"]}
    for ax, (key, metrics) in zip(axes, metric_map.items()):
        before_vals = [results[key]["before"][m] for m in metrics]
        after_vals = [results[key]["after"][m] for m in metrics]
        x = np.arange(len(metrics))
        width = 0.35
        ax.bar(x - width / 2, before_vals, width, label="Before", color=ACCENTS["red"])
        ax.bar(x + width / 2, after_vals, width, label="After", color=ACCENTS["mint"])
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_title(key)
        ax.legend()

    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/intervention_effects.png", dpi=120)
    plt.close(fig)

    return {"interventions": results}


# ---------------------------------------------------------------- (7)
def build_persona_trajectories(df, personas, cluster_labels, km, x_scaled, days=180):
    for persona in personas:
        c = persona["_cluster_id"]
        mask = cluster_labels == c
        idxs = np.where(mask)[0]
        dists = np.linalg.norm(x_scaled[mask] - km.cluster_centers_[c], axis=1)
        rep_idx = idxs[np.argmin(dists)]
        mu0 = df.iloc[rep_idx]["DOHI"]

        slope = 0.008 if persona["name"].startswith("균형형") else -0.015
        rng = np.random.default_rng(242 + c)

        z = rng.normal(0, 1, days)
        phi, sigma = 0.85, 2.5
        e = np.zeros(days)
        e[0] = sigma * z[0]
        for t in range(1, days):
            e[t] = phi * e[t - 1] + sigma * z[t]

        trajectory = np.clip(mu0 + slope * np.arange(days) + e, 0, 100)
        persona["trajectory"] = trajectory.round(2).tolist()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors_list = [ACCENTS["sky"], ACCENTS["mint"], ACCENTS["red"], ACCENTS["amber"]]
    for persona, color in zip(personas, colors_list):
        ax.plot(range(days), persona["trajectory"], label=persona["name"], color=color)
    ax.set_xlabel("Day")
    ax.set_ylabel("DOHI")
    ax.set_title("180-Day Persona DOHI Trajectories")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/persona_trajectories.png", dpi=120)
    plt.close(fig)

    for persona in personas:
        del persona["_cluster_id"]

    return personas


# ---------------------------------------------------------------- (8)
def build_demo_patient(df, seed=442):
    demo_idx = (df["DOHI"] - df["DOHI"].mean()).abs().idxmin()
    row = df.loc[demo_idx]

    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-8, 8, size=18)
    buccal = np.clip(row["buccal_coverage_pct"] + jitter[0:8], 0, 100)
    occlusal = np.clip(row["occlusal_coverage_pct"] + jitter[8:12], 0, 100)
    lingual = np.clip(row["lingual_coverage_pct"] + jitter[12:18], 0, 100)

    return {
        "person_id": row["person_id"],
        "brush_time_sec": round(float(row["brush_time_sec"]), 2),
        "brush_force_N": round(float(row["brush_force_N"]), 2),
        "brush_freq": int(row["brush_freq"]),
        "buccal_coverage_pct": round(float(row["buccal_coverage_pct"]), 2),
        "occlusal_coverage_pct": round(float(row["occlusal_coverage_pct"]), 2),
        "lingual_coverage_pct": round(float(row["lingual_coverage_pct"]), 2),
        "sugar_freq_day": round(float(row["sugar_freq_day"]), 2),
        "food_type": row["food_type"],
        "brush_delay_min": round(float(row["brush_delay_min"]), 2),
        "DRS": round(float(row["DRS"]), 2),
        "ARS": round(float(row["ARS"]), 2),
        "BES": round(float(row["BES"]), 2),
        "DOHI": round(float(row["DOHI"]), 2),
        "grade": row["grade"],
        "zone_coverage": {
            "buccal": buccal.round(2).tolist(),
            "occlusal": occlusal.round(2).tolist(),
            "lingual": lingual.round(2).tolist(),
        },
    }


def analyze_icdas_progression(df):
    by_grade = df.groupby("grade")["progressed"].mean() * 100
    by_grade = by_grade.reindex(["Excellent", "Good", "Fair", "Poor"])
    return {
        "icdas_progression": {
            "progressed_pct": round(df["progressed"].mean() * 100, 2),
            "by_grade": {k: round(float(v), 2) if pd.notna(v) else None
                         for k, v in by_grade.items()},
        }
    }


def main():
    df = load_scored()
    results = {}

    results.update(analyze_distribution(df))
    results.update(analyze_correlation(df))
    results.update(analyze_coverage_heatmap(df))

    personas, cluster_labels, km, x_scaled = analyze_personas(df)
    results.update(analyze_roc(df))
    results.update(analyze_interventions(df))

    personas = build_persona_trajectories(df, personas, cluster_labels, km, x_scaled)
    results["personas"] = personas

    results.update(analyze_icdas_progression(df))
    results["demo_patient"] = build_demo_patient(df)

    out_path = f"{DATA_DIR}/cohort_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Wrote {out_path}")
    print("Personas:", [(p["name"], p["n"]) for p in personas])
    print("ROC AUCs:", results["roc"])
    print("Demo patient:", results["demo_patient"]["person_id"], results["demo_patient"]["grade"])


if __name__ == "__main__":
    main()
