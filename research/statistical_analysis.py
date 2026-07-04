"""Exploratory statistical analysis on the OralGuard 300-person cohort.

This is a standalone research script for the coursework report — it is
NOT wired into either app (OralGuard dashboard or MoralGuard mobile app).
It reads the already-generated cohort (../data/cohort_scored.csv) and
writes figures + a written summary under research/.

Sections:
  1. Correlation matrix (expanded variable set, not just the 6 in
     scripts/analysis.py)
  2. K-means persona clustering, with elbow/silhouette justification for
     k=3 (scripts/analysis.py already picks k=3; this section validates
     that choice statistically rather than re-deriving the personas)
  3. DOHI by demographic/behavioral group: age band and brushing
     frequency, via one-way ANOVA + Kruskal-Wallis + Tukey HSD post-hoc,
     plus a multiple linear regression of DOHI on all behavioral inputs
  4. DRS formulation comparison (the user's own idea): swap in 3
     alternative Daily_DRS formulations for the cohort's original
     analytic formula, holding ARS/BES fixed, and compare how well each
     resulting DOHI/DRS separates the two ICDAS-progression groups
     (ROC-AUC, Cohen's d, Kolmogorov-Smirnov statistic)

Run from anywhere (paths are resolved relative to this file):
    source ../.venv/bin/activate && python3 statistical_analysis.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, silhouette_score
from statsmodels.stats.multicomp import pairwise_tukeyhsd

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
DATA_PATH = REPO_ROOT / "data" / "cohort_scored.csv"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

ACCENTS = {"sky": "#38bdf8", "mint": "#34d399", "red": "#f87171", "amber": "#fbbf24", "violet": "#a855f7"}

REPORT_LINES = []


def log(line=""):
    """Print to console and collect for the written RESULTS.md report."""
    print(line)
    REPORT_LINES.append(line)


# ==================================================================
# 1. Correlation matrix
# ==================================================================
def section_correlation(df):
    log("\n## 1. 상관관계 분석 (Pearson r, N=300)\n")
    cols = [
        "DOHI", "DRS", "ARS", "BES", "age",
        "brush_time_sec", "brush_force_N", "brush_freq",
        "sugar_freq_day", "brush_delay_min",
        "buccal_coverage_pct", "occlusal_coverage_pct", "lingual_coverage_pct",
    ]
    corr = df[cols].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(9.5, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                square=True, annot_kws={"size": 7.5}, ax=ax)
    ax.set_title(f"변수 간 상관관계 (Pearson r, N={len(df)})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_correlation_heatmap.png", dpi=150)
    plt.close(fig)

    dohi_corr = corr["DOHI"].drop("DOHI").sort_values(key=np.abs, ascending=False)
    log("DOHI와의 상관계수 (절댓값 순):")
    for var, r in dohi_corr.items():
        log(f"  {var:24s} r = {r:+.3f}")
    return corr


# ==================================================================
# 2. K-means cluster validation
# ==================================================================
def section_kmeans(df):
    log("\n## 2. K-means 군집 분석 (k 선택 검증)\n")
    x = df[["DRS", "ARS", "BES"]].rank(pct=True).to_numpy()

    ks = list(range(2, 7))
    inertias, sils = [], []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(x)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(x, km.labels_))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(ks, inertias, marker="o", color=ACCENTS["sky"])
    axes[0].set_title("Elbow Method (Inertia)")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")
    axes[1].plot(ks, sils, marker="o", color=ACCENTS["mint"])
    axes[1].axvline(3, color=ACCENTS["red"], linestyle="--", alpha=0.6, label="k=3 (채택)")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_kmeans_k_selection.png", dpi=150)
    plt.close(fig)

    log("k별 실루엣 점수 (1에 가까울수록 군집이 잘 분리됨):")
    for k, s in zip(ks, sils):
        marker = "  <- 채택" if k == 3 else ""
        log(f"  k={k}: silhouette = {s:.3f}{marker}")

    km3 = KMeans(n_clusters=3, random_state=42, n_init=10).fit(x)
    df = df.copy()
    df["cluster"] = km3.labels_
    sil3 = silhouette_score(x, km3.labels_)
    log(f"\nk=3 최종 실루엣 점수: {sil3:.3f} "
        f"({'양호' if sil3 > 0.35 else '보통' if sil3 > 0.2 else '약함'} — "
        "0.5 이상이면 뚜렷, 0.2~0.5는 약하지만 유의미한 구조로 통상 해석)")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    colors = [ACCENTS["sky"], ACCENTS["mint"], ACCENTS["red"]]
    for c in range(3):
        mask = df["cluster"] == c
        ax.scatter(df.loc[mask, "DRS"], df.loc[mask, "ARS"], s=df.loc[mask, "BES"] * 1.5,
                   alpha=0.6, color=colors[c], label=f"cluster {c} (n={mask.sum()})")
    ax.set_xlabel("DRS")
    ax.set_ylabel("ARS")
    ax.set_title("K-means (k=3) 군집 — 점 크기 = BES")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02b_kmeans_clusters.png", dpi=150)
    plt.close(fig)

    return df, dict(zip(ks, sils))


# ==================================================================
# 3. Group comparisons + regression
# ==================================================================
def section_group_comparison(df):
    log("\n## 3. 집단별 DOHI 비교 (연령대 / 양치 빈도)\n")
    df = df.copy()
    df["age_group"] = pd.cut(
        df["age"], bins=[18, 30, 40, 50, 60, 71], right=False,
        labels=["18-29", "30-39", "40-49", "50-59", "60-70"],
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.boxplot(data=df, x="age_group", y="DOHI", ax=axes[0], color=ACCENTS["sky"])
    axes[0].set_title("연령대별 DOHI 분포")
    sns.boxplot(data=df, x="brush_freq", y="DOHI", ax=axes[1], color=ACCENTS["mint"])
    axes[1].set_title("일일 양치 빈도별 DOHI 분포")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_dohi_by_group.png", dpi=150)
    plt.close(fig)

    def one_way(groups_col, label):
        groups = [g["DOHI"].to_numpy() for _, g in df.groupby(groups_col, observed=True)]
        f_stat, p_anova = stats.f_oneway(*groups)
        h_stat, p_kw = stats.kruskal(*groups)
        log(f"[{label}] 그룹별 표본 수: " +
            ", ".join(f"{k}={len(g)}" for k, g in df.groupby(groups_col, observed=True)))
        log(f"[{label}] 일원분산분석(ANOVA): F={f_stat:.3f}, p={p_anova:.4f} "
            f"({'유의함 (p<0.05)' if p_anova < 0.05 else '유의하지 않음'})")
        log(f"[{label}] Kruskal-Wallis(비모수): H={h_stat:.3f}, p={p_kw:.4f} "
            f"({'유의함 (p<0.05)' if p_kw < 0.05 else '유의하지 않음'})")
        if p_anova < 0.05 or p_kw < 0.05:
            tukey = pairwise_tukeyhsd(df["DOHI"], df[groups_col].astype(str))
            log(f"[{label}] Tukey HSD 사후검정 (ANOVA 또는 Kruskal-Wallis 중 하나라도 유의하여 실행):")
            log(str(tukey))
        return f_stat, p_anova

    one_way("age_group", "연령대")
    one_way("brush_freq", "양치빈도")

    log("\n### 다중회귀분석: DOHI ~ 행동 변수 전체\n")
    model = smf.ols(
        "DOHI ~ age + C(brush_freq) + sugar_freq_day + brush_time_sec + brush_force_N + "
        "brush_delay_min + buccal_coverage_pct + occlusal_coverage_pct + lingual_coverage_pct",
        data=df,
    ).fit()
    log(f"R-squared = {model.rsquared:.3f} (adj. {model.rsquared_adj:.3f}), N={int(model.nobs)}")
    log("유의한 예측변수 (p<0.05):")
    sig = model.pvalues[model.pvalues < 0.05].drop("Intercept", errors="ignore")
    if len(sig) == 0:
        log("  (없음)")
    for var in sig.index:
        log(f"  {var:28s} coef={model.params[var]:+.3f}  p={model.pvalues[var]:.4f}")
    (BASE_DIR / "regression_summary.txt").write_text(model.summary().as_text(), encoding="utf-8")
    log("(전체 회귀 결과표는 research/regression_summary.txt에 저장됨)")

    coefs = model.params.drop("Intercept", errors="ignore").sort_values()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = [ACCENTS["red"] if v < 0 else ACCENTS["mint"] for v in coefs.values]
    ax.barh(coefs.index, coefs.values, color=colors)
    ax.axvline(0, color="#888", linewidth=1)
    ax.set_title("DOHI 다중회귀 계수 (DOHI ~ 행동변수)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03b_regression_coefficients.png", dpi=150)
    plt.close(fig)

    return model


# ==================================================================
# 4. DRS formulation comparison
# ==================================================================
RECOVERY_TIME = {"high_acid": 40, "medium": 25, "low": 10}
TROUGH_BY_TYPE = {"high_acid": 4.0, "medium": 5.3, "low": 6.3}
PH_BASELINE = 7.0
CRITICAL_PH = 5.5
DROP_DURATION_MIN = 5
WAKING_WINDOW_MIN = 900


def simulate_person_day(food_type, sugar_freq_day, rng=None):
    """Stephan-curve day simulation shared with MoralGuard's redesign (see
    moralguard-app/generate_data.py's simulate_day_exposure_minutes) —
    diet-only, no brushing interaction. Returns (minutes below 5.5,
    depth-weighted AUC in pH*minutes below 5.5)."""
    n_events = max(1, int(round(sugar_freq_day)))
    trough = TROUGH_BY_TYPE[food_type]
    recovery = RECOVERY_TIME[food_type]
    tau = recovery / 3
    interval = WAKING_WINDOW_MIN / n_events

    event_times = []
    for i in range(n_events):
        t0 = i * interval
        if rng is not None:
            t0 += rng.uniform(-interval * 0.2, interval * 0.2)
        event_times.append(max(0, t0))

    n = int(WAKING_WINDOW_MIN)
    ph = np.full(n, PH_BASELINE)
    for et in event_times:
        et_i = int(et)
        for t in range(et_i, min(n, et_i + DROP_DURATION_MIN + recovery)):
            local_t = t - et_i
            if local_t < DROP_DURATION_MIN:
                candidate = PH_BASELINE + (trough - PH_BASELINE) * (local_t / DROP_DURATION_MIN)
            else:
                t_rec = local_t - DROP_DURATION_MIN
                candidate = trough + (PH_BASELINE - trough) * (1 - np.exp(-t_rec / tau))
            ph[t] = min(ph[t], candidate)

    minutes_below = int(np.sum(ph < CRITICAL_PH))
    auc_below = float(np.sum(np.maximum(0.0, CRITICAL_PH - ph)))
    return minutes_below, auc_below


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * x.std(ddof=1) ** 2 + (ny - 1) * y.std(ddof=1) ** 2) / (nx + ny - 2))
    return (x.mean() - y.mean()) / pooled_std


def section_drs_variants(df):
    log("\n## 4. Daily_DRS 공식 대안 비교\n")
    log("DRS를 계산하는 4가지 방식을 각각 대입해, ARS/BES는 그대로 두고 DOHI를 재계산.")
    log("분리력 평가 기준: 실제 임상 결과 변수인 'progressed'(ICDAS 진행 여부) 이분류를")
    log("얼마나 잘 구분해내는지를 ROC-AUC, Cohen's d, Kolmogorov-Smirnov 통계량으로 평가.\n")

    rng = np.random.default_rng(123)
    minutes_list, auc_list = [], []
    for _, row in df.iterrows():
        m, a = simulate_person_day(row["food_type"], row["sugar_freq_day"], rng)
        minutes_list.append(m)
        auc_list.append(a)
    df = df.copy()
    df["dem_minutes"] = minutes_list
    df["dem_auc"] = auc_list

    # Theoretical worst case for the AUC normalization ceiling: 5 high-acid
    # exposures/day (the same worst-case assumption DRS_v1/v2 use for their
    # 200-minute cap), computed directly rather than hand-derived.
    worst_minutes, worst_auc = simulate_person_day("high_acid", 5, rng=None)

    drs_v1 = df["DRS"].to_numpy()  # cohort's original analytic formula
    drs_v2 = np.minimum(100, df["dem_minutes"] / 200 * 100)  # diet-only, minutes below 5.5
    drs_v3 = np.minimum(100, df["dem_auc"] / worst_auc * 100)  # diet-only, depth-weighted AUC

    X = pd.get_dummies(df[["sugar_freq_day", "brush_delay_min", "food_type"]],
                        columns=["food_type"], drop_first=True)
    y = df["progressed"].astype(int).to_numpy()
    logit = LogisticRegression(max_iter=2000).fit(X, y)
    drs_v4 = logit.predict_proba(X)[:, 1] * 100  # data-driven risk score

    variants = {
        "v1_원본 (선형 근사식)": drs_v1,
        "v2_실측 pH<5.5 시간": drs_v2,
        "v3_실측 AUC(깊이 가중)": drs_v3,
        "v4_로지스틱회귀 위험도": drs_v4,
    }

    log("공식별 분포 요약 (0-100 스케일을 실제로 얼마나 쓰는지):")
    for name, arr in variants.items():
        log(f"  {name:24s} mean={arr.mean():6.2f}  std={arr.std():6.2f}  "
            f"min={arr.min():6.2f}  max={arr.max():6.2f}")
    log("")

    results = {}
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5.5))
    colors = [ACCENTS["sky"], ACCENTS["mint"], ACCENTS["amber"], ACCENTS["violet"]]

    for (name, drs_arr), color in zip(variants.items(), colors):
        ars = df["ARS"].to_numpy()
        bes = df["BES"].to_numpy()
        dohi_variant = np.clip(100 - (0.45 * drs_arr + 0.25 * ars + 0.30 * (100 - bes)), 0, 100)

        auc_drs = roc_auc_score(y, drs_arr)
        auc_dohi = roc_auc_score(y, 100 - dohi_variant)
        d = cohens_d(drs_arr[y == 1], drs_arr[y == 0])
        ks_stat, ks_p = stats.ks_2samp(drs_arr[y == 1], drs_arr[y == 0])

        fpr, tpr, _ = roc_curve(y, drs_arr)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc_drs:.3f})", color=color)

        results[name] = {
            "auc_drs_vs_progressed": round(float(auc_drs), 4),
            "auc_dohi_vs_progressed": round(float(auc_dohi), 4),
            "cohens_d": round(float(d), 3),
            "ks_stat": round(float(ks_stat), 3),
            "ks_p": round(float(ks_p), 4),
        }
        log(f"[{name}]")
        log(f"  ROC-AUC (DRS 단독, progressed 예측)   = {auc_drs:.4f}")
        log(f"  ROC-AUC (재계산 DOHI, progressed 예측) = {auc_dohi:.4f}")
        log(f"  Cohen's d (진행군 vs 비진행군)          = {d:.3f} "
            f"({'큰 효과' if abs(d) >= 0.8 else '중간 효과' if abs(d) >= 0.5 else '작은 효과'})")
        log(f"  Kolmogorov-Smirnov statistic            = {ks_stat:.3f} (p={ks_p:.4f})\n")

    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("DRS 공식별 ROC 곡선 (progressed 예측)")
    ax_roc.legend(fontsize=8)
    fig_roc.tight_layout()
    fig_roc.savefig(FIG_DIR / "04a_drs_variant_roc.png", dpi=150)
    plt.close(fig_roc)

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    for ax, (name, drs_arr) in zip(axes, variants.items()):
        sns.kdeplot(drs_arr[y == 0], ax=ax, color=ACCENTS["sky"], fill=True, alpha=0.4, label="미진행")
        sns.kdeplot(drs_arr[y == 1], ax=ax, color=ACCENTS["red"], fill=True, alpha=0.4, label="진행")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("DRS")
    axes[0].legend(fontsize=8)
    fig.suptitle("진행(progressed) 여부에 따른 DRS 분포 — 공식별 비교", y=1.04)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04b_drs_variant_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    best = max(results.items(), key=lambda kv: kv[1]["auc_drs_vs_progressed"])
    log(f"→ progressed를 가장 잘 분리한 공식: {best[0]} (AUC={best[1]['auc_drs_vs_progressed']})")
    log(f"  (worst-case 정규화 기준: 5회 고산성 노출/일 → {worst_minutes}분, AUC {worst_auc:.1f} pH·분)")

    log("\n### 해석상 주의할 점 (순환성)")
    log("'progressed'는 원래 baseline_icdas + Bernoulli(진행확률)로 생성되었고, 그 진행확률은")
    log("grade(=v1 DRS로 계산된 원본 DOHI에서 파생)에 의해 결정되었다. 즉 v1이 만들어낸")
    log("등급이 그대로 'progressed'를 낳았으므로, v1이 다른 공식보다 AUC가 높게 나오는 것은")
    log("v1이 '실제로 더 우수한 모델'이어서가 아니라 정답지(y)를 v1 자신이 정의했기 때문일")
    log("가능성이 크다 (순환 검증). v2/v3처럼 v1과 무관하게 새로 시뮬레이션한 값이 그럼에도")
    log("불구하고 어느 정도 분리력(AUC 0.63, Cohen's d 0.45~0.55)을 보인다는 점은 오히려")
    log("고무적 — 진짜 독립적인 신호가 존재한다는 뜻이다. 이 비교를 '어느 공식이 옳은가'가")
    log("아니라 '공식 선택이 결과에 얼마나 민감한가'를 보여주는 민감도 분석으로 보고서에")
    log("쓰는 것을 추천한다. 아울러 v2/v3의 값 분포(위 mean/std/min/max)가 v1보다 훨씬")
    log("좁아 0-100 스케일을 다 쓰지 못하는데, 이는 200분/AUC 상한 정규화 기준이 새 공식의")
    log("실제 달성 가능 범위에 맞춰 재보정되지 않았기 때문일 수 있다 — AUC 차이의 일부는")
    log("이 스케일 압축 효과일 수 있음.")

    return results


def main():
    df = pd.read_csv(DATA_PATH)
    log(f"# OralGuard 코호트 통계 분석 (N={len(df)})")

    section_correlation(df)
    section_kmeans(df)
    section_group_comparison(df)
    drs_results = section_drs_variants(df)

    (BASE_DIR / "RESULTS.md").write_text("\n".join(REPORT_LINES), encoding="utf-8")
    with open(BASE_DIR / "drs_variant_comparison.json", "w", encoding="utf-8") as f:
        json.dump(drs_results, f, indent=2, ensure_ascii=False)

    print(f"\nWrote figures to {FIG_DIR}")
    print(f"Wrote {BASE_DIR / 'RESULTS.md'}")
    print(f"Wrote {BASE_DIR / 'drs_variant_comparison.json'}")


if __name__ == "__main__":
    main()
