# OralGuard 코호트 통계 분석 (N=300)

## 1. 상관관계 분석 (Pearson r, N=300)

DOHI와의 상관계수 (절댓값 순):
  DRS                      r = -0.970
  brush_delay_min          r = +0.594
  BES                      r = +0.238
  lingual_coverage_pct     r = +0.210
  sugar_freq_day           r = -0.208
  brush_freq               r = +0.130
  age                      r = +0.116
  brush_time_sec           r = +0.088
  brush_force_N            r = +0.078
  ARS                      r = +0.014
  occlusal_coverage_pct    r = +0.013
  buccal_coverage_pct      r = +0.010

## 2. K-means 군집 분석 (k 선택 검증)

k별 실루엣 점수 (1에 가까울수록 군집이 잘 분리됨):
  k=2: silhouette = 0.432
  k=3: silhouette = 0.526  <- 채택
  k=4: silhouette = 0.573
  k=5: silhouette = 0.555
  k=6: silhouette = 0.544

k=3 최종 실루엣 점수: 0.526 (양호 — 0.5 이상이면 뚜렷, 0.2~0.5는 약하지만 유의미한 구조로 통상 해석)

## 3. 집단별 DOHI 비교 (연령대 / 양치 빈도)

[연령대] 그룹별 표본 수: 18-29=101, 30-39=94, 40-49=71, 50-59=29, 60-70=5
[연령대] 일원분산분석(ANOVA): F=1.241, p=0.2937 (유의하지 않음)
[연령대] Kruskal-Wallis(비모수): H=3.179, p=0.5283 (유의하지 않음)
[양치빈도] 그룹별 표본 수: 1=67, 2=183, 3=50
[양치빈도] 일원분산분석(ANOVA): F=2.929, p=0.0550 (유의하지 않음)
[양치빈도] Kruskal-Wallis(비모수): H=49.563, p=0.0000 (유의함 (p<0.05))
[양치빈도] Tukey HSD 사후검정 (ANOVA 또는 Kruskal-Wallis 중 하나라도 유의하여 실행):
Multiple Comparison of Means - Tukey HSD, FWER=0.05 
====================================================
group1 group2 meandiff p-adj   lower   upper  reject
----------------------------------------------------
     1      2   3.8575 0.0865 -0.4177  8.1328  False
     1      3   5.1041  0.082 -0.4912 10.6995  False
     2      3   1.2466 0.8123 -3.5312  6.0243  False
----------------------------------------------------

### 다중회귀분석: DOHI ~ 행동 변수 전체

R-squared = 0.483 (adj. 0.465), N=300
유의한 예측변수 (p<0.05):
  C(brush_freq)[T.2]           coef=+3.303  p=0.0157
  C(brush_freq)[T.3]           coef=+5.834  p=0.0011
  age                          coef=+0.116  p=0.0171
  sugar_freq_day               coef=-1.318  p=0.0008
  brush_time_sec               coef=+0.045  p=0.0074
  brush_delay_min              coef=+0.268  p=0.0000
  lingual_coverage_pct         coef=+0.154  p=0.0000
(전체 회귀 결과표는 research/regression_summary.txt에 저장됨)

## 4. Daily_DRS 공식 대안 비교

DRS를 계산하는 4가지 방식을 각각 대입해, ARS/BES는 그대로 두고 DOHI를 재계산.
분리력 평가 기준: 실제 임상 결과 변수인 'progressed'(ICDAS 진행 여부) 이분류를
얼마나 잘 구분해내는지를 ROC-AUC, Cohen's d, Kolmogorov-Smirnov 통계량으로 평가.

공식별 분포 요약 (0-100 스케일을 실제로 얼마나 쓰는지):
  v1_원본 (선형 근사식)           mean= 13.48  std= 27.57  min=  0.00  max=100.00
  v2_실측 pH<5.5 시간          mean=  8.77  std= 10.51  min=  0.00  max= 42.00
  v3_실측 AUC(깊이 가중)         mean= 23.10  std= 35.68  min=  0.00  max=100.00
  v4_로지스틱회귀 위험도            mean= 16.67  std= 11.94  min=  0.97  max= 75.20

[v1_원본 (선형 근사식)]
  ROC-AUC (DRS 단독, progressed 예측)   = 0.7075
  ROC-AUC (재계산 DOHI, progressed 예측) = 0.7373
  Cohen's d (진행군 vs 비진행군)          = 1.103 (큰 효과)
  Kolmogorov-Smirnov statistic            = 0.404 (p=0.0000)

[v2_실측 pH<5.5 시간]
  ROC-AUC (DRS 단독, progressed 예측)   = 0.6335
  ROC-AUC (재계산 DOHI, progressed 예측) = 0.6461
  Cohen's d (진행군 vs 비진행군)          = 0.551 (중간 효과)
  Kolmogorov-Smirnov statistic            = 0.208 (p=0.0491)

[v3_실측 AUC(깊이 가중)]
  ROC-AUC (DRS 단독, progressed 예측)   = 0.6295
  ROC-AUC (재계산 DOHI, progressed 예측) = 0.6248
  Cohen's d (진행군 vs 비진행군)          = 0.449 (작은 효과)
  Kolmogorov-Smirnov statistic            = 0.208 (p=0.0491)

[v4_로지스틱회귀 위험도]
  ROC-AUC (DRS 단독, progressed 예측)   = 0.7294
  ROC-AUC (재계산 DOHI, progressed 예측) = 0.7402
  Cohen's d (진행군 vs 비진행군)          = 1.000 (큰 효과)
  Kolmogorov-Smirnov statistic            = 0.376 (p=0.0000)

→ progressed를 가장 잘 분리한 공식: v4_로지스틱회귀 위험도 (AUC=0.7294)
  (worst-case 정규화 기준: 5회 고산성 노출/일 → 60분, AUC 40.5 pH·분)

### 해석상 주의할 점 (순환성)
'progressed'는 원래 baseline_icdas + Bernoulli(진행확률)로 생성되었고, 그 진행확률은
grade(=v1 DRS로 계산된 원본 DOHI에서 파생)에 의해 결정되었다. 즉 v1이 만들어낸
등급이 그대로 'progressed'를 낳았으므로, v1이 다른 공식보다 AUC가 높게 나오는 것은
v1이 '실제로 더 우수한 모델'이어서가 아니라 정답지(y)를 v1 자신이 정의했기 때문일
가능성이 크다 (순환 검증). v2/v3처럼 v1과 무관하게 새로 시뮬레이션한 값이 그럼에도
불구하고 어느 정도 분리력(AUC 0.63, Cohen's d 0.45~0.55)을 보인다는 점은 오히려
고무적 — 진짜 독립적인 신호가 존재한다는 뜻이다. 이 비교를 '어느 공식이 옳은가'가
아니라 '공식 선택이 결과에 얼마나 민감한가'를 보여주는 민감도 분석으로 보고서에
쓰는 것을 추천한다. 아울러 v2/v3의 값 분포(위 mean/std/min/max)가 v1보다 훨씬
좁아 0-100 스케일을 다 쓰지 못하는데, 이는 200분/AUC 상한 정규화 기준이 새 공식의
실제 달성 가능 범위에 맞춰 재보정되지 않았기 때문일 수 있다 — AUC 차이의 일부는
이 스케일 압축 효과일 수 있음.