# OralGuard 코호트 통계 분석 (N=300)

## 1. 상관관계 분석 (Pearson r, N=300)

DOHI와의 상관계수 (절댓값 순):
  DRS                      r = -0.820
  BES                      r = -0.457
  brush_freq               r = +0.429
  sugar_freq_day           r = -0.301
  ARS                      r = -0.291
  lingual_coverage_pct     r = +0.230
  brush_force_N            r = -0.213
  buccal_coverage_pct      r = +0.105
  brush_delay_min          r = -0.063
  age                      r = -0.049
  occlusal_coverage_pct    r = +0.042
  brush_time_sec           r = +0.021

## 2. K-means 군집 분석 (k 선택 검증)

k별 실루엣 점수 (1에 가까울수록 군집이 잘 분리됨):
  k=2: silhouette = 0.276
  k=3: silhouette = 0.269  <- 채택
  k=4: silhouette = 0.283
  k=5: silhouette = 0.296
  k=6: silhouette = 0.302

k=3 최종 실루엣 점수: 0.269 (보통 — 0.5 이상이면 뚜렷, 0.2~0.5는 약하지만 유의미한 구조로 통상 해석)

## 3. 집단별 DOHI 비교 (연령대 / 양치 빈도)

[연령대] 그룹별 표본 수: 18-29=101, 30-39=94, 40-49=71, 50-59=29, 60-70=5
[연령대] 일원분산분석(ANOVA): F=2.344, p=0.0549 (유의하지 않음)
[연령대] Kruskal-Wallis(비모수): H=5.681, p=0.2243 (유의하지 않음)
[양치빈도] 그룹별 표본 수: 1=67, 2=183, 3=50
[양치빈도] 일원분산분석(ANOVA): F=37.566, p=0.0000 (유의함 (p<0.05))
[양치빈도] Kruskal-Wallis(비모수): H=67.771, p=0.0000 (유의함 (p<0.05))
[양치빈도] Tukey HSD 사후검정 (ANOVA 또는 Kruskal-Wallis 중 하나라도 유의하여 실행):
Multiple Comparison of Means - Tukey HSD, FWER=0.05
===================================================
group1 group2 meandiff p-adj  lower   upper  reject
---------------------------------------------------
     1      2   5.4629    0.0  3.706  7.2198   True
     1      3   7.7501    0.0 5.4506 10.0495   True
     2      3   2.2871 0.0177 0.3237  4.2506   True
---------------------------------------------------

### 다중회귀분석: DOHI ~ 행동 변수 전체

R-squared = 0.369 (adj. 0.347), N=300
유의한 예측변수 (p<0.05):
  C(brush_freq)[T.2]           coef=+4.789  p=0.0000
  C(brush_freq)[T.3]           coef=+7.131  p=0.0000
  sugar_freq_day               coef=-1.070  p=0.0000
  brush_force_N                coef=-2.306  p=0.0000
  buccal_coverage_pct          coef=+0.046  p=0.0195
  lingual_coverage_pct         coef=+0.062  p=0.0000
(전체 회귀 결과표는 research/regression_summary.txt에 저장됨)

## 4. Daily_DRS 공식 대안 비교

DRS를 계산하는 4가지 방식을 각각 대입해, ARS/BES는 그대로 두고 DOHI를 재계산.
분리력 평가 기준: 실제 임상 결과 변수인 'progressed'(ICDAS 진행 여부) 이분류를
얼마나 잘 구분해내는지를 ROC-AUC, Cohen's d, Kolmogorov-Smirnov 통계량으로 평가.

공식별 분포 요약 (0-100 스케일을 실제로 얼마나 쓰는지):
  v1_원본 (선형 근사식)           mean= 13.48  std= 27.57  min=  0.00  max=100.00
  v2_실측 pH<5.5 시간          mean=  8.77  std= 10.51  min=  0.00  max= 42.00
  v3_실측 AUC(깊이 가중)         mean= 23.10  std= 35.68  min=  0.00  max=100.00
  v4_로지스틱회귀 위험도            mean= 13.00  std=  8.09  min=  2.60  max= 46.42

[v1_원본 (선형 근사식)]
  ROC-AUC (DRS 단독, progressed 예측)   = 0.5917
  ROC-AUC (재계산 DOHI, progressed 예측) = 0.6676
  Cohen's d (진행군 vs 비진행군)          = 0.342 (작은 효과)
  Kolmogorov-Smirnov statistic            = 0.179 (p=0.1985)

[v2_실측 pH<5.5 시간]
  ROC-AUC (DRS 단독, progressed 예측)   = 0.6392
  ROC-AUC (재계산 DOHI, progressed 예측) = 0.7207
  Cohen's d (진행군 vs 비진행군)          = 0.733 (중간 효과)
  Kolmogorov-Smirnov statistic            = 0.272 (p=0.0102)

[v3_실측 AUC(깊이 가중)]
  ROC-AUC (DRS 단독, progressed 예측)   = 0.6375
  ROC-AUC (재계산 DOHI, progressed 예측) = 0.6923
  Cohen's d (진행군 vs 비진행군)          = 0.669 (중간 효과)
  Kolmogorov-Smirnov statistic            = 0.272 (p=0.0102)

[v4_로지스틱회귀 위험도]
  ROC-AUC (DRS 단독, progressed 예측)   = 0.6972
  ROC-AUC (재계산 DOHI, progressed 예측) = 0.7460
  Cohen's d (진행군 vs 비진행군)          = 0.788 (중간 효과)
  Kolmogorov-Smirnov statistic            = 0.328 (p=0.0009)

→ progressed를 가장 잘 분리한 공식: v4_로지스틱회귀 위험도 (AUC=0.6972)
  (worst-case 정규화 기준: 5회 고산성 노출/일 → 60분, AUC 40.5 pH·분)

### 해석상 주의할 점 (순환성)
코호트 파이프라인이 session-first 알고리즘(scripts/scoring.py)으로 바뀌면서
cohort_scored.csv의 실제 DRS는 이제 v1(옛 선형 근사식)이 아니라 v2와 동일한 방식
(Stephan 곡선 시뮬레이션 후 pH<5.5 시간 정규화)으로 계산된다. 즉 'progressed'를
낳은 grade/DOHI는 v2 계열 DRS에서 파생된 것이므로, 순환성 우려가 있다면 이제는
v1이 아니라 v2 쪽에 있다 (다만 v2는 서로 다른 rng 시드로 독립 재시뮬레이션한 값이라
완전히 동일하지는 않음 — 약한 잔여 순환성). 반대로 v1(옛 analytic 공식)은 이제 코호트
생성 과정과 무관한 진짜 외부 비교군이 되었고, AUC가 v2/v3/v4보다 낮게 나온 것은
자연스러운 결과다. 가장 중요한 주의점은 v4(로지스틱 회귀)로, 이는 'progressed'를
예측하도록 같은 300명 데이터에 직접 학습(fit)한 뒤 같은 데이터로 평가한 것이라
(train/test 분리 없음) AUC가 가장 높게 나오는 것이 당연하다 — '데이터 기반 공식이
이론적으로 우월하다'는 근거로 쓰기보다는, in-sample 적합의 상한선 정도로 해석해야
한다. 이 비교는 '어느 공식이 옳은가'가 아니라 '공식 선택이 결과에 얼마나 민감한가'를
보여주는 민감도 분석으로 보고서에 쓰는 것을 추천한다.