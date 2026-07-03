# OralGuard Dashboard

스마트 칫솔 센서 데이터와 구강 pH 시뮬레이션을 결합한 일별 구강 건강 지수(DOHI)를
설계하고, 가상 종단 코호트(N=300, 180일)를 분석하여 인터랙티브 대시보드로
시각화하는 프로젝트입니다.

## 데이터 파이프라인 재생성

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 scripts/generate_cohort.py   # -> data/cohort_raw.csv
python3 scripts/scoring.py           # -> data/cohort_scored.csv
python3 scripts/analysis.py          # -> figures/*.png, data/cohort_results.json
```

## 대시보드 실행

`index.html`은 `fetch()`로 `data/cohort_results.json`을 불러오므로, 로컬에서
`file://`로 직접 열면 브라우저가 로컬 파일 fetch를 차단합니다. 반드시 아래처럼
간단한 로컬 서버를 통해 열어주세요 (GitHub Pages 배포 시에는 이 과정이 필요 없습니다):

```bash
python3 -m http.server 8000
# 브라우저에서 http://localhost:8000/index.html 접속
```

## 구조

```
index.html              # 단일 대시보드 (Chart.js CDN, vanilla JS)
scripts/
  generate_cohort.py     # 300명 가상 코호트 생성 (seed=42)
  scoring.py              # DRS/ARS/BES/DOHI 산출 + ICDAS 진행 시뮬레이션
  analysis.py             # 분포/상관/군집/ROC/개입 시뮬레이션 -> JSON+PNG
data/
  cohort_raw.csv          # 원본 코호트
  cohort_scored.csv       # 점수 산출 결과
  cohort_results.json     # 대시보드가 fetch하는 데이터
figures/                  # analysis.py 산출 분석 그림
```
