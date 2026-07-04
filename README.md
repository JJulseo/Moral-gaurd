# MoralGuard

스마트 칫솔 센서 데이터와 구강 pH 시뮬레이션을 결합한 일별 구강 건강 지수
DOHI(Daily Oral Health Index)를 설계하고, 가상 환자 "민재"의 180일 데이터를
모바일 앱 프로토타입으로 시각화하는 프로젝트입니다. DOHI 채점 방법론은 300인
합성 코호트에 대한 통계 분석으로 별도 검증했습니다.

## 구조

```
moralguard-app/          # 최종 앱 (Home/Trends/Calendar/Comparison 4탭)
  generate_data.py       # 민재 180일 데이터 + 오늘의 세션/pH 상세 데이터 생성
  index.html             # 단일 앱 (스마트폰 프레임, Chart.js CDN)
research/
  statistical_analysis.py  # 300인 코호트 통계 분석 (상관/군집/회귀/DRS 공식 비교)
report/
  report.md, report.docx   # 최종 보고서 (이론적 근거 -> 통계 검증 -> 앱 소개)
data/
  cohort_scored.csv       # research/statistical_analysis.py가 읽는 300인 코호트 채점 결과
  cohort_results.json     # moralguard-app Comparison 탭이 fetch하는 코호트 요약(페르소나/분포 등)
archive/oralguard/        # (보관용) 초기 6탭 데스크톱 대시보드 원본 코드 — 더 이상 유지보수 안 함
```

## 앱 실행

```bash
python3 -m http.server 8000
# http://localhost:8000/moralguard-app/index.html
```

`moralguard-app/index.html`은 `fetch()`로 `moralguard-app/data/patient_data.json`과
`data/cohort_results.json`을 불러오므로, `file://`로 직접 열면 브라우저가 로컬
fetch를 차단합니다. 반드시 위처럼 로컬 서버를 통해 열어주세요.

## 데이터/통계 분석 재실행

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 moralguard-app/generate_data.py     # -> moralguard-app/data/patient_data.json
python3 research/statistical_analysis.py    # -> research/figures/*.png, research/RESULTS.md
```

`data/cohort_scored.csv`·`data/cohort_results.json`은 `archive/oralguard/scripts/`의
코호트 생성·채점·분석 파이프라인이 만든 결과물입니다. 이 파이프라인 자체를 다시
돌리고 싶다면 `archive/oralguard/README.md`를 참고하세요.
