# MoralGuard

스마트 칫솔 센서 데이터와 구강 pH 시뮬레이션을 결합한 일별 구강 건강 지수
DOHI(Daily Oral Health Index)를 설계하고, 가상 환자 "민재"의 180일 데이터를
모바일 앱 프로토타입으로 시각화하는 프로젝트입니다. DOHI 채점 방법론은 300인
합성 코호트에 대한 통계 분석으로 별도 검증했습니다. 자세한 이론적 근거와 통계
분석 결과는 `report/report.docx`(또는 `report/report.md`)를 참고하세요.

## 구조

```
moralguard-app/           # 최종 앱 (Home/Insights/Calendar/Compare 4탭)
  generate_data.py        # 민재 180일 데이터 + 오늘의 세션/pH 상세 데이터 생성
  index.html              # 단일 앱 (스마트폰 프레임, Chart.js CDN)
research/
  statistical_analysis.py   # 300인 코호트 통계 분석 (상관/군집/회귀/DRS 공식 비교)
report/
  report.md, report.docx    # 최종 보고서 (이론적 근거 -> 통계 검증 -> 앱 소개)
data/
  cohort_scored.csv        # research/statistical_analysis.py가 읽는 300인 코호트 채점 결과
  cohort_results.json      # moralguard-app Compare 탭이 fetch하는 코호트 요약(페르소나/분포 등)
archive/oralguard/         # (보관용) 초기 6탭 데스크톱 대시보드 원본 코드 — 더 이상 유지보수 안 함
reproduce.sh              # 전체 파이프라인 재현 스크립트 (아래 참고)
requirements.txt           # 버전 고정된 파이썬 의존성
```

## 재현성 확인 (한 번에 재생성)

이 저장소의 `data/`·`research/`·`archive/oralguard/figures/`·`moralguard-app/data/`
아래 파일은 전부 시드가 고정된 스크립트의 출력물이며, 저장소에도 이미 커밋되어
있습니다. 직접 재현해보려면:

```bash
./reproduce.sh
```

Python venv 생성 → `requirements.txt`(버전 고정) 설치 → 코호트 생성/채점/분석 →
민재 180일 데이터 생성 → 통계 분석까지 한 번에 실행합니다. 전부 시드 고정 +
결정론적 연산이라 실행 후 아래 명령으로 **변경 사항이 없어야** 재현이 검증된
것입니다:

```bash
git status --short   # 아무 것도 출력되지 않아야 정상
```

## 앱 실행

```bash
python3 -m http.server 8000
# http://localhost:8000/moralguard-app/index.html
```

`moralguard-app/index.html`은 `fetch()`로 `moralguard-app/data/patient_data.json`과
`data/cohort_results.json`을 불러오므로, `file://`로 직접 열면 브라우저가 로컬
fetch를 차단합니다. 반드시 위처럼 로컬 서버를 통해 열어주세요.

## 개별 스크립트만 다시 돌리고 싶다면

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python3 archive/oralguard/scripts/generate_cohort.py   # -> data/cohort_raw.csv
python3 archive/oralguard/scripts/scoring.py            # -> data/cohort_scored.csv
python3 archive/oralguard/scripts/analysis.py           # -> data/cohort_results.json, archive/oralguard/figures/*.png

(cd moralguard-app && python3 generate_data.py)         # -> moralguard-app/data/patient_data.json
python3 research/statistical_analysis.py                # -> research/figures/*.png, research/RESULTS.md
```
