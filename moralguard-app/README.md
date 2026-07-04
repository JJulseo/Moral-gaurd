# MoralGuard (모바일 앱 프로토타입)

'민재'라는 특정 환자 한 명의 180일 구강 건강 데이터와, 스마트폰 프레임 안에서
동작하는 인터랙티브 모바일 앱 UI(Home / Insights / Calendar / Compare 4탭)를
제공하는 프로토타입입니다. Compare 탭의 코호트 비교/페르소나 궤적은
`../data/cohort_results.json`(300인 합성 코호트 분석 결과)을 그대로 재사용합니다
(동일한 DRS/ARS/BES/DOHI 스코어링 방법론 공유 — `archive/oralguard/`에 보관된
코호트 생성/채점/분석 파이프라인 산출물).

## 데이터 재생성

```bash
cd moralguard-app
source ../.venv/bin/activate   # 저장소 루트 venv 재사용 (pandas/numpy 등)
python3 generate_data.py       # -> data/patient_data.json
```

## 앱 실행

`index.html`은 `fetch()`로 `data/patient_data.json`과 `../data/cohort_results.json`
을 함께 불러옵니다. `file://`로 직접 열면 브라우저가 로컬 fetch를 차단하므로,
저장소 루트에서 아래처럼 서버를 띄운 뒤 접속하세요:

```bash
python3 -m http.server 8000
# http://localhost:8000/moralguard-app/index.html
```

## 구조

```
moralguard-app/
├── index.html         # 단일 앱 (스마트폰 프레임 + 4탭, Chart.js CDN)
├── generate_data.py   # 민재 180일 데이터 + 오늘의 세션/pH 상세 데이터 생성
└── data/
    └── patient_data.json
```
