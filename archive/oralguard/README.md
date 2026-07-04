# (Archived) OralGuard Dashboard

이 폴더는 프로젝트 초기에 만든 6탭 OralGuard 대시보드(300인 코호트 전용
데스크톱 대시보드)의 원본 코드를 보관용으로 옮겨둔 것입니다. 최종 제출물은
`moralguard-app/`(모바일 앱)과 `research/`(통계 분석), `report/`(보고서)이며,
이 폴더는 더 이상 유지보수 대상이 아닙니다.

## 왜 코드는 여기 있는데 데이터는 저장소 루트 `data/`에 남아있나요?

`scripts/generate_cohort.py` → `scripts/scoring.py` → `scripts/analysis.py`
파이프라인이 만들어낸 `data/cohort_results.json`을 `moralguard-app/index.html`의
Comparison 탭이 여전히 `fetch('../data/cohort_results.json')`으로 불러와
개인 대 코호트 비교, 페르소나 궤적 비교에 사용하고 있고, `data/cohort_scored.csv`는
`research/statistical_analysis.py`가 그대로 읽습니다. 두 의존성을 깨지 않기 위해
`data/` 폴더는 저장소 루트에 그대로 두었습니다.

## 참고

이 안의 스크립트들은 저장소 루트를 현재 작업 디렉터리(cwd)로 가정하고 상대
경로(`data/...`, `figures/...`)를 사용합니다. `archive/oralguard/` 안에서 직접
실행하면 이 폴더 안에 새 `data/`, `figures/`를 만들 뿐 루트의 실제 데이터에는
영향을 주지 않으니, 재실행이 필요하면 저장소 루트에서
`python3 archive/oralguard/scripts/generate_cohort.py` 형태로 실행하거나
경로를 직접 조정하세요.
