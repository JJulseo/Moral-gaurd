#!/usr/bin/env bash
# Regenerates every derived artifact in this repo from scratch (synthetic
# cohort -> scoring -> analysis -> MoralGuard patient data -> statistics).
# All scripts are seeded, so output is byte-for-byte deterministic: after
# this finishes, `git status` should show no changes to tracked files.
#
# Cross-platform (macOS/Linux/Windows Git Bash or WSL): auto-detects the
# Python launcher (python3 / python / py -3) and the venv activate path
# (bin/ on Unix, Scripts/ on Windows).
set -euo pipefail
cd "$(dirname "$0")"

if command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON="python"
elif command -v py >/dev/null 2>&1; then
  PYTHON="py -3"
else
  echo "Python 3를 찾을 수 없습니다. https://www.python.org/downloads/ 에서 설치 후 다시 실행해주세요."
  exit 1
fi

echo "[1/6] Python venv 설정 및 의존성 설치"
$PYTHON -m venv .venv

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
  source .venv/Scripts/activate
else
  echo "venv 활성화 스크립트를 찾을 수 없습니다 (.venv/bin/activate 또는 .venv/Scripts/activate)."
  exit 1
fi

python -m pip install -q -r requirements.txt

echo "[2/6] 300인 합성 코호트 생성 (archive/oralguard)"
python archive/oralguard/scripts/generate_cohort.py

echo "[3/6] DRS/ARS/BES/DOHI 채점 + ICDAS 진행 시뮬레이션"
python archive/oralguard/scripts/scoring.py

echo "[4/6] 코호트 분석 (분포/상관/페르소나/ROC-AUC/개입 시뮬레이션)"
python archive/oralguard/scripts/analysis.py

echo "[5/6] MoralGuard 환자 '민재' 180일 데이터 생성"
(cd moralguard-app && python generate_data.py)

echo "[6/6] 통계 분석 심화 (research/statistical_analysis.py)"
python research/statistical_analysis.py

echo ""
echo "완료. 재현성 확인: git status --short  (추적 중인 파일에 변경 사항이 없어야 정상입니다)"
echo "앱 실행: python -m http.server 8000  ->  http://localhost:8000/moralguard-app/index.html"
