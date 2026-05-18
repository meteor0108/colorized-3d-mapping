#!/bin/bash
# 전체 파이프라인 0→7 자동 실행 (configs/default.yaml 기준)

set -e
cd "$(dirname "$0")/.."

echo "[0/7] Nav reader & trajectory visualization"
python -m src.pipeline.step1_nav_reader_plot

echo "[1/7] (스킵 - 인터랙티브 단계)"
echo "[2/7] (스킵 - 단일 폴더용)"

echo "[3/7] Auto colorize (ouster2 only, GNSS)"
python -m src.pipeline.step4_colorize_auto

echo "[4/7] VGICP + PGO 융합"
python -m src.pipeline.step5_colorize_vgicp_pgo

echo "[5/7] Global map merge (자동 그룹핑)"
python -m src.pipeline.step0_merge_auto

echo "[6/7] Mesh 재구성 (NKSR + post-process)"
bash scripts/run_step6_mesh_batch.sh

echo "[7/7] Done."
