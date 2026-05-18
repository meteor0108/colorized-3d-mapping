#!/bin/bash
# Step5 VGICP+PGO 배치 실행. configs/colorize.yaml > step5_vgicp_pgo 사용.
# 환경변수로 VGICP/PGO on-off 가능.
#   USE_VGICP=0 ./scripts/run_step5_colorize.sh

set -e
cd "$(dirname "$0")/.."

ARGS=()
[ "${USE_VGICP:-1}" = "0" ] && ARGS+=("--no-vgicp")
[ "${USE_PGO:-1}" = "0" ] && ARGS+=("--no-pgo")

python -m src.pipeline.step5_colorize_vgicp_pgo "${ARGS[@]}"
