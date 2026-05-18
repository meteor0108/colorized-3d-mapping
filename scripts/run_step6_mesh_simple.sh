#!/bin/bash
# 단일 PCD → mesh 변환 (간단 버전).
# 사용: ./scripts/run_step6_mesh_simple.sh <input.pcd> <output.ply>

set -e
cd "$(dirname "$0")/.."

INPUT="${1:-data/sample_pcd/2024-06-15-14-43-18_11_full3_corrected_vgicp_pgo.pcd}"
OUTPUT="${2:-data/output_mesh/$(basename "${INPUT%.*}_mesh.ply")}"

python -m src.pipeline.step6_mesh_nksr \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --normal-k 10 \
    --normal-radius 0.3 \
    --mise-iter 4
