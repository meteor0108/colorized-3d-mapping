#!/bin/bash

# ========================================
# 2D Gaussian Splatting Mesh Reconstruction
# step7b (submap → GS dataset) → step8 (2DGS 학습 + TSDF mesh)
# ========================================

set -e

GREEN='\033[0;32m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'; NC='\033[0m'

# --- 설정 (환경변수로 override) ---
# 예) FOLDER=/path/to/submap ITERS=20000 ./scripts/run_step8_gs_mesh.sh
PYBIN="${PYBIN:-/home/airlab/anaconda3/envs/kctc_gs/bin/python}"   # 2DGS 컴파일된 env
FOLDER="${FOLDER:-}"                       # 입력 submap (비우면 config의 input_folder)
DATASET_ROOT="${DATASET_ROOT:-./data/gs_dataset}"
ITERS="${ITERS:-20000}"
MAX_FRAMES="${MAX_FRAMES:-}"               # 비우면 전체 프레임
OUTPUT="${OUTPUT:-}"                       # 비우면 자동 명명

cd "$(dirname "$0")/.."                     # → Data_Preprocessing/
export PYTHONPATH="$PWD"

if [ ! -x "$PYBIN" ]; then
  echo -e "${YELLOW}[경고] 2DGS env python 없음: $PYBIN${NC}"
  echo "  setup: third_party/2d-gaussian-splatting 클론 + diff-surfel-rasterization 컴파일 필요"
  exit 1
fi

# --- step7b: GS dataset 준비 ---
echo -e "${BLUE}[1/2] step7b: GS dataset 준비${NC}"
S7B_ARGS=()
[ -n "$FOLDER" ] && S7B_ARGS+=(--folder "$FOLDER")
[ -n "$MAX_FRAMES" ] && S7B_ARGS+=(--max-frames "$MAX_FRAMES")
"$PYBIN" -u -m src.pipeline.step7b_prepare_gs_dataset "${S7B_ARGS[@]}"

# dataset 폴더 추론 (가장 최근 생성된 것)
DATASET="${DATASET:-$(ls -td "$DATASET_ROOT"/*/ 2>/dev/null | head -1)}"
echo -e "${GREEN}  dataset: $DATASET${NC}"

# --- step8: 2DGS 학습 + mesh ---
echo -e "${BLUE}[2/2] step8: 2DGS 학습 (${ITERS} iters) + TSDF mesh${NC}"
S8_ARGS=(--dataset "$DATASET" --iterations "$ITERS")
[ -n "$OUTPUT" ] && S8_ARGS+=(--output "$OUTPUT")
"$PYBIN" -u -m src.pipeline.step8_mesh_gaussian "${S8_ARGS[@]}"

echo -e "${GREEN}✅ 완료${NC}"
