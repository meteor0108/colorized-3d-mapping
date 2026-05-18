#!/bin/bash

# ========================================
# NKSR Mesh Reconstruction Pipeline
# 자동화 배치 스크립트
# ========================================

set -e  # 에러 발생시 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 설정 변수 (configs/mesh.yaml > batch 와 동기화. 환경변수로 override 가능)
# 예) INPUT_PCD=foo.pcd ./scripts/run_step6_mesh_batch.sh
INPUT_PCD="${INPUT_PCD:-2024-10-22-10-45-46_0_full3_corrected_vgicp_pgo.pcd}"
OUTPUT_DIR="${OUTPUT_DIR:-./data/output_mesh}"
DEVICE="${DEVICE:-cuda:0}"

# NKSR 파라미터 (configs/mesh.yaml > batch 와 일치)
NORMAL_K="${NORMAL_K:-100}"
NORMAL_RADIUS="${NORMAL_RADIUS:-0.5}"
MAX_POINTS="${MAX_POINTS:-20000000}"
MISE_ITER="${MISE_ITER:-4}"

# 후처리 파라미터
SIMPLIFY_TRIANGLES="${SIMPLIFY_TRIANGLES:-2000000}"
REMOVE_SMALL_COMPONENTS="${REMOVE_SMALL_COMPONENTS:-100}"

# 프로젝트 루트로 이동 (스크립트가 어디서 호출되든 src.* import 가능)
cd "$(dirname "$0")/.."

echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}  NKSR Mesh Reconstruction Pipeline${NC}"
echo -e "${BLUE}=======================================${NC}"
echo ""

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"

# 파일 경로 설정
RAW_MESH="$OUTPUT_DIR/mesh_raw.ply"
CLEAN_MESH="$OUTPUT_DIR/mesh_clean.ply"
FINAL_MESH="$OUTPUT_DIR/mesh_final.ply"

# 입력 파일 확인
if [ ! -f "$INPUT_PCD" ]; then
    echo -e "${RED}❌ Error: Input file not found: $INPUT_PCD${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Input file found: $INPUT_PCD${NC}"
echo ""

# ========================================
# Step 1: NKSR 메쉬 재구성
# ========================================
echo -e "${YELLOW}Step 1: Running NKSR Reconstruction...${NC}"
echo "Parameters:"
echo "  - Normal K: $NORMAL_K"
echo "  - Normal Radius: $NORMAL_RADIUS m"
echo "  - Max Points: $MAX_POINTS"
echo "  - MISE Iterations: $MISE_ITER"
echo ""

python -m src.pipeline.step6_mesh_nksr \
    --input "$INPUT_PCD" \
    --output "$RAW_MESH" \
    --device "$DEVICE" \
    --normal-k "$NORMAL_K" \
    --normal-radius "$NORMAL_RADIUS" \
    --max-points "$MAX_POINTS" \
    --mise-iter "$MISE_ITER"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Step 1 completed${NC}"
else
    echo -e "${RED}❌ Step 1 failed${NC}"
    exit 1
fi
echo ""

# ========================================
# Step 2: 이상치 제거 및 정리
# ========================================
echo -e "${YELLOW}Step 2: Cleaning mesh (outlier removal)...${NC}"

python -m src.pipeline.step6_mesh_post_process \
    --input "$RAW_MESH" \
    --output "$CLEAN_MESH" \
    --remove-outliers \
    --outlier-neighbors 20 \
    --outlier-std 2.0 \
    --remove-small "$REMOVE_SMALL_COMPONENTS"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Step 2 completed${NC}"
else
    echo -e "${RED}❌ Step 2 failed${NC}"
    exit 1
fi
echo ""

# ========================================
# Step 3: 메쉬 단순화 (옵션)
# ========================================
echo -e "${YELLOW}Step 3: Simplifying mesh (optional)...${NC}"

python -m src.pipeline.step6_mesh_post_process \
    --input "$CLEAN_MESH" \
    --output "$FINAL_MESH" \
    --simplify "$SIMPLIFY_TRIANGLES" \
    --smooth \
    --smooth-iter 1 \
    --smooth-lambda 0.3

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Step 3 completed${NC}"
else
    echo -e "${RED}❌ Step 3 failed${NC}"
    exit 1
fi
echo ""

# ========================================
# Step 4: 통계 출력
# ========================================
echo -e "${YELLOW}Step 4: Final mesh statistics...${NC}"

python -m src.pipeline.step6_mesh_post_process \
    --input "$FINAL_MESH" \
    --stats

echo ""

# ========================================
# 완료
# ========================================
echo -e "${BLUE}=======================================${NC}"
echo -e "${GREEN}✅ Pipeline completed successfully!${NC}"
echo -e "${BLUE}=======================================${NC}"
echo ""
echo "Output files:"
echo "  - Raw mesh: $RAW_MESH"
echo "  - Clean mesh: $CLEAN_MESH"
echo "  - Final mesh: $FINAL_MESH"
echo ""
echo "To visualize:"
echo "  python -m src.pipeline.step6_mesh_post_process --input $FINAL_MESH --visualize"
echo ""

# 파일 크기 출력
echo "File sizes:"
if [ -f "$RAW_MESH" ]; then
    SIZE=$(du -h "$RAW_MESH" | cut -f1)
    echo "  - Raw mesh: $SIZE"
fi
if [ -f "$CLEAN_MESH" ]; then
    SIZE=$(du -h "$CLEAN_MESH" | cut -f1)
    echo "  - Clean mesh: $SIZE"
fi
if [ -f "$FINAL_MESH" ]; then
    SIZE=$(du -h "$FINAL_MESH" | cut -f1)
    echo "  - Final mesh: $SIZE"
fi