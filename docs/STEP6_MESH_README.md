# NKSR Point Cloud to Mesh Reconstruction

## 📋 Requirements

```bash
# 1. PyTorch 설치 (CUDA 11.8 기준)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 2. NKSR 설치
pip install nksr -f https://nksr.s3.ap-northeast-1.amazonaws.com/whl/torch-2.0.0%2Bcu118.html

# 3. 기타 의존성
pip install open3d numpy tqdm
```

## 🚀 Usage

### 기본 사용법

```bash
python nksr_mesh_generator.py \
    --input /path/to/your/pointcloud.pcd \
    --output /path/to/output/mesh.ply
```

### 전체 옵션 사용 예시

```bash
python nksr_mesh_generator.py \
    --input outdoor_map.pcd \
    --output outdoor_mesh.ply \
    --device cuda:0 \
    --normal-k 30 \
    --normal-radius 0.5 \
    --max-points 15000000 \
    --mise-iter 2
```

## ⚙️ Parameters

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--input` | **필수** | 입력 PCD 파일 경로 |
| `--output` | **필수** | 출력 PLY 파일 경로 |
| `--device` | `cuda:0` | CUDA 디바이스 |
| `--normal-k` | `30` | Normal 계산시 KNN 이웃 개수 |
| `--normal-radius` | `0.5` | Normal 계산 검색 반경 (미터) |
| `--max-points` | `10000000` | 최대 처리 포인트 수 (메모리 제약) |
| `--mise-iter` | `2` | 메쉬 해상도 (2-3 권장) |

## 🎯 파라미터 튜닝 가이드

### 1. Normal 계산 파라미터

**`--normal-k` (KNN 이웃 개수)**
- **작은 값 (10-20)**: 디테일한 표면, 노이즈에 민감
- **중간 값 (30-50)**: 균형잡힌 결과 ✅ **권장**
- **큰 값 (50-100)**: 부드러운 표면, 디테일 손실

**`--normal-radius` (검색 반경)**
- 포인트 밀도에 따라 조정
- 야외 환경 (100m × 100m, 17M points): `0.3 - 0.7m` 권장
- 스파스한 데이터: 반경 증가
- 밀집한 데이터: 반경 감소

### 2. 메쉬 품질 파라미터

**`--mise-iter` (메쉬 해상도)**
- `1`: 빠른 프리뷰 (낮은 해상도)
- `2`: 균형잡힌 품질 ✅ **권장**
- `3`: 높은 디테일 (처리 시간 증가)
- `4+`: 매우 높은 디테일 (매우 느림, 메모리 대량 사용)

### 3. 메모리 관리

**`--max-points` (최대 포인트)**
- GPU VRAM에 따라 조정
- **8GB VRAM**: `8,000,000` points
- **12GB VRAM**: `12,000,000` points
- **16GB+ VRAM**: `15,000,000+` points

현재 데이터셋 (16.9M points): 
- 12GB+ VRAM이면 전체 처리 가능
- 8GB VRAM이면 다운샘플링 필요

## 📊 예상 처리 시간 (RTX 3090 기준)

| 포인트 수 | Normal 계산 | NKSR 재구성 | 총 시간 |
|-----------|-------------|-------------|---------|
| 1M | ~10초 | ~30초 | ~40초 |
| 5M | ~30초 | ~2분 | ~2.5분 |
| 10M | ~1분 | ~4분 | ~5분 |
| 17M | ~2분 | ~7분 | **~9분** |

## 💡 Tips

### 1. 메모리 부족 에러가 발생하면

```bash
# max-points 줄이기
--max-points 8000000
```

### 2. 더 부드러운 메쉬를 원하면

```bash
# normal-k 증가
--normal-k 50 --normal-radius 0.7
```

### 3. 더 세밀한 디테일을 원하면

```bash
# mise-iter 증가
--mise-iter 3
```

### 4. 야외 환경 최적화 설정 (권장)

```bash
python nksr_mesh_generator.py \
    --input outdoor_map.pcd \
    --output outdoor_mesh.ply \
    --normal-k 40 \
    --normal-radius 0.6 \
    --max-points 16000000 \
    --mise-iter 2
```

## 🔍 결과 확인

### CloudCompare로 확인
```bash
# Ubuntu
sudo snap install cloudcompare

cloudcompare outdoor_mesh.ply
```

### MeshLab으로 확인
```bash
# Ubuntu
sudo apt install meshlab

meshlab outdoor_mesh.ply
```

### Python으로 시각화
```python
import open3d as o3d

mesh = o3d.io.read_triangle_mesh("outdoor_mesh.ply")
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])
```

## ⚠️ 주의사항

1. **Normal 방향**: 야외 환경의 경우 센서 위치 정보가 있으면 더 정확한 normal 계산 가능
2. **노이즈**: 입력 포인트 클라우드의 노이즈가 많으면 사전에 필터링 권장
3. **홀/구멍**: NKSR은 watertight 메쉬를 생성하므로 일부 홀이 자동으로 채워질 수 있음

## 🐛 Troubleshooting

### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
➡️ `--max-points` 값을 줄이거나, `mise-iter`를 1로 낮추기

### Invalid Normals
```
⚠️  Found X invalid normals, filtering...
```
➡️ 정상적인 동작. 자동으로 필터링됨

### 메쉬가 너무 거칠게 나올 때
➡️ `--mise-iter 3` 으로 증가 또는 `--normal-k` 증가