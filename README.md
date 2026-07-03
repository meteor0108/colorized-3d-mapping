# colorized-3d-mapping

다중 LiDAR · Camera · GNSS rosbag 데이터를 입력으로 받아, **컬러라이즈된 3D 포인트클라우드 맵**과
**메쉬 / Gaussian Splatting 3D 재구성**을 생성하는 오프라인 파이프라인입니다.

포즈는 GNSS/INS(NovAtel CPT7) odom + VGICP/PGO로 도출하며(별도 COLMAP SfM 불필요),
동적 객체 제거 후 **(A) LiDAR 기반 메쉬** 또는 **(B) Gaussian Splatting + Depth 감독** 두 트랙으로 재구성합니다.

---

## Overview

```
                                              ┌─► [A] Classical Mesh   : NKSR / Poisson / dense   (Step 6·7·9)
rosbag ─► 추출/검증 ─► Trajectory ─► Colorize ─┤
 (.bag)   (Pre)        (Step 0~1)    (Step 2~5) └─► [B] Gaussian Splatting: 2DGS/3DGS + Depth 감독  (Step 7b~14)
                                          ▲
                        Dynamic Object Removal (pre-pass: SAM3 mask / LaMa inpaint)
```

큰 단계 구성:

1. **Pre** : rosbag을 폴더 구조(image/PCD/nav)로 추출하고 동기화/누락을 검증
2. **Trajectory** : navigation CSV → 2D/3D 궤적 시각화 및 캘리브 점검
3. **Colorize** : LiDAR 포인트클라우드에 카메라 색상을 투영해 컬러 PCD 생성
4. **Dynamic Removal** : SAM3로 이동체 마스킹 → 융합에서 제외(마스크) 또는 LaMa inpaint
5. **재구성 (택1/병행)**
   - **[A] Mesh** : 컬러 PCD를 글로벌 좌표로 병합 후 NKSR / Poisson / dense-LiDAR 메쉬
   - **[B] Gaussian Splatting** : GS dataset 준비 → 2DGS/3DGS 학습(+LiDAR depth 감독) → mesh/렌더

> **현재 방향성**: 맵 품질 개선을 위해 **[B] GS + depth 감독**을 주력으로 전환 중.
> 로드맵과 미해결 과제는 [`TODO.md`](TODO.md) 참고.

---

## Directory Structure

```
colorized-3d-mapping/
├── configs/                        # 모든 파라미터 (YAML)
│   ├── default.yaml                  # 엔트리포인트 (아래 yaml들을 include)
│   ├── sensor_calibration.yaml       # 카메라 / LiDAR extrinsic·intrinsic
│   ├── paths.yaml                    # 입출력 경로
│   ├── extraction.yaml               # rosbag 추출 / 검증
│   ├── projection.yaml               # LiDAR-Camera projection
│   ├── trajectory.yaml               # Navigation / trajectory
│   ├── colorize.yaml                 # Colorize 단계 전반
│   ├── merge.yaml                    # 글로벌 맵 병합
│   └── mesh.yaml                     # NKSR / Poisson + Gaussian Splatting(mesh.gaussian.*)
├── src/
│   ├── common/                     # Config 로더, 캘리브, 포인트클라우드 유틸
│   ├── pgo/                        # VGICP, Pose Graph Optimization
│   ├── pipeline/                   # 파이프라인 단계 스크립트 (step_pre*, step0~14, gs_backbone_*)
│   └── utils/                      # Viewer, dataset_check
├── scripts/                        # 실행용 셸 스크립트
│   ├── run_full_pipeline.sh
│   ├── run_step5_colorize.sh
│   ├── run_step6_mesh_*.sh
│   └── run_step8_gs_mesh.sh          # GS(2DGS) dataset+학습+mesh
├── docs/                           # 단계별 상세 문서
├── TODO.md                         # 로드맵 / depth 감독 개선 과제
└── requirements.txt

# 외부(저장소 미포함): third_party/  ← 2DGS/3DGS 백본 repo (CUDA rasterizer 컴파일)
```

---

## Pipeline Stages

### Pre 단계 — rosbag → 폴더

| Step  | Script                              | 역할                                              |
| ----- | ----------------------------------- | ----------------------------------------------- |
| pre1  | `step_pre1_extract_rosbag.py`       | rosbag에서 이미지 / PCD / nav 토픽을 폴더로 분리 추출           |
| pre1  | `step_pre1_extract_native.py`       | LiDAR/카메라/nav를 **native rate 전량** 추출 (mesh 밀도용) |
| pre1b | `step_pre1b_timestamp_convert.py`   | rosbag header timestamp 기준으로 파일명 reindex        |
| pre1c | `step_pre1c_rosbag_to_csv.py`       | 지정한 토픽을 CSV로 export                             |
| pre2  | `step_pre2_validate_dataset.py`     | 추출 결과의 누락 / 동기화 상태 검증                           |
| pre3  | `step_pre3_timestamp_viz.py`        | 센서별 timestamp 분포를 Gantt chart로 시각화 (plotly)     |

### Step 0 — Calibration & Visualization

| Step | Script                              | 역할                                              |
| ---- | ----------------------------------- | ----------------------------------------------- |
| 0    | `step0_calibration_test.py`         | LiDAR-Camera extrinsic 검증 (투영 오버레이)             |
| 0    | `step0_vignetting_test.py`          | 카메라 비네팅 마스크 인터랙티브 튜닝                            |
| 0    | `step0_projection_show.py`          | 멀티 LiDAR projection 결과 확인                       |
| 0    | `step0_trajectory_from_csv.py`      | 단일 `navigation.csv` → trajectory plot           |
| 0    | `step0_trajectory_show.py`          | 폴더 내 전체 trajectory를 3D viewer로 표시               |
| 0    | `step0_map_compare.py`              | 맵 PCD vs GPS trajectory 2D 비교 이미지               |
| 0    | `step0_merge_auto.py`               | Radius 기반 자동 글로벌 맵 병합                           |
| 0    | `step0_merge_input.py`              | 입력 리스트 기반 UTM 글로벌 맵 병합                          |

### Step 1 — Navigation 시각화

| Step | Script                              | 역할                                              |
| ---- | ----------------------------------- | ----------------------------------------------- |
| 1    | `step1_nav_reader_html.py`          | 전체 폴더 trajectory → plotly HTML                  |
| 1    | `step1_nav_reader_plot.py`          | 전체 폴더 trajectory → matplotlib PNG               |

### Step 2~5 — Colorization

| Step | Script                              | 역할                                              |
| ---- | ----------------------------------- | ----------------------------------------------- |
| 2    | `step2_colorize_folder.py`          | 단일 폴더 입력 LiDAR-Camera 융합                        |
| 3    | `step3_colorize_rosbag_deskew.py`   | rosbag 직접 입력 + LiDAR deskew                     |
| 4    | `step4_colorize_auto.py`            | 폴더 배치 처리 (ouster2 단일 LiDAR, GNSS 변환)            |
| 5    | `step5_colorize_vgicp_pgo.py`       | 배치 + VGICP sliding window + PGO (GPU 가속)        |

### Dynamic Object Removal — pre-pass (선택)

이동체(차량/사람 등)를 제거해 재구성 품질을 높입니다. 두 전략을 지원합니다.

| Script                    | 역할                                                                 |
| ------------------------- | ------------------------------------------------------------------ |
| `step_pre_dynmask.py`     | SAM3 마스크 → dilate → `dynmask/`. 융합 시 해당 depth 제외 → 배경을 타 프레임 관측으로 채움 |
| `step_pre_inpaint.py`     | SAM3 마스크 → LaMa inpaint → `blackfly_nodyn/`. mono-only 융합용 이미지 생성    |

> 권장: **마스크 방식**(dynmask). LaMa 환각을 메쉬/GS에 굽지 않아 맵 품질에 유리.

### Step 6~7 · 9 — Classical Mesh [Track A]

| Step | Script                              | 역할                                              |
| ---- | ----------------------------------- | ----------------------------------------------- |
| 6    | `step6_mesh_nksr.py`                | NKSR 기반 메쉬 생성                                   |
| 6    | `step6_mesh_post_process.py`        | 메쉬 outlier 제거 / 단순화                             |
| 7    | `step7_mesh_poisson.py`             | Chunked Poisson surface reconstruction          |
| 9    | `step9_mesh_lidar_dense.py`         | native-rate LiDAR 전량 누적 → dense Poisson (LiDAR-primary) |
| 9b   | `step9b_dense_refined_textured.py`  | scan-to-map ICP 포즈정밀화 + 이미지 텍스처 베이킹             |
| 9c   | `step9c_texture_bake.py`            | LiDAR Poisson 기하 + 2DGS 렌더 텍스처 베이킹              |

### Step 7b~8 · 13 — Gaussian Splatting [Track B]

| Step | Script                              | 역할                                              |
| ---- | ----------------------------------- | ----------------------------------------------- |
| 7b   | `step7b_prepare_gs_dataset.py`      | submap → GS dataset (images/cameras.json/depth/points3D.ply), 1Hz+raw odom |
| 7c   | `step7c_prepare_gs_dense.py`        | **native 10Hz + sliding-window ICP** GS dataset (멀티뷰 overlap↑, ghosting↓) |
| 8    | `step8_mesh_gaussian.py`            | 2DGS 학습(L1+SSIM + LiDAR depth L1 + normal) → TSDF mesh 추출 |
| —    | `gs_backbone_2dgs.py`               | 2DGS 백본 어댑터 (hbb1/2d-gaussian-splatting)         |
| —    | `gs_backbone_3dgs.py`               | 3DGS 백본 어댑터 (graphdeco-inria, 렌더용)               |
| 13   | `step13_gsplat_depth.py`            | gsplat 3DGS + **LiDAR depth 감독 + sky alpha** (expected-depth 신뢰) |

### Step 10~12 — Rendering & Viewer [Track B]

| Step | Script                              | 역할                                              |
| ---- | ----------------------------------- | ----------------------------------------------- |
| 10   | `step10_render_gs_video.py`         | 학습된 2DGS로 궤적 flythrough 영상(mp4)                 |
| 11   | `step11_3dgs_flythrough.py`         | 3DGS 학습 → flythrough 영상 (2DGS보다 sharp)          |
| 12   | `step12_interactive_3dgs.py`        | 학습된 3DGS 실시간 자유시점 뷰어 (키보드 탐색)                   |

### Step 14 — Mono Depth Fusion (Depth Completion) [Track B]

| Step | Script                              | 역할                                              |
| ---- | ----------------------------------- | ----------------------------------------------- |
| 14   | `step14_mono_depth_fusion.py`       | Depth-Anything-V2 metric + LiDAR scale-align → TSDF dense 메쉬 |
| 14   | `step14_seq_mono_fusion.py`         | 시퀀스 전체를 chunk 타일링해 mono 융합 (RAM 절약)             |

> ⚠️ 현재 mono depth는 **TSDF mesh 전용**이며 아직 GS depth 감독으로 연결돼 있지 않음.
> sparse LiDAR depth → **dense depth 감독** 전환이 로드맵의 핵심 ([`TODO.md`](TODO.md) P1).

---

## Installation

```bash
pip install -r requirements.txt
```

### 외부 의존성

- Python ≥ 3.8
- ROS Noetic — `step_pre1`, `step3`에서 사용 (`rosbag`, `cv_bridge`, `sensor_msgs`)
- Open3D (CUDA 빌드 권장)
- PyTorch + [NKSR](https://nksr.s3.ap-northeast-1.amazonaws.com/whl/torch-2.0.0+cu118.html) — Step 6
- pygicp — Step 5 (VGICP)
- rosbags — Step 3 (ROS-independent rosbag 파싱)
- plotly, matplotlib

### Gaussian Splatting (Track B, 전용 env 권장 — 예: `kctc_gs`)

```bash
# 2DGS (mesh) — CUDA rasterizer 컴파일
git clone --recursive https://github.com/hbb1/2d-gaussian-splatting.git third_party/2d-gaussian-splatting
TORCH_CUDA_ARCH_LIST=8.9 pip install ./third_party/2d-gaussian-splatting/submodules/diff-surfel-rasterization \
                                    ./third_party/2d-gaussian-splatting/submodules/simple-knn

# 3DGS (렌더) — 선택
git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git third_party/gaussian-splatting

pip install gsplat transformers pillow   # step13 / step14 (Depth-Anything-V2)
```

- 동적 제거(pre-pass)는 **SAM3 + LaMa** 가중치/설치가 별도로 필요합니다.

---

## Quick Start

```bash
# 1) rosbag 추출 → 2) 검증 → 3) 캘리브 확인
python -m src.pipeline.step_pre1_extract_rosbag --bag /path/to/data.bag
python -m src.pipeline.step_pre2_validate_dataset --folder /path/to/extracted
python -m src.pipeline.step0_calibration_test

# 4) Colorize (VGICP + PGO 배치)
python -m src.pipeline.step5_colorize_vgicp_pgo        # 또는 ./scripts/run_step5_colorize.sh

# 5-A) [Track A] 글로벌 맵 병합 → 메쉬
python -m src.pipeline.step0_merge_auto
./scripts/run_step6_mesh_batch.sh

# 5-B) [Track B] Gaussian Splatting (dataset 준비 + 2DGS 학습 + mesh)
./scripts/run_step8_gs_mesh.sh                          # step7b → step8
# depth 감독 3DGS + flythrough:
python -m src.pipeline.step13_gsplat_depth --dataset ./data/gs_dataset/<submap> -o out.mp4
```

전체 자동 실행: `./scripts/run_full_pipeline.sh`

---

## Configuration

모든 스크립트는 기본적으로 `configs/default.yaml`을 로드합니다. `default.yaml`은 다른 yaml들을
`includes`로 묶어주는 엔트리포인트이며, 단계별 yaml에서 파라미터를 수정합니다.
다른 config는 `--config <yaml>`로 지정합니다.

### 주요 파라미터

| 영역                       | Config 키                                          |
| ------------------------ | ------------------------------------------------ |
| rosbag 토픽 목록             | `extraction.topics.*`                            |
| LiDAR-Image timestamp 매칭 | `colorize.processing.time_tolerance`             |
| Submap chunking          | `colorize.submap.frame_limit`                    |
| VGICP sliding window     | `colorize.step5_vgicp_pgo.vgicp.*`               |
| Pose Graph Optimization  | `colorize.step5_vgicp_pgo.pgo.*`                 |
| **GS dataset 준비**        | `mesh.gaussian.dataset.*` (lidars/pose_lidar/depth_export) |
| **GS 학습(2DGS/3DGS)**     | `mesh.gaussian.train.*` (lambda_depth/lambda_normal/optimize_camera_pose) |
| **GS depth 감독**          | `mesh.gaussian.train.lambda_depth`, `depth_ramp_iters`, `lambda_sky` |
| **TSDF mesh 추출**         | `mesh.gaussian.extract.*` (tsdf_voxel/depth_trunc) |

입출력 경로는 `configs/paths.yaml`에서 변경합니다 (데이터 디렉토리는 저장소 미포함).

---

## Roadmap / TODO

맵 품질 개선(특히 single-pass GS)을 위한 진행 방향과 미해결 과제는 [`TODO.md`](TODO.md)에 정리돼 있습니다.
핵심 우선순위:

1. **(P0)** `depth/*.npy` z-buffer 반전 버그 검증/픽스 — GS 두 백본이 이 파일을 depth 감독에 사용 중.
2. **(P1)** sparse LiDAR depth → **dense depth 감독** (mono affine-align + confidence map).
3. **(P2)** depth loss를 confidence 가중으로 전환 + GS camera pose-opt 활성화.

---

## Documentation

- NKSR 메쉬 단계 상세: [`docs/STEP6_MESH_README.md`](docs/STEP6_MESH_README.md)
- 로드맵 / depth 감독 개선: [`TODO.md`](TODO.md)
