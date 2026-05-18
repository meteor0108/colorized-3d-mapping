# colorized-3d-mapping

다중 LiDAR · Camera · GNSS rosbag 데이터를 입력으로 받아, **컬러라이즈된 3D 포인트클라우드 맵**과 **메쉬**를 생성하는 파이프라인입니다.

---

## Overview

```
rosbag  ──►  추출 / 검증  ──►  Trajectory  ──►  Colorize  ──►  Map Merge  ──►  Mesh
 (.bag)      (Pre 단계)        (Step 0~1)      (Step 2~5)     (Step 0_merge)  (Step 6~7)
```

전체 흐름은 4개의 큰 단계로 구성됩니다.

1. **Pre** : rosbag을 폴더 구조(image/PCD/nav)로 추출하고 동기화/누락을 검증
2. **Trajectory** : navigation CSV → 2D/3D 궤적 시각화 및 캘리브 점검
3. **Colorize** : LiDAR 포인트클라우드에 카메라 색상을 투영해 컬러 PCD 생성
4. **Map / Mesh** : 컬러 PCD를 글로벌 좌표로 병합한 뒤 NKSR / Poisson 메쉬로 변환

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
│   └── mesh.yaml                     # NKSR / Poisson 메쉬
├── src/
│   ├── common/                     # Config 로더, 캘리브, 포인트클라우드 유틸
│   ├── pgo/                        # VGICP, Pose Graph Optimization
│   ├── pipeline/                   # 파이프라인 단계 스크립트 (step_pre*, step0~7)
│   └── utils/                      # Viewer, dataset_check
├── scripts/                        # 실행용 셸 스크립트
│   ├── run_full_pipeline.sh
│   ├── run_step5_colorize.sh
│   └── run_step6_mesh_*.sh
├── docs/                           # 단계별 상세 문서
└── requirements.txt
```

---

## Pipeline Stages

### Pre 단계 — rosbag → 폴더

| Step  | Script                              | 역할                                              |
| ----- | ----------------------------------- | ----------------------------------------------- |
| pre1  | `step_pre1_extract_rosbag.py`       | rosbag에서 이미지 / PCD / nav 토픽을 폴더로 분리 추출           |
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

### Step 6~7 — Mesh

| Step | Script                              | 역할                                              |
| ---- | ----------------------------------- | ----------------------------------------------- |
| 6    | `step6_mesh_nksr.py`                | NKSR 기반 메쉬 생성                                   |
| 6    | `step6_mesh_post_process.py`        | 메쉬 outlier 제거 / 단순화                             |
| 7    | `step7_mesh_poisson.py`             | Chunked Poisson surface reconstruction          |

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

---

## Quick Start

```bash
# 1) rosbag 추출
python -m src.pipeline.step_pre1_extract_rosbag --bag /path/to/data.bag

# 2) 데이터셋 검증
python -m src.pipeline.step_pre2_validate_dataset --folder /path/to/extracted

# 3) 캘리브레이션 확인
python -m src.pipeline.step0_calibration_test

# 4) Colorize (VGICP + PGO 배치)
python -m src.pipeline.step5_colorize_vgicp_pgo
# 또는: ./scripts/run_step5_colorize.sh

# 5) 글로벌 맵 병합
python -m src.pipeline.step0_merge_auto

# 6) 메쉬 변환
./scripts/run_step6_mesh_batch.sh
```

전체 자동 실행:

```bash
./scripts/run_full_pipeline.sh
```

---

## Configuration

모든 스크립트는 기본적으로 `configs/default.yaml`을 로드합니다. `default.yaml`은 다른 yaml들을 `includes`로 묶어주는 엔트리포인트이며, 단계별 yaml에서 파라미터를 수정합니다.

```yaml
# 예: configs/extraction.yaml
extraction:
  rosbag_path: "/path/to/dataset.bag"
  sync_reference_topic: "/blackfly/image_raw/compressed"
  topics:
    raw_image:        [...]
    compressed_image: [...]
    pointcloud:       [...]
    gps:              [...]
  skip_already_processed: true
```

다른 config 파일을 사용하려면 `--config <yaml>` 인자를 지정합니다.

### 주요 파라미터

| 영역                       | Config 키                                          |
| ------------------------ | ------------------------------------------------ |
| rosbag 토픽 목록             | `extraction.topics.*`                            |
| Sync 윈도우                 | `extraction.sync_window_ns`                      |
| 이미 처리된 폴더 skip           | `extraction.skip_already_processed`              |
| LiDAR-Image timestamp 매칭 | `colorize.processing.time_tolerance`             |
| Submap chunking          | `colorize.submap.frame_limit`                    |
| GPU 가속 (Open3D Tensor)   | `colorize.gpu.enabled`, `colorize.gpu.device`    |
| 카메라 비네팅 보정               | `camera.vignetting.enabled`                      |
| VGICP sliding window     | `colorize.step5_vgicp_pgo.vgicp.*`               |
| Pose Graph Optimization  | `colorize.step5_vgicp_pgo.pgo.*`                 |
| 듀얼 GPU 라운드로빈 (병합)        | `merge.auto_merge.use_dual_gpu`                  |

### 동적 override

- CLI 플래그 : `--bag PATH`, `--folder PATH`, `--no-vgicp`, `--no-pgo` 등
- 환경 변수  : `USE_VGICP=0 ./scripts/run_step5_colorize.sh`

입출력 경로는 `configs/paths.yaml`에서 모두 변경할 수 있으며, 데이터 디렉토리는 별도로 준비합니다 (저장소에는 포함되지 않음).

---

## Documentation

- NKSR 메쉬 단계 상세: [`docs/STEP6_MESH_README.md`](docs/STEP6_MESH_README.md)
