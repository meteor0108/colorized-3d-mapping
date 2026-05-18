# colorized-3d-mapping

다중 LiDAR + Camera + GNSS 데이터를 rosbag에서 추출하여 컬러라이즈 3D 맵과 메쉬를 생성하는 파이프라인.

## 디렉토리 구조

```
processing/
├── configs/             # YAML 설정 (모든 파라미터)
│   ├── default.yaml         # 다른 yaml을 include하는 엔트리포인트
│   ├── sensor_calibration.yaml   # 카메라/LiDAR 캘리브레이션
│   ├── paths.yaml           # 입출력 경로
│   ├── extraction.yaml      # rosbag 추출/검증/타임스탬프 (pre 단계)
│   ├── projection.yaml      # 0_ projection 단계
│   ├── trajectory.yaml      # 0_/1_ trajectory 단계
│   ├── colorize.yaml        # 2_/3_/4_/5_ colorize 단계
│   ├── merge.yaml           # 0_merge_* 단계
│   └── mesh.yaml            # 6_/7_ mesh 단계
├── src/
│   ├── common/          # 공통 모듈 (config 로더, 캘리브, 포인트클라우드 유틸)
│   ├── pgo/             # VGICP, Pose Graph Optimization
│   ├── pipeline/        # step_pre1~3, step0~7
│   └── utils/           # viewer, dataset_check
├── scripts/             # 실행용 쉘 스크립트
├── docs/                # 단계별 추가 문서
├── data/                # 입출력 데이터
│   ├── raw_pcd/             # 원본 PCD (raw, deskew, merged_output)
│   ├── sample_pcd/          # 테스트용 샘플 PCD
│   ├── output_map/          # step4 결과 (ouster2 GNSS)
│   ├── output_map_processed/# step5 결과 (VGICP+PGO)
│   ├── output_mesh/         # step6/7 메쉬
│   ├── output_pcd/          # step2/3 단일폴더 결과
│   ├── output_etc/          # trajectory, 시각화, 리포트
│   └── csv/                 # step_pre1c CSV 출력
└── _legacy/             # 구버전/원본 파일 보관 (50GB map_backup 포함)
```

## 파이프라인 단계

### Pre 단계: rosbag → 폴더 추출
| Step | 스크립트 | 설명 |
|---|---|---|
| pre1 | `step_pre1_extract_rosbag.py` | rosbag → 폴더 (이미지/PCD/nav 분리). **최종버전** (extract2 기반: image_color/mono + progress bar) |
| pre1b | `step_pre1b_timestamp_convert.py` | rosbag header timestamp 기준 reindex |
| pre1c | `step_pre1c_rosbag_to_csv.py` | 토픽을 CSV로 export (rostopic echo) |
| pre2 | `step_pre2_validate_dataset.py` | 추출된 데이터셋 누락/sync 검증. **최종버전** (check2 기반) |
| pre3 | `step_pre3_timestamp_viz.py` | Timestamp Gantt chart (plotly) |

### Main 단계: 컬러라이즈/맵 빌드/메쉬
| Step | 스크립트 | 설명 |
|---|---|---|
| 0 | `step0_calibration_test.py` | LiDAR-Camera extrinsic 시각화로 확인 |
| 0 | `step0_vignetting_test.py` | 카메라 비네팅/마스크 인터랙티브 튜닝 |
| 0 | `step0_projection_show.py` | 멀티-LiDAR projection 오버레이 |
| 0 | `step0_trajectory_from_csv.py` | 단일 navigation.csv → trajectory plot |
| 0 | `step0_trajectory_show.py` | 폴더의 trajectory를 3D viewer로 |
| 0 | `step0_map_compare.py` | 맵 PCD vs GPS trajectory 2D 비교 이미지 |
| 0 | `step0_merge_auto.py` | radius 기반 자동 글로벌 맵 병합 |
| 0 | `step0_merge_input.py` | 입력 리스트 기반 UTM 글로벌 맵 병합 |
| 1 | `step1_nav_reader_html.py` | 전체 폴더 trajectory → plotly HTML |
| 1 | `step1_nav_reader_plot.py` | 전체 폴더 trajectory → matplotlib PNG |
| 2 | `step2_colorize_folder.py` | **단일 폴더** LiDAR-Camera 융합 |
| 3 | `step3_colorize_rosbag_deskew.py` | **rosbag** 입력 + LiDAR deskew |
| 4 | `step4_colorize_auto.py` | 자동 배치 ouster2만, GNSS 변환 |
| 5 | `step5_colorize_vgicp_pgo.py` | **자동 배치 + VGICP + PGO** (GPU) |
| 6 | `step6_mesh_nksr.py` | NKSR으로 메쉬 생성 |
| 6 | `step6_mesh_post_process.py` | 메쉬 outlier 제거/단순화 |
| 7 | `step7_mesh_poisson.py` | Chunked Poisson surface reconstruction |

## 빠른 시작

```bash
# 의존성 설치
pip install -r requirements.txt

# Pre1) rosbag → 폴더 추출
python -m src.pipeline.step_pre1_extract_rosbag --bag /path/to/data.bag

# Pre2) 데이터셋 검증
python -m src.pipeline.step_pre2_validate_dataset --folder /path/to/extracted

# 캘리브레이션 시각 확인
python -m src.pipeline.step0_calibration_test

# Main 4) 자동 배치 처리 (VGICP+PGO)
python -m src.pipeline.step5_colorize_vgicp_pgo
# 또는: ./scripts/run_step5_colorize.sh

# Main 5) 글로벌 맵 병합
python -m src.pipeline.step0_merge_auto

# Main 6) 메쉬 변환
./scripts/run_step6_mesh_batch.sh

# 전체 자동 실행
./scripts/run_full_pipeline.sh
```

## Config 사용법

모든 스크립트는 `configs/default.yaml`을 기본 로드합니다.
`--config <yaml>` 인자로 다른 yaml을 지정 가능.

```yaml
# 예: configs/extraction.yaml > extraction
extraction:
  rosbag_path: "/path/to/dataset.bag"   # ← 입력 변경
  sync_reference_topic: "/blackfly/image_raw/compressed"
  topics:
    raw_image: [...]
    compressed_image: [...]
    pointcloud: [...]
    gps: [...]
  skip_already_processed: true
```

### Config 동적 override
- CLI 플래그: `--no-vgicp`, `--no-pgo` (step5), `--bag PATH`, `--folder PATH`
- 환경 변수: `USE_VGICP=0 ./scripts/run_step5_colorize.sh`

## 주요 기능 

원래 여러 버전(`*_backup.py`, `*_original.py`, `*_gpu.py`, `extract1/2/3.py`, `check.py/check2.py` 등)에 흩어져 있던 기능을 **모두 최신 버전 한 곳으로 통합**하고 config로 켜고 끌 수 있게 만들었습니다.

| 기능 | Config 위치 |
|---|---|
| rosbag 토픽 추출 목록 | `extraction.topics.{raw_image,compressed_image,pointcloud,gps}` |
| image_color/mono 추가 토픽 | `extraction.topics.compressed_image` (주석 해제) |
| Sync 윈도우 (timestamp 매칭) | `extraction.sync_window_ns` |
| 이미 처리된 폴더 skip | `extraction.skip_already_processed` |
| Timestamp 기반 LiDAR-Image 매칭 | `colorize.processing.time_tolerance` |
| Submap chunking | `colorize.submap.frame_limit` |
| GPU 가속 (Open3D Tensor) | `colorize.gpu.enabled`, `colorize.gpu.device` |
| 카메라 비네팅 보정 | `camera.vignetting.enabled` |
| VGICP sliding window 정합 | `colorize.step5_vgicp_pgo.vgicp.*` |
| Pose Graph Optimization | `colorize.step5_vgicp_pgo.pgo.*` |
| 듀얼 GPU 라운드로빈 (병합) | `merge.auto_merge.use_dual_gpu` |

## 외부 의존성

- Python 3.8+
- ROS Noetic (step_pre1, step3) - rosbag, cv_bridge, sensor_msgs
- Open3D (with CUDA)
- PyTorch + nksr (step6)
- pygicp (step5 VGICP)
- rosbags (step3 ROS-independent rosbag 처리)
- plotly, matplotlib

자세한 NKSR 단계 가이드는 `docs/STEP6_MESH_README.md` 참조.
