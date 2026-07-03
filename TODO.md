# TODO / Roadmap

맵 품질 개선을 위한 진행 과제. 배경: 오프라인 SLAM(GNSS/INS CPT7 + VGICP 포즈) → 동적 제거 →
Gaussian Splatting 재구성. **대부분 데이터가 single-pass(주행 1회)** 라 photometric multi-view가
부족 → 기하 prior(LiDAR/mono depth) 주입이 품질의 핵심.

진단 결론 요약:
- 병목 1순위 = **single-pass(멀티뷰 부족)** → depth 감독을 dense로 강화해야 함.
- 병목 2순위 = **extrinsic + time-sync**(CPT7 pose는 INS body 기준이라 카메라 정합은 캘리브가 좌우).
- orientation(CPT7 SPAN)은 주범 아님. depth supervision은 **이미 배선돼 있으나 sparse LiDAR만** 사용.

---

## P0 — depth/*.npy 반전 버그 검증·픽스 (최우선, 착수 전 필수)

GS 두 백본이 `depth/*.npy`를 그대로 depth 감독에 사용하는데,
`step14_mono_depth_fusion.py` docstring이 *"step7b/7c 저장 depth/*.npy 는 반전 버그가 있어 사용 안 함"* 이라고 명시.
→ 사실이면 현재 depth loss가 품질을 **악화**시키는 중.

- [ ] `depth/<stem>.npy` 1~2장을 이미지 위에 컬러 오버레이해 근/원 순서 확인 (step7b `overlay/*.jpg` 우선 확인).
- [ ] 원인 추정 검증: `depth_buf[v,u] = np.where(...)` **fancy-index scatter가 같은 픽셀 충돌 시 min이 아닌 마지막 값**을 취함
      → 먼 점이 가까운 점을 덮어써 z-buffer가 깨짐. (step7b:167, step7c:136-137 동일 패턴)
- [ ] 픽스: 픽셀별 min z-buffer를 `np.minimum.at()` 또는 정렬 후 scatter로 교체.
- [ ] 버그 확정 시, 픽스 전까지 `lambda_depth=0`으로 두고 재학습 baseline 확보.

관련: `step7b_prepare_gs_dataset.py`, `step7c_prepare_gs_dense.py`

---

## P1 — sparse LiDAR depth → dense depth 감독 (핵심 기능)

single-pass의 빈 픽셀(원거리·지평선·면 전체)을 dense depth로 메워 loss 커버리지 확대.
step14의 검증된 mono+LiDAR 정렬 로직을 **GS 감독용 dense depth exporter**로 재사용.

- [ ] dense depth exporter 신설 (프레임별):
  - [ ] `D_lidar` = sparse LiDAR metric depth (anchor, conf=1.0)
  - [ ] `D_mono`  = Depth-Anything-V2-Metric-Outdoor (step14 재사용)
  - [ ] **affine 정렬**: `D_lidar ≈ s·D_mono + t` robust fit (현재 step14는 scale-only median → affine으로 업그레이드)
  - [ ] **fuse**: LiDAR 유효=LiDAR, 나머지=aligned mono → dense `depth/<stem>.npy`
  - [ ] **confidence map** `depth_conf/<stem>.npy`: LiDAR=1.0, mono-fill=0.1~0.3
        (원거리·depth/color edge 감쇠, sky/차량/비네팅/dynmask=0)
- [ ] dynamic mask(`dynmask/`) 영역 depth/conf=0 처리로 이동체 기하 주입 차단.

관련: `step14_mono_depth_fusion.py`, `gs_backbone_2dgs.py`, `step13_gsplat_depth.py`

---

## P2 — loss / pose 정련

- [ ] depth loss를 confidence 가중 L1로 교체:
      `loss += w_depth * (W*(render_depth - D_dense).abs()).sum() / W.sum().clamp_min(1)`
      (LiDAR 픽셀 지배 + mono-fill은 약하게 → 틀린 mono 기하 hard 주입 방지)
      · `gs_backbone_2dgs.py:184-190`, `step13_gsplat_depth.py:122-124`
- [ ] **GS camera pose-opt 활성화**(`mesh.gaussian.train.optimize_camera_pose: true`) —
      VGICP는 LiDAR-to-LiDAR만 맞춤, sub-pixel 잔차를 photometric으로 마무리.
- [ ] appearance/exposure 보정(프레임 간 노출·AWB 불일치) — per-frame appearance embedding 검토.

---

## Preconditions / 진단 (P1 착수 전 통과 권장)

- [ ] **time-sync·extrinsic 정량 검증**: LiDAR reprojection 오차가 차속에 비례하면 타이밍 문제.
      dense depth는 오차를 전 픽셀에 퍼뜨리므로 이게 선결.
- [ ] **결합 테스트**: VGICP pose로 누적 LiDAR를 RGB에 투영 →
      (1) double-wall 없음, (2) depth 불연속이 RGB edge에 정렬 → 통과 시 depth 감독 진행.
- [ ] **VGICP가 IMU/CPT7 anchor 결합인지 확인**: 순수 scan-matching이면 개활지(야지)에서 drift →
      IMU factor 1개 추가로 안정화.

---

## 방향성 / 실험 (중기)

- [ ] **driving-scene / LiDAR-IC GS 벤치마크**: 한 구간에 Gaussian-LIC / GS-LIVM 또는
      Street Gaussians / OmniRe를 돌려 기존 파이프라인 대비 품질 격차 측정 → backbone 교체 판단.
- [ ] **submap 타일링**: 긴 주행은 bounded submap으로 끊어 학습(발산·blur 방지). 2DGS geometry 우선 유지.
- [ ] 멀티카메라 rig 여부 확인 — 동기 다중 카메라면 baseline 확보로 single-pass 약점 완화.

---

## Repo 정리 (hygiene)

- [ ] `third_party/`(2DGS/3DGS) 설치·CUDA arch 빌드 문서화 (README에 요약, 상세는 docs/).
- [ ] GS 파이프라인 상세 문서 `docs/GS_DEPTH_README.md` 작성 (step7b~14 흐름 + config).
- [ ] 실험 스크립트(step9b/9c, step11/12, step14_seq) core vs experimental 구분/정리.
- [ ] `mesh.gaussian.*` config 기본값을 dense-depth 감독 기준으로 갱신 (P1 반영 후).
