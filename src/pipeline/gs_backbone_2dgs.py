"""2D Gaussian Splatting backbone 어댑터 (hbb1/2d-gaussian-splatting).

step8_mesh_gaussian.py가 호출. step7b dataset(cameras.json/images/depth/points3D.ply)을
2DGS의 GaussianModel/Camera로 변환하고, LiDAR depth L1 + 2DGS 정규화(normal/dist)로 학습한
뒤 각 뷰 dense depth를 렌더한다.

2DGS repo를 라이브러리로 import (sys.path). diff-surfel-rasterization 컴파일 필요:
    TORCH_CUDA_ARCH_LIST=8.9 pip install ./submodules/diff-surfel-rasterization ./submodules/simple-knn

설계 메모:
- Camera(R,T): getWorld2View2가 world2view=[R.T | T]를 만들므로 R=R_w2c.T, T=t_w2c.
- TSDF 입력은 우리 cameras.json의 w2c(OpenCV) + K + 렌더 surf_depth(카메라 z) 재사용.
- bounded submap이라 depth_ratio=1(median depth) 권장.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# CUDA 메모리 단편화 완화 (첫 cuda alloc 전에 설정되어야 함)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import torch


def _add_repo_to_path(repo_path: str) -> Path:
    repo = Path(repo_path)
    if not repo.is_absolute():
        # configs는 Data_Preprocessing/ 기준 상대경로
        repo = (Path(__file__).resolve().parents[2] / repo_path).resolve()
    if not (repo / "gaussian_renderer" / "__init__.py").exists():
        raise SystemExit(
            f"2DGS repo를 찾을 수 없음: {repo}\n"
            "  git clone --recursive https://github.com/hbb1/2d-gaussian-splatting.git")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


def _focal2fov(focal: float, pixels: float) -> float:
    return 2 * np.arctan(pixels / (2 * focal))


def _default_opt(train_cfg, sh_degree: int) -> SimpleNamespace:
    """arguments/OptimizationParams 기본값 + config override."""
    iters = int(train_cfg.iterations)
    return SimpleNamespace(
        iterations=iters,
        position_lr_init=0.00016, position_lr_final=0.0000016,
        position_lr_delay_mult=0.01, position_lr_max_steps=iters,
        feature_lr=0.0025, opacity_lr=0.05, scaling_lr=0.005, rotation_lr=0.001,
        percent_dense=0.01,
        lambda_dssim=float(train_cfg.lambda_dssim),
        lambda_dist=float(train_cfg.lambda_dist),
        lambda_normal=float(train_cfg.lambda_normal),
        opacity_cull=0.05,
        densification_interval=100, opacity_reset_interval=3000,
        densify_from_iter=500,
        densify_until_iter=min(15000, max(7000, iters // 2)),
        densify_grad_threshold=0.0002,
        sh_degree=sh_degree,
    )


def _build_cameras(frames, dataset_dir: Path, calib, undistorted: bool):
    """cameras.json frame → 2DGS Camera 리스트 + LiDAR depth 텐서."""
    from scene.cameras import Camera

    mask_params = calib.mask_params() if calib else None
    cutoff_y = calib.cutoff_y() if calib else None

    cams, depths, centers = [], [], []
    for i, fr in enumerate(frames):
        img = cv2.imread(str(dataset_dir / fr["img"]))[:, :, ::-1]  # BGR→RGB
        h, w = img.shape[:2]
        K = np.array(fr["K"], np.float64)
        w2c = np.array(fr["w2c"], np.float64)
        R = w2c[:3, :3].T          # getWorld2View2 컨벤션
        T = w2c[:3, 3]
        fovx = _focal2fov(K[0, 0], w)
        fovy = _focal2fov(K[1, 1], h)

        img_t = torch.from_numpy(np.ascontiguousarray(img)).float().permute(2, 0, 1) / 255.0

        # alpha mask: 원형 비네팅 + 차량 하단 제외 (loss에서 무시)
        alpha = None
        if mask_params is not None or cutoff_y is not None:
            m = np.ones((h, w), np.float32)
            yy, xx = np.mgrid[0:h, 0:w]
            if mask_params is not None:
                mcx, mcy, mr = mask_params
                m[((xx - mcx) ** 2 + (yy - mcy) ** 2) > mr ** 2] = 0.0
            if cutoff_y is not None:
                m[yy >= cutoff_y] = 0.0
            alpha = torch.from_numpy(m)[None]

        cam = Camera(colmap_id=i, R=R, T=T, FoVx=fovx, FoVy=fovy,
                     image=img_t, gt_alpha_mask=alpha,
                     image_name=fr["stem"], uid=i)
        cams.append(cam)
        centers.append(np.linalg.inv(w2c)[:3, 3])

        dpath = dataset_dir / "depth" / f"{fr['stem']}.npy"
        d = np.load(dpath) if dpath.exists() else np.zeros((h, w), np.float32)
        depths.append(torch.from_numpy(d.astype(np.float32)))  # CPU; per-iter로 GPU 전송

    centers = np.stack(centers)
    radius = float(np.linalg.norm(centers - centers.mean(0), axis=1).max()) * 1.1 + 1e-6
    return cams, depths, radius


def train_2dgs(frames, dataset_dir, init_pcd, gs_cfg, calib):
    """2DGS 학습 → (gaussians, pipe, cameras) 반환."""
    repo = _add_repo_to_path(str(gs_cfg.repo_path))
    print(f"[2dgs] repo: {repo}")
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from utils.graphics_utils import BasicPointCloud
    from utils.loss_utils import l1_loss, ssim
    from random import randint

    tcfg = gs_cfg.train
    opt = _default_opt(tcfg, int(gs_cfg.sh_degree))
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False,
                           depth_ratio=float(gs_cfg.depth_ratio))
    bg = torch.zeros(3, device="cuda")

    cams, depths, radius = _build_cameras(frames, Path(dataset_dir), calib,
                                          undistorted=bool(gs_cfg.dataset.undistort))
    lr_scale = min(radius, float(getattr(tcfg, "lr_scale_cap", 25.0)))
    max_gaussians = int(getattr(tcfg, "max_gaussians", 3000000))
    max_init = int(getattr(tcfg, "max_init_points", 1200000))
    print(f"[2dgs] cameras={len(cams)} | scene radius={radius:.1f}m | lr_scale={lr_scale:.1f}")
    if radius > 300:
        print(f"[2dgs] ⚠️  scene radius {radius:.0f}m — bounded submap 아님. "
              f"step7b --max-frames 로 윈도우를 끊는 것을 권장 (sparse-view + 발산 위험).")

    pts = np.asarray(init_pcd.points)
    cols = np.asarray(init_pcd.colors)
    if len(pts) > max_init:                       # init Gaussian 상한 (optimizer VRAM)
        sel = np.random.default_rng(0).choice(len(pts), max_init, replace=False)
        pts, cols = pts[sel], cols[sel]
        print(f"[2dgs] init 다운샘플 {len(init_pcd.points):,} → {len(pts):,}")
    pcd = BasicPointCloud(points=pts, colors=cols, normals=np.zeros_like(pts))

    gaussians = GaussianModel(opt.sh_degree)
    gaussians.create_from_pcd(pcd, lr_scale)      # lr은 clamp된 scale로, densify extent는 radius
    gaussians.training_setup(opt)

    lambda_depth = float(tcfg.lambda_depth)
    depth_ramp = max(1, int(tcfg.depth_ramp_iters))

    from tqdm import tqdm
    stack = []
    pbar = tqdm(range(1, opt.iterations + 1), desc="[2dgs] train")
    for it in pbar:
        gaussians.update_learning_rate(it)
        if it % 1000 == 0:
            gaussians.oneupSHdegree()
        if not stack:
            stack = list(range(len(cams)))
        idx = stack.pop(randint(0, len(stack) - 1))
        cam = cams[idx]

        pkg = render(cam, gaussians, pipe, bg)
        image = pkg["render"]
        gt = cam.original_image.cuda()

        if cam.gt_alpha_mask is not None:
            mask = cam.gt_alpha_mask.cuda()
            image = image * mask + bg[:, None, None] * (1 - mask)
            gt = gt * mask + bg[:, None, None] * (1 - mask)

        Ll1 = l1_loss(image, gt)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt))

        # LiDAR sparse depth L1 (저텍스처/원거리 기하 고정)
        ld = depths[idx].cuda(non_blocking=True)   # CPU→GPU per-iter
        dmask = ld > 0
        if cam.gt_alpha_mask is not None:
            dmask = dmask & (cam.gt_alpha_mask.cuda()[0] > 0)
        if dmask.any():
            surf = pkg["surf_depth"][0]
            w_depth = lambda_depth * min(1.0, it / depth_ramp)
            loss = loss + w_depth * torch.abs(surf[dmask] - ld[dmask]).mean()

        # 2DGS 정규화 (warm-up 후 활성)
        lam_n = opt.lambda_normal if it > 7000 else 0.0
        lam_d = opt.lambda_dist if it > 3000 else 0.0
        rend_normal, surf_normal = pkg["rend_normal"], pkg["surf_normal"]
        normal_loss = lam_n * (1 - (rend_normal * surf_normal).sum(0))[None].mean()
        dist_loss = lam_d * pkg["rend_dist"].mean()
        loss = loss + normal_loss + dist_loss

        loss.backward()

        with torch.no_grad():
            vis = pkg["visibility_filter"]
            radii = pkg["radii"]
            # Gaussian 예산 초과 시 densify 중단 (OOM 방지). prune/reset은 유지.
            if it < opt.densify_until_iter and len(gaussians.get_xyz) < max_gaussians:
                gaussians.max_radii2D[vis] = torch.max(gaussians.max_radii2D[vis], radii[vis])
                gaussians.add_densification_stats(pkg["viewspace_points"], vis)
                if it > opt.densify_from_iter and it % opt.densification_interval == 0:
                    size_thr = 20 if it > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull,
                                                radius, size_thr)
                if it % opt.opacity_reset_interval == 0:
                    gaussians.reset_opacity()
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            if it % 50 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}", pts=len(gaussians.get_xyz))

    # 학습 결과 체크포인트 저장 → mesh 재추출 시 재학습 불필요 (--extract-only)
    ckpt = Path(dataset_dir) / "point_cloud_2dgs.ply"
    try:
        gaussians.save_ply(str(ckpt))
        print(f"[2dgs] 체크포인트 저장: {ckpt} ({len(gaussians.get_xyz):,} gaussians)")
    except Exception as e:
        print(f"[2dgs] 체크포인트 저장 실패(무시): {e}")

    return gaussians, pipe, cams, frames


def load_2dgs(frames, dataset_dir, gs_cfg, calib):
    """저장된 체크포인트(point_cloud_2dgs.ply)에서 gaussians 로드 (학습 생략)."""
    repo = _add_repo_to_path(str(gs_cfg.repo_path))
    from scene.gaussian_model import GaussianModel
    ckpt = Path(dataset_dir) / "point_cloud_2dgs.ply"
    if not ckpt.exists():
        raise SystemExit(f"체크포인트 없음: {ckpt} — 먼저 학습(--extract-only 없이) 필요")
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False,
                           depth_ratio=float(gs_cfg.depth_ratio))
    cams, _, _ = _build_cameras(frames, Path(dataset_dir), calib,
                               undistorted=bool(gs_cfg.dataset.undistort))
    g = GaussianModel(int(gs_cfg.sh_degree))
    g.load_ply(str(ckpt))
    g.active_sh_degree = g.max_sh_degree
    print(f"[2dgs] 체크포인트 로드: {ckpt} ({len(g.get_xyz):,} gaussians)")
    return g, pipe, cams, frames


def render_depths_2dgs(model, frames, alpha_thresh=0.5):
    """학습된 gaussians로 각 뷰 dense rgb/depth 렌더 → TSDF 입력 dict 리스트.
    rend_alpha < alpha_thresh(하늘/미렌더)는 depth=0으로 무시."""
    from gaussian_renderer import render
    gaussians, pipe, cams, frames = model
    bg = torch.zeros(3, device="cuda")
    views = []
    with torch.no_grad():
        for cam, fr in zip(cams, frames):
            pkg = render(cam, gaussians, pipe, bg)
            rgb = (pkg["render"].clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
            depth_t = pkg["surf_depth"][0]
            alpha = pkg["rend_alpha"][0]
            depth_t = torch.where(alpha < alpha_thresh, torch.zeros_like(depth_t), depth_t)
            depth = depth_t.cpu().numpy().astype(np.float32)
            views.append({
                "rgb": np.ascontiguousarray(rgb),
                "depth": depth,
                "w2c": np.array(fr["w2c"], np.float64),
                "K": np.array(fr["K"], np.float64),
                "width": int(fr["width"]),
                "height": int(fr["height"]),
            })
    return views
