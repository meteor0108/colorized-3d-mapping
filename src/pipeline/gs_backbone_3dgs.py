"""3D Gaussian Splatting backbone 어댑터 (graphdeco-inria/gaussian-splatting).

렌더링 산출물 목적: 2DGS(surfel, mesh용)보다 3DGS(3D 타원체)가 더 sharp.
같은 GS dataset(images/cameras.json/points3D.ply) 재사용, photometric(L1+SSIM)만으로 학습.
복잡한 Camera 클래스 대신 render가 읽는 필드만 가진 최소 cam 사용.

실행: kctc_gs env (diff-gaussian-rasterization 컴파일 필요)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import torch


def _add_repo(repo_path: str) -> Path:
    repo = Path(repo_path)
    if not repo.is_absolute():
        repo = (Path(__file__).resolve().parents[2] / repo_path).resolve()
    if not (repo / "gaussian_renderer" / "__init__.py").exists():
        raise SystemExit(f"3DGS repo 없음: {repo}\n  git clone --recursive "
                         "https://github.com/graphdeco-inria/gaussian-splatting.git")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


def _fov(focal, px):
    return 2 * np.arctan(px / (2 * focal))


def make_cam(w2c, FoVx, FoVy, W, H, gt=None, znear=0.01, zfar=100.0):
    """render가 읽는 필드만 가진 최소 카메라."""
    from utils.graphics_utils import getWorld2View2, getProjectionMatrix
    R = w2c[:3, :3].T; T = w2c[:3, 3]
    wv = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).float().cuda()
    proj = getProjectionMatrix(znear, zfar, FoVx, FoVy).transpose(0, 1).float().cuda()
    full = (wv.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
    return SimpleNamespace(FoVx=float(FoVx), FoVy=float(FoVy), image_height=int(H),
                           image_width=int(W), world_view_transform=wv, full_proj_transform=full,
                           camera_center=wv.inverse()[3, :3], image_name="", original_image=gt)


def _opt(train_cfg, sh_degree):
    it = int(train_cfg.iterations)
    g = lambda k, d: float(getattr(train_cfg, k, d))
    return SimpleNamespace(
        iterations=it, position_lr_init=0.00016, position_lr_final=0.0000016,
        position_lr_delay_mult=0.01, position_lr_max_steps=it,
        feature_lr=0.0025, opacity_lr=0.025, scaling_lr=0.005, rotation_lr=0.001,
        exposure_lr_init=0.0, exposure_lr_final=0.0, exposure_lr_delay_steps=0,
        exposure_lr_delay_mult=0.0, percent_dense=0.01,
        lambda_dssim=float(train_cfg.lambda_dssim),
        densification_interval=100, opacity_reset_interval=3000, densify_from_iter=500,
        densify_until_iter=int(g("densify_until_iter", min(15000, max(7000, it // 2)))),
        densify_grad_threshold=g("densify_grad_threshold", 0.0002),
        sh_degree=int(sh_degree),
    )


def train_3dgs(frames, ds_dir, init_pcd, gs_cfg, calib):
    _add_repo(str(gs_cfg.repo_path_3dgs))
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from utils.graphics_utils import BasicPointCloud
    from utils.loss_utils import l1_loss, ssim
    from random import randint
    from tqdm import tqdm

    ds = Path(ds_dir)
    tcfg = gs_cfg.train
    opt = _opt(tcfg, gs_cfg.sh_degree)
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False,
                           debug=False, antialiasing=bool(getattr(tcfg, "antialiasing", False)))
    bg = torch.zeros(3, device="cuda")
    K = frames[0]["K"]; W, H = frames[0]["width"], frames[0]["height"]
    FoVx, FoVy = _fov(K[0, 0], W), _fov(K[1, 1], H)

    centers = np.array([np.linalg.inv(f["w2c"])[:3, 3] for f in frames])
    radius = float(np.linalg.norm(centers - centers.mean(0), axis=1).max()) * 1.1 + 1e-6
    lr_scale = min(radius, float(getattr(tcfg, "lr_scale_cap", 25.0)))
    max_g = int(getattr(tcfg, "max_gaussians", 3000000))
    max_init = int(getattr(tcfg, "max_init_points", 1200000))
    print(f"[3dgs] cams={len(frames)} radius={radius:.1f} lr_scale={lr_scale:.1f}", flush=True)

    cams = []
    for f in frames:
        img = cv2.imread(str(ds / f["img"]))[:, :, ::-1]  # BGR→RGB
        gt = torch.from_numpy(np.ascontiguousarray(img)).float().permute(2, 0, 1) / 255.0
        c = make_cam(f["w2c"], FoVx, FoVy, W, H, gt=gt)
        c.image_name = f["stem"]                       # create_from_pcd 의 exposure_mapping 용 (고유)
        cams.append(c)

    pts = np.asarray(init_pcd.points); cols = np.asarray(init_pcd.colors)
    if len(pts) > max_init:
        sel = np.random.default_rng(0).choice(len(pts), max_init, replace=False)
        pts, cols = pts[sel], cols[sel]
    pcd = BasicPointCloud(points=pts, colors=cols, normals=np.zeros_like(pts))
    g = GaussianModel(opt.sh_degree)
    g.create_from_pcd(pcd, cams, lr_scale)             # cam_infos = 객체 리스트(.image_name 필요)
    g.training_setup(opt)

    # --- LiDAR 기하 제약 (rasterizer depth 불가 → 표면-밖 Gaussian prune + scale reg) ---
    from scipy.spatial import cKDTree
    lidar_tree = cKDTree(np.asarray(init_pcd.points))
    lam_scale = float(getattr(tcfg, "lambda_scale", 0.0))
    max_scale_m = float(getattr(tcfg, "max_scale_m", 0.3))
    prune_dist = float(getattr(tcfg, "lidar_prune_dist", 1.0))
    prune_int = int(getattr(tcfg, "lidar_prune_interval", 1000))
    print(f"[3dgs] LiDAR 제약: scale_reg λ={lam_scale} (>{max_scale_m}m), "
          f"prune>{prune_dist}m every {prune_int}it", flush=True)

    stack = []
    for it in tqdm(range(1, opt.iterations + 1), desc="[3dgs] train"):
        g.update_learning_rate(it)
        if it % 1000 == 0:
            g.oneupSHdegree()
        if not stack:
            stack = list(range(len(cams)))
        cam = cams[stack.pop(randint(0, len(stack) - 1))]
        pkg = render(cam, g, pipe, bg)
        image = pkg["render"]
        gt = cam.original_image.cuda()
        Ll1 = l1_loss(image, gt)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt))
        if lam_scale > 0:                              # 바늘형(큰 scale) Gaussian penalty
            loss = loss + lam_scale * torch.relu(g.get_scaling - max_scale_m).mean()
        loss.backward()
        with torch.no_grad():
            vis = pkg["visibility_filter"]; radii = pkg["radii"]
            if it < opt.densify_until_iter and len(g.get_xyz) < max_g:
                g.max_radii2D[vis] = torch.max(g.max_radii2D[vis], radii[vis])
                g.add_densification_stats(pkg["viewspace_points"], vis)
                if it > opt.densify_from_iter and it % opt.densification_interval == 0:
                    sz = 20 if it > opt.opacity_reset_interval else None
                    g.densify_and_prune(opt.densify_grad_threshold, 0.005, radius, sz, radii)
                if it % opt.opacity_reset_interval == 0:
                    g.reset_opacity()
            # LiDAR 표면에서 먼 Gaussian prune (floater 제거 = LiDAR 기하 감독)
            if prune_dist > 0 and it > opt.densify_from_iter and it < opt.densify_until_iter \
                    and it % prune_int == 0:
                xyz = g.get_xyz.detach().cpu().numpy()
                dist, _ = lidar_tree.query(xyz, workers=-1)
                pm = torch.tensor(dist > prune_dist, device="cuda")
                if pm.any():
                    g.tmp_radii = torch.zeros(xyz.shape[0], device="cuda")
                    g.prune_points(pm)
            g.optimizer.step(); g.optimizer.zero_grad(set_to_none=True)

    ckpt = ds / "point_cloud_3dgs.ply"
    try:
        g.save_ply(str(ckpt)); print(f"[3dgs] 체크포인트 저장: {ckpt} ({len(g.get_xyz):,})", flush=True)
    except Exception as e:
        print(f"[3dgs] save 실패: {e}", flush=True)
    return g, pipe, FoVx, FoVy, W, H


def load_3dgs(frames, ds_dir, gs_cfg):
    _add_repo(str(gs_cfg.repo_path_3dgs))
    from scene.gaussian_model import GaussianModel
    ds = Path(ds_dir)
    ckpt = ds / "point_cloud_3dgs.ply"
    if not ckpt.exists():
        raise SystemExit(f"체크포인트 없음: {ckpt}")
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False,
                           debug=False, antialiasing=bool(getattr(gs_cfg.train, "antialiasing", False)))
    g = GaussianModel(int(gs_cfg.sh_degree)); g.load_ply(str(ckpt)); g.active_sh_degree = g.max_sh_degree
    K = frames[0]["K"]; W, H = frames[0]["width"], frames[0]["height"]
    return g, pipe, _fov(K[0, 0], W), _fov(K[1, 1], H), W, H
