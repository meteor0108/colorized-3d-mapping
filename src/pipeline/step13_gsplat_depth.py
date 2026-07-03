"""Step13: gsplat 3DGS + LiDAR depth 감독 + sky 처리 (Street Gaussians 핵심요소, dynamic 제외).

graphdeco rasterizer의 depth가 신뢰 불가 → gsplat(expected-depth 신뢰)로 backbone 교체.
손실 = L1+SSIM(rgb) + λ_depth·L1(render_depth, LiDAR_depth) + λ_sky·alpha(sky).
densification = gsplat DefaultStrategy. 학습 후 보간 flythrough.

sky 마스크(모델 없이): 상단영역 & LiDAR 무반사 & 밝음 (LiDAR는 하늘에 안 맞음).
실행: kctc_gs env (gsplat)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot, Slerp

from src.common import Calibration, load_config

C0 = 0.28209479177387814  # SH DC


def _ssim(a, b):  # a,b: (1,3,H,W)
    win = torch.ones(3, 1, 11, 11, device=a.device) / (11 * 11)
    mu_a = F.conv2d(a, win, padding=5, groups=3); mu_b = F.conv2d(b, win, padding=5, groups=3)
    va = F.conv2d(a * a, win, padding=5, groups=3) - mu_a ** 2
    vb = F.conv2d(b * b, win, padding=5, groups=3) - mu_b ** 2
    vab = F.conv2d(a * b, win, padding=5, groups=3) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * vab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    return s.mean()


def _sky_mask(img_bgr, lidar_depth, calib):
    h, w = lidar_depth.shape
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    yy, xx = np.mgrid[0:h, 0:w]
    sky = (yy < 0.55 * h) & (lidar_depth <= 0) & (gray > 135)
    mp = calib.mask_params()
    if mp is not None:                                  # 비네팅 원형 밖 제외
        cx, cy, r = mp; sky &= ((xx - cx) ** 2 + (yy - cy) ** 2) < r ** 2
    return sky


def _load(ds, frames, calib):
    """gt(0..1, HWC), lidar depth(m), sky mask 를 GPU 텐서로."""
    data = []
    for f in frames:
        bgr = cv2.imread(str(ds / f["img"]))
        gt = torch.from_numpy(bgr[:, :, ::-1].copy()).float() / 255.0   # H,W,3 RGB (CPU)
        ld = np.load(ds / "depth" / f"{f['stem']}.npy")
        sky = _sky_mask(bgr, ld, calib)
        data.append((gt, torch.from_numpy(ld).float(), torch.from_numpy(sky),   # CPU (per-iter GPU 전송)
                     torch.from_numpy(f["w2c"]).float().cuda(),
                     torch.from_numpy(f["K"]).float().cuda()))
    return data


def train(ds, frames, init_pcd, cfg, densify=True):
    from gsplat import rasterization, MCMCStrategy
    gs = cfg.mesh.gaussian; tcfg = gs.train
    iters = int(tcfg.iterations)
    lam_d = float(getattr(tcfg, "lambda_depth", 0.5))
    lam_sky = float(getattr(tcfg, "lambda_sky", 0.05))
    sh_deg = int(gs.sh_degree)
    calib = Calibration.from_config(cfg)
    data = _load(ds, frames, calib)

    pts = np.asarray(init_pcd.points); cols = np.clip(np.asarray(init_pcd.colors), 0, 1)
    maxN = int(getattr(tcfg, "max_init_points", 1400000))
    if len(pts) > maxN:
        s = np.random.default_rng(0).choice(len(pts), maxN, replace=False); pts, cols = pts[s], cols[s]
    centers = np.array([np.linalg.inv(f["w2c"])[:3, 3] for f in frames])
    scene = float(np.linalg.norm(centers - centers.mean(0), axis=1).max()) + 1e-6

    dist, _ = cKDTree(pts).query(pts, k=4); sc = np.log(np.clip(dist[:, 1:].mean(1), 1e-3, None))
    dev = "cuda"
    means = torch.nn.Parameter(torch.tensor(pts, dtype=torch.float, device=dev))
    scales = torch.nn.Parameter(torch.tensor(sc[:, None].repeat(3, 1), dtype=torch.float, device=dev))
    quats = torch.nn.Parameter(torch.tensor([1., 0, 0, 0], device=dev).repeat(len(pts), 1))
    opac = torch.nn.Parameter(torch.logit(0.1 * torch.ones(len(pts), device=dev)))
    K_sh = (sh_deg + 1) ** 2
    sh0 = torch.nn.Parameter(((torch.tensor(cols, dtype=torch.float, device=dev) - 0.5) / C0)[:, None, :])
    shN = torch.nn.Parameter(torch.zeros(len(pts), K_sh - 1, 3, device=dev))
    params = torch.nn.ParameterDict({"means": means, "scales": scales, "quats": quats,
                                     "opacities": opac, "sh0": sh0, "shN": shN})
    lrs = {"means": 1.6e-4 * scene, "scales": 5e-3, "quats": 1e-3,
           "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    opt = {k: torch.optim.Adam([params[k]], lr=lrs[k], eps=1e-15) for k in params}
    strat = state = None
    cap = min(int(getattr(tcfg, "max_gaussians", 2500000)), 2500000)  # 하드 상한(OOM 방지)
    if densify:
        strat = MCMCStrategy(cap_max=cap, refine_start_iter=500, refine_stop_iter=15000, refine_every=100)
        strat.check_sanity(params, opt)
        state = strat.initialize_state()
    print(f"[13] gsplat 학습: N={len(pts):,} scene={scene:.1f} λdepth={lam_d} λsky={lam_sky} "
          f"densify={densify}", flush=True)

    from tqdm import tqdm
    from random import randint
    stack = []
    for step in tqdm(range(iters), desc="[13] gsplat"):
        if not stack:
            stack = list(range(len(data)))
        gt, ld, sky, w2c, K = data[stack.pop(randint(0, len(stack) - 1))]
        gt = gt.cuda(non_blocking=True); ld = ld.cuda(non_blocking=True); sky = sky.cuda(non_blocking=True)
        H, W = gt.shape[:2]
        cur_sh = min(sh_deg, step // 1000)
        rc, ra, info = rasterization(
            params["means"], params["quats"], torch.exp(params["scales"]), torch.sigmoid(params["opacities"]),
            torch.cat([params["sh0"], params["shN"]], 1), w2c[None], K[None], W, H,
            sh_degree=cur_sh, render_mode="RGB+ED", packed=False, near_plane=0.01, far_plane=1000.0)
        rgb = rc[0, ..., :3].clamp(0, 1); dep = rc[0, ..., 3]; alp = ra[0, ..., 0]
        l1 = (rgb - gt).abs().mean()
        loss = 0.8 * l1 + 0.2 * (1 - _ssim(rgb.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
        m = ld > 0
        if m.any():
            loss = loss + lam_d * (dep[m] - ld[m]).abs().mean()
        if sky.any():
            loss = loss + lam_sky * alp[sky].mean()

        loss.backward()
        for o in opt.values():
            o.step(); o.zero_grad(set_to_none=True)
        if densify:                                    # MCMC: optimizer.step 후 relocation+noise
            strat.step_post_backward(params, opt, state, step, info, lr=lrs["means"])

    print(f"[13] 학습완료 N={params['means'].shape[0]:,}", flush=True)
    means, scales, quats, opac, sh0, shN = (params["means"], params["scales"], params["quats"],
                                            params["opacities"], params["sh0"], params["shN"])
    return params, sh_deg, frames[0]["K"], frames[0]["width"], frames[0]["height"]


def _interp(c2w, n):
    out = []
    for i in range(len(c2w) - 1):
        A, B = c2w[i], c2w[i + 1]
        sl = Slerp([0, 1], Rot.from_matrix(np.stack([A[:3, :3], B[:3, :3]])))
        for t in np.linspace(0, 1, n, endpoint=False):
            T = np.eye(4); T[:3, :3] = sl([t])[0].as_matrix(); T[:3, 3] = (1 - t) * A[:3, 3] + t * B[:3, 3]
            out.append(T)
    out.append(c2w[-1]); return out


def render_video(params, sh_deg, K, W, H, frames, out_mp4, interp=5, fps=30):
    from gsplat import rasterization
    Kt = torch.tensor(K, dtype=torch.float, device="cuda")
    poses = _interp([np.linalg.inv(f["w2c"]) for f in frames], interp)
    out = Path(out_mp4); out.parent.mkdir(parents=True, exist_ok=True)
    fdir = out.parent / (out.stem + "_frames"); fdir.mkdir(exist_ok=True)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    with torch.no_grad():
        for i, c2w in enumerate(poses):
            w2c = torch.tensor(np.linalg.inv(c2w), dtype=torch.float, device="cuda")
            rc, _, _ = rasterization(params["means"], params["quats"], torch.exp(params["scales"]),
                                     torch.sigmoid(params["opacities"]),
                                     torch.cat([params["sh0"], params["shN"]], 1), w2c[None], Kt[None],
                                     W, H, sh_degree=sh_deg, render_mode="RGB", packed=False)
            bgr = (rc[0, ..., :3].clamp(0, 1) * 255).byte().cpu().numpy()[:, :, ::-1]
            vw.write(np.ascontiguousarray(bgr)); cv2.imwrite(str(fdir / f"{i:04d}.png"), bgr)
            if i % 30 == 0:
                print(f"  render {i}/{len(poses)}", flush=True)
    vw.release()
    print(f"[13] 저장: {out} ({len(poses)}f)", flush=True)


def main():
    import open3d as o3d
    ap = argparse.ArgumentParser(description="Step13: gsplat 3DGS + LiDAR depth + sky")
    ap.add_argument("--config", default="default.yaml")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", "-o", required=True)
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--interp", type=int, default=5)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.iterations:
        cfg.mesh.gaussian.train["iterations"] = args.iterations
    ds = Path(args.dataset)
    cams = json.load(open(ds / "cameras.json"))["frames"]
    frames = [{"stem": Path(c["img"]).stem, "img": c["img"], "w2c": np.array(c["w2c"]),
               "K": np.array(c["K"]), "width": int(c["width"]), "height": int(c["height"])} for c in cams]
    init = o3d.io.read_point_cloud(str(ds / "points3D.ply"))
    params, sh_deg, K, W, H = train(ds, frames, init, cfg)
    torch.save({k: v.detach().cpu() for k, v in params.items()} | {"sh_degree": sh_deg},
               ds / "gsplat_params.pt")                       # off-axis 재렌더용
    print(f"[13] params 저장: {ds/'gsplat_params.pt'} (N={params['means'].shape[0]:,})", flush=True)
    render_video(params, sh_deg, K, W, H, frames, args.output, interp=args.interp)


if __name__ == "__main__":
    main()
