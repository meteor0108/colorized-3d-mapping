"""Step9c: LiDAR Poisson 기하 + 2DGS 렌더 텍스처 베이킹 (경로 A).

결론: 기하는 LiDAR(coherent ~5cm), 외관은 2DGS(렌더 우수)가 강점.
→ ① LiDAR Poisson mesh ② 학습된 2DGS로 각 뷰 RGB+depth 렌더
   ③ mesh vertex를 여러 뷰에 투영해 색 베이킹 (occlusion은 GS depth로 z-test)

입력: step7c가 만든 dense GS dataset(+ point_cloud_2dgs.ply 체크포인트)
실행: kctc_gs env
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

from src.common import Calibration, load_config
from src.pipeline.gs_backbone_2dgs import load_2dgs


def build_lidar_mesh(ds: Path, cam_centers, voxel=0.03, depth=11, trim=0.04):
    pcd = o3d.io.read_point_cloud(str(ds / "points3D.ply"))
    pcd = pcd.voxel_down_sample(voxel)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 6, max_nn=30))
    from scipy.spatial import cKDTree
    P = np.asarray(pcd.points); N = np.asarray(pcd.normals)
    _, ci = cKDTree(cam_centers).query(P, workers=-1)
    N[np.einsum("ij,ij->i", N, cam_centers[ci] - P) < 0] *= -1
    pcd.normals = o3d.utility.Vector3dVector(N)
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=1.1, linear_fit=False)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, trim))
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def run(ds_dir, out_path, cfg, voxel=0.03, poisson_depth=11, occ_thresh=0.3):
    t0 = time.time()
    calib = Calibration.from_config(cfg)
    ds = Path(ds_dir)
    cams = json.load(open(ds / "cameras.json"))["frames"]
    frames = [{"stem": Path(c["img"]).stem, "img": c["img"], "w2c": np.array(c["w2c"]),
               "K": np.array(c["K"]), "width": int(c["width"]), "height": int(c["height"])}
              for c in cams]

    # ② 2DGS 렌더 (RGB + depth) per view
    model = load_2dgs(frames, ds, cfg.mesh.gaussian, calib)  # repo 경로를 sys.path에 추가
    from gaussian_renderer import render
    gaussians, pipe, camlist, _ = model
    bg = torch.zeros(3, device="cuda")
    views = []
    with torch.no_grad():
        for cam, fr in zip(camlist, frames):
            pkg = render(cam, gaussians, pipe, bg)
            rgb = (pkg["render"].clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()  # RGB
            d = pkg["surf_depth"][0].cpu().numpy().astype(np.float32)
            views.append((rgb, d, fr["w2c"], fr["K"], fr["width"], fr["height"]))
    print(f"[9c] 2DGS 렌더 {len(views)}뷰  {time.time()-t0:.0f}s", flush=True)

    cam_centers = np.array([np.linalg.inv(f["w2c"])[:3, 3] for f in frames])

    # ① LiDAR Poisson 기하
    mesh = build_lidar_mesh(ds, cam_centers, voxel=voxel, depth=poisson_depth)
    V = np.asarray(mesh.vertices)
    VN = np.asarray(mesh.vertex_normals)
    VN = VN / (np.linalg.norm(VN, axis=1, keepdims=True) + 1e-9)
    print(f"[9c] LiDAR Poisson mesh: {len(V):,} v / {len(mesh.triangles):,} f  {time.time()-t0:.0f}s", flush=True)

    # ③ vertex 텍스처 베이킹 (멀티뷰 + GS depth occlusion)
    acc = np.zeros((len(V), 3)); accw = np.zeros(len(V))
    for rgb, d, w2c, K, w, h in views:
        cc = np.linalg.inv(w2c)[:3, 3]
        cam_v = (w2c @ np.vstack((V.T, np.ones(len(V)))))[:3].T
        z = cam_v[:, 2]
        front = z > 0.1
        u = np.full(len(V), -1); vv = np.full(len(V), -1)
        uv = (K @ (cam_v[front] / z[front, None]).T).T
        u[front] = np.round(uv[:, 0]).astype(int); vv[front] = np.round(uv[:, 1]).astype(int)
        infov = front & (u >= 0) & (u < w) & (vv >= 0) & (vv < h)
        # 법선이 카메라 향함
        dirc = cc - V; dirn = dirc / (np.linalg.norm(dirc, axis=1, keepdims=True) + 1e-9)
        facing = np.einsum("ij,ij->i", VN, dirn)
        cand = infov & (facing > 0.1)
        ci = np.where(cand)[0]
        if len(ci) == 0:
            continue
        # mesh 자체 z-buffer로 self-occlusion (가장 가까운 면만 색칠)
        zbuf = np.full((h, w), np.inf)
        np.minimum.at(zbuf, (vv[ci], u[ci]), z[ci])
        vis = z[ci] <= zbuf[vv[ci], u[ci]] + occ_thresh
        idx = ci[vis]
        if len(idx) == 0:
            continue
        col = rgb[vv[idx], u[idx]].astype(np.float64) / 255.0  # RGB (2DGS 렌더)
        wgt = facing[idx]
        acc[idx] += col * wgt[:, None]; accw[idx] += wgt

    have = accw > 0
    vcol = np.full((len(V), 3), 0.5)
    vcol[have] = acc[have] / accw[have, None]
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(vcol, 0, 1))
    print(f"[9c] 텍스처 베이킹: {have.sum():,}/{len(V):,} vertex 색칠 ({100*have.mean():.0f}%)  {time.time()-t0:.0f}s", flush=True)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(out_path, mesh, write_vertex_colors=True)
    print(f"[9c] 저장: {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Step9c: LiDAR Poisson + 2DGS texture bake")
    ap.add_argument("--config", default="default.yaml")
    ap.add_argument("--dataset", required=True, help="step7c dense dataset (+ 2DGS 체크포인트)")
    ap.add_argument("--output", "-o", required=True)
    ap.add_argument("--voxel", type=float, default=0.03)
    ap.add_argument("--poisson-depth", type=int, default=11)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run(args.dataset, args.output, cfg, voxel=args.voxel, poisson_depth=args.poisson_depth)


if __name__ == "__main__":
    main()
