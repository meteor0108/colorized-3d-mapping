"""Step14: 단안 metric depth(Depth-Anything-V2) + LiDAR 스케일 정렬 + TSDF 융합 → dense 컬러 메쉬.

GS의 sparse-view 약점도, Poisson의 LiDAR 해상도 한계(~5cm)도 우회:
픽셀단위 dense depth를 카메라마다 얻어 누적 LiDAR로 metric 스케일만 맞추고 TSDF 융합.

주의: step7b/7c의 저장 depth/*.npy 는 반전 버그가 있어 사용하지 않고,
누적 points3D.ply 를 카메라로 직접 투영해 스케일 기준을 만든다(프레임별 robust median ratio).

실행: kctc_gs env (transformers, open3d). 모델 가중치는 최초 1회 다운로드.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from PIL import Image

from src.common import Calibration, load_config


def _cloud_depth(P, w2c, K, W, H):
    """누적 cloud → 카메라 depth (근사 z-buffer)."""
    cam = (w2c @ np.vstack((P.T, np.ones(len(P)))))[:3].T
    z = cam[:, 2]; fr = z > 0.1
    uv = (K @ (cam[fr] / z[fr, None]).T).T
    u = np.round(uv[:, 0]).astype(int); v = np.round(uv[:, 1]).astype(int); zz = z[fr]
    ins = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, zz = u[ins], v[ins], zz[ins]
    o = np.argsort(-zz)                       # 먼 것 먼저 → 가까운 것이 덮어씀
    dep = np.zeros((H, W), np.float32); dep[v[o], u[o]] = zz[o]
    return dep


def run(ds_dir, out_path, cfg, model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
        voxel=0.05, depth_trunc=40.0, near=1.0):
    import torch
    from transformers import pipeline
    calib = Calibration.from_config(cfg)
    mask_p = calib.mask_params(); cutoff = calib.cutoff_y()
    ds = Path(ds_dir)
    cams = json.load(open(ds / "cameras.json"))["frames"]
    pcd = o3d.io.read_point_cloud(str(ds / "points3D.ply")); P = np.asarray(pcd.points)
    pipe = pipeline("depth-estimation", model=model, device=0)
    print(f"[14] {len(cams)} frames, mono={model.split('/')[-1]}", flush=True)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=voxel * 4,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    for i, f in enumerate(cams):
        w2c = np.array(f["w2c"]); K = np.array(f["K"]); W, H = f["width"], f["height"]
        bgr = cv2.imread(str(ds / f["img"]))
        m = pipe(Image.open(ds / f["img"]).convert("RGB"))["predicted_depth"]
        m = m.squeeze().float().cpu().numpy()
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H))

        # 프레임별 robust 스케일 (누적 cloud 투영 기준)
        cd = _cloud_depth(P, w2c, K, W, H)
        lvalid = (cd > near) & (cd < depth_trunc)
        vmask = lvalid & (m > 0.1)
        if vmask.sum() > 500:
            r = cd[vmask] / m[vmask]
            s = float(np.median(r[(r > np.percentile(r, 10)) & (r < np.percentile(r, 90))]))
        else:
            s = 1.0
        depth = (m * s).astype(np.float32)   # 순수 mono (스케일만 LiDAR로 정렬)

        # 마스크: 차량 하단/비네팅/원거리(하늘) 제외
        keep = (depth > near) & (depth < depth_trunc)
        yy, xx = np.mgrid[0:H, 0:W]
        if cutoff is not None:
            keep &= yy < cutoff
        if mask_p is not None:
            cx, cy, rr = mask_p; keep &= ((xx - cx) ** 2 + (yy - cy) ** 2) < rr ** 2
        depth[~keep] = 0.0

        color = o3d.geometry.Image(np.ascontiguousarray(bgr[:, :, ::-1]))   # RGB
        do = o3d.geometry.Image(depth)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, do, depth_scale=1.0, depth_trunc=depth_trunc, convert_rgb_to_intensity=False)
        intr = o3d.camera.PinholeCameraIntrinsic(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        volume.integrate(rgbd, intr, w2c)
        if i % 10 == 0:
            print(f"  fuse {i}/{len(cams)} scale={s:.2f}", flush=True)

    mesh = volume.extract_triangle_mesh(); mesh.compute_vertex_normals()
    print(f"[14] TSDF mesh: {len(mesh.vertices):,} v / {len(mesh.triangles):,} f", flush=True)

    # 벡터화 후처리 (작은조각 제거)
    tc, nt, _ = mesh.cluster_connected_triangles()
    tc = np.asarray(tc); nt = np.asarray(nt)
    mesh.remove_triangles_by_mask(nt[tc] < max(200, int(nt.max() * 0.02)))
    mesh.remove_unreferenced_vertices(); mesh.compute_vertex_normals()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(out_path, mesh, write_vertex_colors=True)
    print(f"[14] 저장: {out_path} ({len(mesh.vertices):,} v / {len(mesh.triangles):,} f)", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Step14: mono depth + LiDAR scale + TSDF fusion")
    ap.add_argument("--config", default="default.yaml")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", "-o", required=True)
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--depth-trunc", type=float, default=40.0)
    args = ap.parse_args()
    run(args.dataset, args.output, load_config(args.config), voxel=args.voxel, depth_trunc=args.depth_trunc)


if __name__ == "__main__":
    main()
