"""Step9: native-rate LiDAR 전량 누적 → dense Poisson mesh (LiDAR-primary, A-track).

step7b는 카메라프레임당 LiDAR 1 scan만 매칭 → 밀도 이득 없음.
이 스크립트는 **LiDAR-frame 구동**: 모든 native scan을 per-scan nav 포즈로 world 변환·누적
→ 밀도 ~10배 → consistent-normal screened Poisson.

입력: step_pre1_extract_native.py 출력 폴더 (ouster{1,2,3}/points + navigation.csv)
실행: kctc_gs env
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial.transform import Rotation as R

from src.common import Calibration, FileManager, PointCloudProcessor, TimestampParser, load_config


def _load_nav(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    tcol = [c for c in df.columns if "time" in c.lower() or "stamp" in c.lower()][0]
    return df, df[tcol].values


def _traj_at(df, ts_arr, t, origin):
    i = int(np.abs(ts_arr - t).argmin())
    pos = np.array([df["position_x"].iloc[i], df["position_y"].iloc[i], df["position_z"].iloc[i]])
    quat = np.array([df["orientation_x"].iloc[i], df["orientation_y"].iloc[i],
                     df["orientation_z"].iloc[i], df["orientation_w"].iloc[i]])
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quat).as_matrix()
    T[:3, 3] = pos - origin
    return T


def run(data_dir, out_path, cfg, num_frames=150, stride=1, range_m=50.0,
        voxel=0.025, poisson_depth=12, trim_q=0.04):
    calib = Calibration.from_config(cfg)
    data = Path(data_dir)
    df, navts = _load_nav(data / "navigation.csv")

    sensors = ["ouster1", "ouster2", "ouster3"]
    files = {s: FileManager.get_files_and_times(str(data / s / "points")) for s in sensors}

    # 윈도우: ouster2 기준 첫 num_frames (stride)
    f2, t2 = files["ouster2"]
    sel_idx = list(range(0, min(num_frames * stride, len(f2)), stride))
    t_lo, t_hi = t2[sel_idx[0]], t2[sel_idx[-1]]
    # origin = 윈도우 첫 포즈
    i0 = int(np.abs(navts - t_lo).argmin())
    origin = np.array([df["position_x"].iloc[i0], df["position_y"].iloc[i0], df["position_z"].iloc[i0]])
    print(f"[step9] window: {t_hi - t_lo:.1f}s  frames/sensor≈{len(sel_idx)}", flush=True)

    acc, centers = [], []
    t0 = time.time()
    for s in sensors:
        fs, ts = files[s]
        sel = np.where((ts >= t_lo) & (ts <= t_hi))[0][::stride]
        gps2lidar = calib.gps_to_lidar[s]
        for k, j in enumerate(sel):
            pcd = o3d.io.read_point_cloud(str(data / s / "points" / fs[j]))
            if pcd.is_empty():
                continue
            pcd = PointCloudProcessor.range_filter(pcd, distance_threshold=range_m)
            traj = _traj_at(df, navts, ts[j], origin)            # world_T_gps
            world_T_lidar = traj @ gps2lidar
            pcd.transform(world_T_lidar)
            acc.append(np.asarray(pcd.points))
            centers.append(traj[:3, 3])
            if len(acc) % 100 == 0:
                # 주기적 병합+다운샘플 (메모리 관리)
                P = np.vstack(acc)
                tmp = o3d.geometry.PointCloud()
                tmp.points = o3d.utility.Vector3dVector(P)
                tmp = tmp.voxel_down_sample(voxel)
                acc = [np.asarray(tmp.points)]
                print(f"  누적 {s} {k}/{len(sel)}  pts={len(acc[0]):,}  {time.time()-t0:.0f}s", flush=True)

    P = np.vstack(acc)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P)
    pcd = pcd.voxel_down_sample(voxel)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"[step9] dense cloud: {len(pcd.points):,} pts (voxel {voxel}m)  {time.time()-t0:.0f}s", flush=True)

    # 밀도 지표 (NN 간격)
    nn = np.asarray(pcd.compute_nearest_neighbor_distance())
    print(f"[step9] NN 간격 median: {np.median(nn)*100:.2f} cm  (dense {len(pcd.points):,} pts)", flush=True)

    # Poisson RAM 안전: 상한 초과 시 voxel 키워 다운샘플
    max_points = 10_000_000
    v = voxel
    while len(pcd.points) > max_points:
        v *= 1.3
        pcd = pcd.voxel_down_sample(v)
    if v != voxel:
        print(f"[step9] Poisson용 다운샘플 voxel {v:.3f}m → {len(pcd.points):,} pts", flush=True)

    # normal 추정 + 센서 위치쪽 정렬 (벡터화: scipy cKDTree)
    from scipy.spatial import cKDTree
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=v * 6, max_nn=30))
    C = np.unique(np.array(centers), axis=0)
    pts = np.asarray(pcd.points); N = np.asarray(pcd.normals)
    _, ci = cKDTree(C).query(pts, workers=-1)
    view = C[ci] - pts
    N[np.einsum("ij,ij->i", N, view) < 0] *= -1
    pcd.normals = o3d.utility.Vector3dVector(N)

    # screened Poisson + trim
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth, scale=1.1, linear_fit=False)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, trim_q))
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(out_path, mesh)
    print(f"[step9] Poisson mesh: {len(mesh.vertices):,} v / {len(mesh.triangles):,} f  {time.time()-t0:.0f}s", flush=True)

    samp = mesh.sample_points_uniformly(number_of_points=200000)
    d = np.asarray(samp.compute_point_cloud_distance(pcd))
    print(f"[step9] mesh→LiDAR (m): median={np.median(d):.3f} mean={d.mean():.3f} p90={np.percentile(d,90):.3f}", flush=True)
    print(f"[step9] 저장: {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Step9: dense native-LiDAR Poisson mesh")
    ap.add_argument("--config", default="default.yaml")
    ap.add_argument("--data", required=True, help="native 추출 폴더")
    ap.add_argument("--output", "-o", required=True)
    ap.add_argument("--num-frames", type=int, default=150)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--voxel", type=float, default=0.025)
    ap.add_argument("--poisson-depth", type=int, default=12)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run(args.data, args.output, cfg, num_frames=args.num_frames, stride=args.stride,
        voxel=args.voxel, poisson_depth=args.poisson_depth)


if __name__ == "__main__":
    main()
