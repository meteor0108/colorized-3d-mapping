"""Step9b: native LiDAR → (①ICP 포즈정밀화) dense sharp 누적 → Poisson → (②이미지 텍스처) mesh.

밀도 진단 결론: 프레임 수↑로는 per-surface 밀도 안 늘음. 한계 = raw odom 포즈 smear.
→ ① scan-to-map ICP(point-to-plane)로 포즈 정밀화 → 누적 smear 제거 → 기하 sharpen
   ② blackfly 이미지를 mesh vertex에 투영·베이킹 → 체감 자연스러움

작은 영역 + fine voxel 전제 (큰 윈도우는 area만 커져 coarse됨).
실행: kctc_gs env
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial.transform import Rotation as R

from src.common import Calibration, FileManager, PointCloudProcessor, TimestampParser, load_config

SENSORS = ["ouster1", "ouster2", "ouster3"]


def _load_nav(p):
    df = pd.read_csv(p); df.columns = df.columns.str.strip()
    tcol = [c for c in df.columns if "time" in c.lower() or "stamp" in c.lower()][0]
    return df, df[tcol].values


def _odom_pose(df, navts, t, origin):
    i = int(np.abs(navts - t).argmin())
    p = np.array([df.position_x.iloc[i], df.position_y.iloc[i], df.position_z.iloc[i]])
    q = np.array([df.orientation_x.iloc[i], df.orientation_y.iloc[i],
                  df.orientation_z.iloc[i], df.orientation_w.iloc[i]])
    T = np.eye(4); T[:3, :3] = R.from_quat(q).as_matrix(); T[:3, 3] = p - origin
    return T


def run(data_dir, out_path, cfg, num_frames=50, range_m=40.0, voxel=0.02,
        poisson_depth=12, icp=True):
    calib = Calibration.from_config(cfg)
    K = np.asarray(calib.intrinsic); dist = np.asarray(calib.distortion)
    data = Path(data_dir)
    df, navts = _load_nav(data / "navigation.csv")
    g2l = calib.gps_to_lidar          # lidar_s -> gps
    l2c = calib.lidar_to_camera       # lidar_s -> cam
    mask = calib.mask_params(); cutoff = calib.cutoff_y()

    files = {s: FileManager.get_files_and_times(str(data / s / "points")) for s in SENSORS}
    img_files, img_times = FileManager.get_files_and_times(str(data / "blackfly"), ext=".png")
    f2, t2 = files["ouster2"]
    sel = list(range(min(num_frames, len(f2))))
    i0 = int(np.abs(navts - t2[sel[0]]).argmin())
    origin = np.array([df.position_x.iloc[i0], df.position_y.iloc[i0], df.position_z.iloc[i0]])

    def combined_gps(idx):
        """frame idx의 3-LiDAR를 gps(vehicle) frame 통합 cloud로."""
        pts = []
        for s in SENSORS:
            fs, ts = files[s]
            # ouster2 시간과 가장 가까운 같은-프레임 파일
            j = int(np.abs(ts - t2[idx]).argmin())
            pc = o3d.io.read_point_cloud(str(data / s / "points" / fs[j]))
            pc = PointCloudProcessor.range_filter(pc, distance_threshold=range_m)
            P = np.asarray(pc.points)
            P = (g2l[s] @ np.vstack((P.T, np.ones(len(P)))))[:3].T
            pts.append(P)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(np.vstack(pts))
        return pc

    # ---------- ① ICP 포즈 정밀화 + 누적 ----------
    t0 = time.time()
    refined = {}           # idx -> T_world_gps (정밀)
    map_pcd = o3d.geometry.PointCloud()
    geo = []
    for k, idx in enumerate(sel):
        src = combined_gps(idx)
        init = _odom_pose(df, navts, t2[idx], origin)
        if icp and len(map_pcd.points) > 5000:
            tgt = map_pcd.voxel_down_sample(voxel * 2)
            tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 6, max_nn=30))
            src_d = src.voxel_down_sample(voxel * 2)
            reg = o3d.pipelines.registration.registration_icp(
                src_d, tgt, 0.5, init,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
            T = reg.transformation
        else:
            T = init
        refined[idx] = T
        w = src.transform(T)
        geo.append(np.asarray(w.points))
        map_pcd += w
        if k % 5 == 0:
            map_pcd = map_pcd.voxel_down_sample(voxel * 2)
        if k % 10 == 0:
            print(f"  [①ICP] {k}/{len(sel)}  map={len(map_pcd.points):,}  {time.time()-t0:.0f}s", flush=True)

    dense = o3d.geometry.PointCloud()
    dense.points = o3d.utility.Vector3dVector(np.vstack(geo))
    dense = dense.voxel_down_sample(voxel)
    dense, _ = dense.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    nn = np.asarray(dense.compute_nearest_neighbor_distance())
    print(f"[step9b] sharp dense cloud: {len(dense.points):,} pts | NN median={np.median(nn)*100:.2f}cm "
          f"| {time.time()-t0:.0f}s", flush=True)

    # ---------- ② 이미지 색 투영 (정밀 포즈로) ----------
    color_pts, color_rgb = [], []
    img_cache = {}
    for idx in sel:
        T = refined[idx]
        jimg = int(np.abs(img_times - t2[idx]).argmin())
        if abs(img_times[jimg] - t2[idx]) > 0.15:
            continue
        ip = str(data / "blackfly" / img_files[jimg])
        img = img_cache.get(ip) or cv2.imread(ip)
        img_cache[ip] = img
        if calib.vignetting.get("enabled", False):
            img = PointCloudProcessor.apply_vignetting_correction(img, calib.vignetting.get("strength", 0.4))
        for s in SENSORS:
            fs, ts = files[s]
            j = int(np.abs(ts - t2[idx]).argmin())
            pc = o3d.io.read_point_cloud(str(data / s / "points" / fs[j]))
            pc = PointCloudProcessor.range_filter(pc, distance_threshold=range_m)
            _, col = PointCloudProcessor.project_to_image(
                pc, img, l2c[s], K, dist, calib.fisheye,
                max_range=range_m, mask_params=mask, cutoff_y=cutoff, draw_overlay=False)
            if col.is_empty():
                continue
            P = np.asarray(col.points)
            world_T_lidar = T @ g2l[s]
            P = (world_T_lidar @ np.vstack((P.T, np.ones(len(P)))))[:3].T
            color_pts.append(P); color_rgb.append(np.asarray(col.colors))
    cloud_col = o3d.geometry.PointCloud()
    if color_pts:
        cloud_col.points = o3d.utility.Vector3dVector(np.vstack(color_pts))
        cloud_col.colors = o3d.utility.Vector3dVector(np.vstack(color_rgb))
        cloud_col = cloud_col.voxel_down_sample(voxel)
    print(f"[step9b] colored pts: {len(cloud_col.points):,}  {time.time()-t0:.0f}s", flush=True)

    # ---------- Poisson (정밀 cloud) ----------
    dense.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 6, max_nn=30))
    cams = np.array([refined[i][:3, 3] for i in sel])
    from scipy.spatial import cKDTree
    P = np.asarray(dense.points); N = np.asarray(dense.normals)
    _, ci = cKDTree(cams).query(P, workers=-1)
    N[np.einsum("ij,ij->i", N, cams[ci] - P) < 0] *= -1
    dense.normals = o3d.utility.Vector3dVector(N)

    mesh, densi = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        dense, depth=poisson_depth, scale=1.1, linear_fit=False)
    densi = np.asarray(densi)
    mesh.remove_vertices_by_mask(densi < np.quantile(densi, 0.04))
    mesh.remove_unreferenced_vertices()

    # ---------- ② vertex 텍스처 베이킹 (KNN from 컬러 cloud) ----------
    if len(cloud_col.points) > 0:
        ctree = cKDTree(np.asarray(cloud_col.points))
        ccol = np.asarray(cloud_col.colors)
        V = np.asarray(mesh.vertices)
        d, vi = ctree.query(V, workers=-1)
        vcol = ccol[vi]
        vcol[d > voxel * 4] = 0.5            # 카메라 미관측 → 회색
        mesh.vertex_colors = o3d.utility.Vector3dVector(vcol)
    mesh.compute_vertex_normals()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(out_path, mesh, write_vertex_colors=True)
    samp = mesh.sample_points_uniformly(number_of_points=200000)
    dd = np.asarray(samp.compute_point_cloud_distance(dense))
    print(f"[step9b] mesh: {len(mesh.vertices):,} v / {len(mesh.triangles):,} f | "
          f"mesh→LiDAR median={np.median(dd):.3f}m | {time.time()-t0:.0f}s", flush=True)
    print(f"[step9b] 저장: {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Step9b: ICP-refined dense + textured LiDAR mesh")
    ap.add_argument("--config", default="default.yaml")
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", "-o", required=True)
    ap.add_argument("--num-frames", type=int, default=50)
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--poisson-depth", type=int, default=12)
    ap.add_argument("--no-icp", action="store_true", help="ICP 끄고 raw odom (비교용)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run(args.data, args.output, cfg, num_frames=args.num_frames, voxel=args.voxel,
        poisson_depth=args.poisson_depth, icp=not args.no_icp)


if __name__ == "__main__":
    main()
