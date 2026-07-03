"""Step7c: native dense(10Hz) 카메라 + ICP 정밀포즈 + LiDAR depth → GS dataset.

step7b는 1Hz·raw odom. 이건 GS의 근본 약점(sparse view, 포즈오차)을 공략:
  - 카메라 10Hz (0.8m 간격) → 멀티뷰 overlap 大 → floater 감소
  - scan-to-map ICP(sliding-window) 로 포즈 정밀화 → ghosting 감소
  - LiDAR depth 감독은 그대로

출력은 step7b와 동일 포맷(images/cameras.json/depth/points3D.ply) → step8 그대로 사용.
실행: kctc_gs env
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial.transform import Rotation as R

from src.common import Calibration, FileManager, PointCloudProcessor, load_config

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


def run(data_dir, out_dir, cfg, num_frames=60, range_m=40.0, icp=True,
        pose_lidar="ouster2", voxel=0.05, start_frame=0):
    import time
    calib = Calibration.from_config(cfg)
    K = np.asarray(calib.intrinsic); dist = np.asarray(calib.distortion)
    mask = calib.mask_params(); cutoff = calib.cutoff_y()
    data = Path(data_dir)
    df, navts = _load_nav(data / "navigation.csv")
    g2l, l2c = calib.gps_to_lidar, calib.lidar_to_camera

    lidar = {s: FileManager.get_files_and_times(str(data / s / "points")) for s in SENSORS}
    imgs, imts = FileManager.get_files_and_times(str(data / "blackfly"), ext=".png")
    sel = list(range(start_frame, min(start_frame + num_frames, len(imgs))))
    i0 = int(np.abs(navts - imts[sel[0]]).argmin())
    origin = np.array([df.position_x.iloc[i0], df.position_y.iloc[i0], df.position_z.iloc[i0]])

    out = Path(out_dir)
    for sub in ("images", "depth", "overlay"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    def combined_gps(t):
        P = []
        for s in SENSORS:
            fs, ts = lidar[s]
            j = int(np.abs(ts - t).argmin())
            pc = PointCloudProcessor.range_filter(
                o3d.io.read_point_cloud(str(data / s / "points" / fs[j])), distance_threshold=range_m)
            Q = np.asarray(pc.points)
            P.append((g2l[s] @ np.vstack((Q.T, np.ones(len(Q)))))[:3].T)
        pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(np.vstack(P)); return pc

    # ---------- ① ICP 포즈 정밀화 (sliding-window local map) ----------
    t0 = time.time()
    refined = {}
    local = deque(maxlen=8)
    for k in sel:
        t = imts[k]
        src = combined_gps(t)
        init = _odom_pose(df, navts, t, origin)
        if icp and local:
            tgt = o3d.geometry.PointCloud()
            for w in local:
                tgt += w
            tgt = tgt.voxel_down_sample(0.1)
            tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
            reg = o3d.pipelines.registration.registration_icp(
                src.voxel_down_sample(0.1), tgt, 0.6, init,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
            T = reg.transformation
        else:
            T = init
        refined[k] = T
        local.append(o3d.geometry.PointCloud(src).transform(T).voxel_down_sample(0.1))
        if k % 10 == 0:
            print(f"  [①ICP] {k}/{len(sel)}  {time.time()-t0:.0f}s", flush=True)

    # ---------- ② dense 카메라 export + depth + init ----------
    cams, frames_gl, init_pts, init_col = [], [], [], []
    for k in sel:
        t = imts[k]
        T = refined[k]                                   # world_T_gps (정밀)
        img = cv2.imread(str(data / "blackfly" / imgs[k]))
        if img is None:
            continue
        if calib.vignetting.get("enabled", False):
            img = PointCloudProcessor.apply_vignetting_correction(img, calib.vignetting.get("strength", 0.4))
        h, w = img.shape[:2]
        cam_T_lp = np.asarray(l2c[pose_lidar])
        world_T_cam = (T @ g2l[pose_lidar]) @ np.linalg.inv(cam_T_lp)
        w2c = np.linalg.inv(world_T_cam)
        stem = f"{t:.6f}"
        depth_buf = np.zeros((h, w), np.float32)

        for s in SENSORS:
            fs, ts = lidar[s]
            j = int(np.abs(ts - t).argmin())
            pc = PointCloudProcessor.range_filter(
                o3d.io.read_point_cloud(str(data / s / "points" / fs[j])), distance_threshold=range_m)
            _, col = PointCloudProcessor.project_to_image(
                pc, img, l2c[s], K, dist, calib.fisheye, max_range=range_m,
                mask_params=mask, cutoff_y=cutoff, draw_overlay=False)
            if col.is_empty():
                continue
            lp = np.asarray(col.points); lc = np.asarray(col.colors)
            init_pts.append(((T @ g2l[s]) @ np.vstack((lp.T, np.ones(len(lp)))))[:3].T)
            init_col.append(lc)
            cam_pts = (np.asarray(l2c[s]) @ np.vstack((lp.T, np.ones(len(lp)))))[:3].T
            z = cam_pts[:, 2]; fr = z > 1e-3
            uv = (K @ (cam_pts[fr] / z[fr, None]).T).T
            u, v = uv[:, 0].astype(int), uv[:, 1].astype(int)
            ins = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            depth_buf[v[ins], u[ins]] = np.where(depth_buf[v[ins], u[ins]] == 0, z[fr][ins],
                                                 np.minimum(depth_buf[v[ins], u[ins]], z[fr][ins]))

        out_img = cv2.undistort(img, K, dist) if not calib.fisheye else img
        cv2.imwrite(str(out / "images" / f"{stem}.png"), out_img)
        np.save(str(out / "depth" / f"{stem}.npy"), depth_buf)
        cams.append({"img": f"images/{stem}.png", "w2c": w2c.tolist(),
                     "K": K.tolist(), "width": w, "height": h})
        c2w_gl = world_T_cam @ np.diag([1.0, -1.0, -1.0, 1.0])
        frames_gl.append({"file_path": f"images/{stem}.png", "transform_matrix": c2w_gl.tolist()})
        if k % 10 == 0:
            print(f"  [②export] {k}/{len(sel)}  {time.time()-t0:.0f}s", flush=True)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(init_pts))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(np.vstack(init_col), 0, 1))
    pcd = pcd.voxel_down_sample(voxel)
    o3d.io.write_point_cloud(str(out / "points3D.ply"), pcd)
    json.dump({"frames": cams}, open(out / "cameras.json", "w"), indent=2)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    json.dump({"camera_model": "OPENCV", "fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy,
               "w": cams[0]["width"], "h": cams[0]["height"],
               "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "frames": frames_gl},
              open(out / "transforms.json", "w"), indent=2)
    print(f"[step7c] 완료 → {out}  frames={len(cams)} init={len(pcd.points):,}  {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Step7c: dense(10Hz)+ICP GS dataset")
    ap.add_argument("--config", default="default.yaml")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-frames", type=int, default=60)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--range-m", type=float, default=40.0)
    ap.add_argument("--no-icp", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run(args.data, args.out, cfg, num_frames=args.num_frames, range_m=args.range_m,
        icp=not args.no_icp, start_frame=args.start_frame)


if __name__ == "__main__":
    main()
