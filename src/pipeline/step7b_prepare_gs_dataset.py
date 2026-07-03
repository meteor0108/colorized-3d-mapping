"""Step7b: submap 폴더 → Gaussian Splatting 학습용 dataset export.

기존 colorize 파이프라인(step2)의 변환 체인을 그대로 재사용하여 GS 입력을 만든다.
COLMAP SfM 없이 GNSS+VGICP+PGO 포즈로 카메라 포즈를 직접 도출하는 것이 핵심.

출력 (configs/mesh.yaml > mesh.gaussian.dataset):
    {output_dir}/{submap}/
        images/<frame>.png        undistort(+vignetting) 적용된 pinhole 이미지
        cameras.json              프레임별 w2c(OpenCV) + K + size
        transforms.json           nerfstudio 형식 c2w(OpenGL) — off-the-shelf 도구용
        points3D.ply              init Gaussian (submap 컬러 PCD, world frame)
        depth/<frame>.npy         LiDAR sparse metric depth (pinhole)
        overlay/<frame>.jpg       재투영 검증 오버레이 (M1 핵심 체크)

카메라 월드 포즈:
    world_T_cam = trajectory @ gps_to_lidar[pose_lidar] @ inv(lidar_to_camera[pose_lidar])
    (trajectory = 이미지 timestamp 기준 GNSS world_T_gps)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial.transform import Rotation as R

from src.common import Calibration, FileManager, PointCloudProcessor, TimestampParser, load_config


def _load_navigation(nav_path: str):
    df = pd.read_csv(nav_path)
    df.columns = df.columns.str.strip()
    time_cols = [c for c in df.columns if "time" in c.lower() or "stamp" in c.lower()]
    if not time_cols:
        raise ValueError(f"navigation.csv에 timestamp 컬럼이 없음: {nav_path}")
    return df, df[time_cols[0]].values


def _pose_at(df: pd.DataFrame, timestamps: np.ndarray, query_time: float):
    idx = int(np.abs(timestamps - query_time).argmin())
    pos = np.array([df["position_x"].iloc[idx], df["position_y"].iloc[idx], df["position_z"].iloc[idx]])
    quat = np.array([
        df["orientation_x"].iloc[idx], df["orientation_y"].iloc[idx],
        df["orientation_z"].iloc[idx], df["orientation_w"].iloc[idx],
    ])
    return pos, quat


def _trajectory(pos: np.ndarray, quat: np.ndarray, origin: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quat).as_matrix()
    T[:3, 3] = pos - origin
    return T


def _project_pinhole(cam_pts: np.ndarray, K: np.ndarray, width: int, height: int):
    """카메라 좌표계 점 → pinhole 픽셀 + depth. 이미지 안 + 전방만 반환."""
    z = cam_pts[:, 2]
    front = z > 1e-3
    cam_pts, z = cam_pts[front], z[front]
    uv = (K @ (cam_pts / z[:, None]).T).T  # Nx3
    u, v = uv[:, 0], uv[:, 1]
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u[inside].astype(int), v[inside].astype(int), z[inside]


class GSDatasetExporter:
    def __init__(self, cfg, folder: str):
        self.cfg = cfg
        self.calib = Calibration.from_config(cfg)
        self.ds = cfg.mesh.gaussian.dataset
        self.folder = folder
        self.name = Path(folder.rstrip("/")).name

        self.K = np.asarray(self.calib.intrinsic, dtype=np.float64)
        self.dist = np.asarray(self.calib.distortion, dtype=np.float64)
        self.fisheye = self.calib.fisheye
        self.pose_lidar = str(self.ds.pose_lidar)

        out_root = Path(cfg.mesh.gaussian.dataset.output_dir) / self.name
        self.out = out_root
        for sub in ("images", "depth", "overlay"):
            (out_root / sub).mkdir(parents=True, exist_ok=True)

        self.image_dir = os.path.join(folder, "blackfly/")
        self.nav_path = os.path.join(folder, "navigation.csv")
        self.lidar_dirs = {s: os.path.join(folder, f"{s}/points/") for s in self.ds.lidars}

        self.origin = None
        self.cameras = []          # cameras.json 항목
        self.frames_gl = []        # transforms.json 항목
        self.init_pts, self.init_col = [], []

    def run(self):
        df, nav_ts = _load_navigation(self.nav_path)
        lidar_files = {s: FileManager.get_files_and_times(d) for s, d in self.lidar_dirs.items()}

        images = sorted(f for f in os.listdir(self.image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
        max_frames = self.ds.get("max_frames")
        if max_frames:
            images = images[: int(max_frames)]
        print(f"[step7b] {self.name}: {len(images)} frames")

        mcx_mcy_mr = self.calib.mask_params()
        cutoff_y = self.calib.cutoff_y()

        for i, img_name in enumerate(images):
            t = TimestampParser.parse(img_name)
            if t is None:
                continue
            img = cv2.imread(os.path.join(self.image_dir, img_name))
            if img is None:
                continue
            if self.ds.apply_vignetting and self.calib.vignetting.get("enabled", False):
                img = PointCloudProcessor.apply_vignetting_correction(
                    img, strength=self.calib.vignetting.get("strength", 0.4))

            pos, quat = _pose_at(df, nav_ts, t)
            if self.origin is None:
                self.origin = pos
            traj = _trajectory(pos, quat, self.origin)  # world_T_gps

            cam_T_l_pose = np.asarray(self.calib.lidar_to_camera[self.pose_lidar])
            world_T_lidar_pose = traj @ self.calib.gps_to_lidar[self.pose_lidar]
            world_T_cam = world_T_lidar_pose @ np.linalg.inv(cam_T_l_pose)
            w2c = np.linalg.inv(world_T_cam)
            h, w = img.shape[:2]

            stem = Path(img_name).stem
            depth_buf = np.zeros((h, w), np.float32)

            for s, ldir in self.lidar_dirs.items():
                files, times = lidar_files[s]
                fname = FileManager.match_by_time(t, files, times, tolerance=float(self.ds.time_tolerance))
                if fname is None:
                    continue
                pcd = o3d.io.read_point_cloud(os.path.join(ldir, fname))
                if pcd.is_empty():
                    continue
                pcd = PointCloudProcessor.range_filter(pcd, distance_threshold=float(self.ds.range_filter_m))

                cam_T_l = np.asarray(self.calib.lidar_to_camera[s])
                # 원본(왜곡) 이미지에서 색 추출 → lidar frame 컬러 포인트
                _, colored = PointCloudProcessor.project_to_image(
                    pcd, img, cam_T_l, self.K, self.dist, self.fisheye,
                    max_range=float(self.ds.range_filter_m),
                    mask_params=mcx_mcy_mr, cutoff_y=cutoff_y, draw_overlay=False)
                if colored.is_empty():
                    continue
                lp = np.asarray(colored.points)
                lc = np.asarray(colored.colors)

                # init 누적 (world frame): world_T_lidar_s = traj @ gps_to_lidar[s]
                world_T_lidar_s = traj @ self.calib.gps_to_lidar[s]
                world_pts = (world_T_lidar_s @ np.vstack((lp.T, np.ones(len(lp)))))[:3].T
                self.init_pts.append(world_pts)
                self.init_col.append(lc)

                # depth/overlay (pinhole, undistort 이미지 기준)
                cam_pts = (cam_T_l @ np.vstack((lp.T, np.ones(len(lp)))))[:3].T
                u, v, z = _project_pinhole(cam_pts, self.K, w, h)
                depth_buf[v, u] = np.where(depth_buf[v, u] == 0, z, np.minimum(depth_buf[v, u], z))

            # 이미지 undistort + 저장
            out_img = img
            if self.ds.undistort and not self.fisheye:
                out_img = cv2.undistort(img, self.K, self.dist)
            cv2.imwrite(str(self.out / "images" / f"{stem}.png"), out_img)

            if self.ds.depth_export:
                np.save(str(self.out / "depth" / f"{stem}.npy"), depth_buf)
            if self.ds.overlay_export:
                self._save_overlay(out_img, depth_buf, stem)

            self._record_pose(stem, world_T_cam, w2c, w, h)
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(images)}")

        self._write_outputs()

    def _save_overlay(self, base_img, depth_buf, stem):
        vis = base_img.copy()
        ys, xs = np.where(depth_buf > 0)
        if len(xs):
            d = depth_buf[ys, xs]
            cidx = np.clip((255 * d / float(self.ds.range_filter_m)).astype(int), 0, 255)
            cmap = (np.array([cv2.applyColorMap(np.array([[i]], np.uint8), cv2.COLORMAP_JET)[0, 0]
                              for i in range(256)]))
            for x, y, c in zip(xs, ys, cmap[cidx]):
                cv2.circle(vis, (int(x), int(y)), 2, tuple(int(v) for v in c), -1)
        cv2.imwrite(str(self.out / "overlay" / f"{stem}.jpg"), vis)

    def _record_pose(self, stem, c2w, w2c, w, h):
        self.cameras.append({
            "img": f"images/{stem}.png",
            "w2c": w2c.tolist(),
            "K": self.K.tolist(),
            "width": w, "height": h,
        })
        # OpenCV(c2w) → OpenGL(c2w): y,z 축 반전
        c2w_gl = c2w @ np.diag([1.0, -1.0, -1.0, 1.0])
        self.frames_gl.append({
            "file_path": f"images/{stem}.png",
            "transform_matrix": c2w_gl.tolist(),
        })

    def _write_outputs(self):
        if not self.init_pts:
            print("[step7b] 경고: init 포인트 없음 — 경로/캘리브 확인")
            return
        pts = np.vstack(self.init_pts)
        col = np.vstack(self.init_col)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(np.clip(col, 0, 1))
        pcd = pcd.voxel_down_sample(0.05)
        o3d.io.write_point_cloud(str(self.out / "points3D.ply"), pcd)

        with open(self.out / "cameras.json", "w") as f:
            json.dump({"frames": self.cameras}, f, indent=2)

        fx, fy = float(self.K[0, 0]), float(self.K[1, 1])
        cx, cy = float(self.K[0, 2]), float(self.K[1, 2])
        w = self.cameras[0]["width"]
        h = self.cameras[0]["height"]
        transforms = {
            "camera_model": "OPENCV",
            "fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy, "w": w, "h": h,
            # undistort된 이미지를 쓰므로 distortion=0
            "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
            "frames": self.frames_gl,
        }
        with open(self.out / "transforms.json", "w") as f:
            json.dump(transforms, f, indent=2)

        print(f"[step7b] 완료 → {self.out}")
        print(f"  init points: {len(pcd.points):,} | frames: {len(self.cameras)}")
        print(f"  ✅ 다음: overlay/*.jpg 에서 LiDAR 점이 이미지 구조와 정렬되는지 확인 (포즈 체인 검증)")


def main():
    parser = argparse.ArgumentParser(description="Step7b: submap → Gaussian Splatting dataset")
    parser.add_argument("--config", default="default.yaml")
    parser.add_argument("--folder", default=None, help="submap 폴더 (없으면 config 사용)")
    parser.add_argument("--max-frames", type=int, default=None, help="처리 프레임 수 제한 (디버그/검증)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_frames is not None:
        cfg.mesh.gaussian.dataset["max_frames"] = args.max_frames
    folder = args.folder or cfg.mesh.gaussian.dataset.input_folder or cfg.projection.input_folder
    if not folder or not os.path.isdir(folder):
        raise SystemExit(f"입력 폴더 없음: {folder}")
    GSDatasetExporter(cfg, folder).run()


if __name__ == "__main__":
    main()
