"""4단계: 자동 배치 LiDAR-Camera colorize (단순 GNSS 변환, ouster2 only).

원본: 4_pointcloud_colorlize_show_auto.py (Dec 23 - timestamp 매칭 + submap chunking 도입한 최종버전).
구버전(`4_pointcloud_colorlize_show_auto_original.py`)는 _legacy/ 로 이동.
"""
from __future__ import annotations

import argparse
import copy
import gc
import glob
import os

import cv2
import natsort
import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial.transform import Rotation as R

from src.common import (
    Calibration,
    ConfigDict,
    FileManager,
    PointCloudProcessor,
    TimestampParser,
    load_config,
)


def run_step4(cfg: ConfigDict) -> None:
    calib = Calibration.from_config(cfg)
    proc = cfg.colorize.processing
    s4 = cfg.colorize.step4_auto
    gpu_cfg = cfg.colorize.gpu

    is_cuda = gpu_cfg.enabled and o3d.core.cuda.is_available()
    device = o3d.core.Device(gpu_cfg.device if is_cuda else "CPU")
    print(f"System Device: {device} (High-Performance GPU Mode)\n")

    root_path = cfg.paths.data_root
    base_save_folder = cfg.outputs.vgicp_pgo_map
    os.makedirs(base_save_folder, exist_ok=True)

    depth3 = glob.glob(os.path.join(root_path, "*", "*", "20*"))
    depth2 = glob.glob(os.path.join(root_path, "*", "2024*"))
    target_folders = sorted(set(depth3 + depth2))
    target_folders = [f for f in target_folders if os.path.exists(os.path.join(f, "ouster1"))]

    for f_idx, folder in enumerate(target_folders):
        if not os.path.isdir(folder):
            continue
        folder_name = os.path.basename(folder)
        check_path = os.path.join(base_save_folder, f"{folder_name}_{s4.output_prefix}_part0.pcd")
        if cfg.colorize.submap.skip_processed and os.path.exists(check_path):
            print(f"Skipping {folder_name} (Already Exists)")
            continue

        print(f"\nProcessing Folder [{f_idx + 1}/{len(target_folders)}]: {folder_name}")

        pc_folder2 = os.path.join(folder, "ouster2/points/")
        image_folder = os.path.join(folder, "blackfly/")
        navigation_file = os.path.join(folder, "navigation.csv")
        if not os.path.exists(navigation_file):
            continue

        df_nav = pd.read_csv(navigation_file)
        df_nav.columns = df_nav.columns.str.strip()
        time_cols = [c for c in df_nav.columns if "time" in c.lower() or "stamp" in c.lower()]
        if not time_cols:
            continue
        nav_timestamps = df_nav[time_cols[0]].values

        file_list = natsort.natsorted(os.listdir(image_folder))

        submap_t = None
        submap_count = 0
        is_origin_set = False
        origin = np.zeros(3)

        for i, file_name in enumerate(file_list):
            if proc.frame_skip > 1 and i % proc.frame_skip != 0:
                continue

            if i % 10 == 0:
                p = len(submap_t.point["positions"]) if submap_t is not None else 0
                print(f"  -> Frame {i}/{len(file_list)} | Part {submap_count} | GPU Pts: {p}", end="\r")

            if submap_t is not None and i > 0 and i % s4.frame_limit == 0:
                save_path = os.path.join(
                    base_save_folder, f"{folder_name}_{s4.output_prefix}_part{submap_count}.pcd"
                )
                o3d.io.write_point_cloud(save_path, submap_t.to_legacy(), write_ascii=False)
                del submap_t
                submap_t = None
                submap_count += 1
                gc.collect()
                o3d.core.cuda.release_cache()

            query_time = TimestampParser.parse(file_name)
            if query_time is None:
                continue
            closest_idx = int(np.abs(nav_timestamps - query_time).argmin())
            try:
                pos = df_nav.loc[closest_idx, ["position_x", "position_y", "position_z"]].to_numpy()
                ori = df_nav.loc[
                    closest_idx, ["orientation_x", "orientation_y", "orientation_z", "orientation_w"]
                ].to_numpy()
            except KeyError:
                continue

            image = cv2.imread(os.path.join(image_folder, file_name))
            if image is None:
                continue

            pcd_path = os.path.join(pc_folder2, file_name.replace(".png", ".pcd"))
            if not os.path.exists(pcd_path):
                continue

            if not is_origin_set:
                origin = pos.copy()
                is_origin_set = True

            trajectory = np.eye(4)
            trajectory[:3, :3] = R.from_quat(ori).as_matrix()
            trajectory[:3, 3] = pos - origin

            pcd = o3d.io.read_point_cloud(pcd_path)
            pcd = PointCloudProcessor.range_filter(pcd, distance_threshold=proc.distance_threshold)
            _, pcd = PointCloudProcessor.project_to_image(
                pcd, image, calib.lidar_to_camera["ouster2"], calib.intrinsic,
                calib.distortion, calib.fisheye,
                max_range=proc.projection_distance,
                mask_params=None, cutoff_y=None,
                draw_overlay=False,
            )
            if len(pcd.points) == 0:
                continue

            try:
                pcd_t = o3d.t.geometry.PointCloud.from_legacy(pcd, device=device)
                transform = trajectory @ calib.gps_to_lidar["ouster2"]
                pcd_t.transform(o3d.core.Tensor(transform, dtype=o3d.core.Dtype.Float64, device=device))
                pcd_t = pcd_t.voxel_down_sample(proc.voxel_size)
                submap_t = pcd_t if submap_t is None else submap_t.append(pcd_t)
                del pcd_t, pcd
            except RuntimeError as e:
                print(f"\n[GPU Error] {file_name}: {e}")
                o3d.core.cuda.release_cache()
                continue

        if submap_t is not None:
            save_path = os.path.join(
                base_save_folder, f"{folder_name}_{s4.output_prefix}_part{submap_count}.pcd"
            )
            o3d.io.write_point_cloud(save_path, submap_t.to_legacy(), write_ascii=False)
            print(f"\n  -> Saved Final Submap Part {submap_count}: {save_path}")
        del submap_t
        gc.collect()
        o3d.core.cuda.release_cache()
        print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Step4: Auto colorize (simple GNSS, ouster2)")
    parser.add_argument("--config", default="default.yaml")
    args = parser.parse_args()
    run_step4(load_config(args.config))


if __name__ == "__main__":
    main()
