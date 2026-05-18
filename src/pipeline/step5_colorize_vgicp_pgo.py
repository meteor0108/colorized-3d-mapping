"""5단계: VGICP + PGO 기반 멀티-LiDAR 컬러라이즈 (GPU 배치 처리).

원본: 5_pointcloud_colorlize_show_vgicp_pgo_gpu.py (Jan 20, 832 lines).
구버전(`5_pointcloud_colorlize_show_vgicp_pgo.py`)는 _legacy/ 로 이동됨.
모든 처리 파라미터는 configs/colorize.yaml > step5_vgicp_pgo 에서 조정.
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
from src.pgo import PoseGraphOptimizer, VGICPAligner


class AutomatedLiDARCameraFusion:
    """Config 기반 자동 배치 LiDAR-Camera 융합."""

    def __init__(self, cfg: ConfigDict):
        self.cfg = cfg
        self.calib = Calibration.from_config(cfg)

        gpu_cfg = cfg.colorize.gpu
        self.is_cuda = gpu_cfg.enabled and o3d.core.cuda.is_available()
        self.device = o3d.core.Device(gpu_cfg.device if self.is_cuda else "CPU")
        print(f"System Device: {self.device}")

        proc = cfg.colorize.processing
        self.distance_threshold = proc.distance_threshold
        self.projection_distance = proc.projection_distance
        self.voxel_size = proc.voxel_size
        self.time_tolerance = proc.time_tolerance
        self.frame_skip = proc.frame_skip

        s5 = cfg.colorize.step5_vgicp_pgo
        self.output_prefix = s5.output_prefix
        self.submap_frame_limit = s5.frame_limit

        self.use_vgicp = bool(s5.vgicp.enabled)
        self.use_pgo = bool(s5.pgo.enabled)
        self.s5 = s5

        self.root_path = cfg.paths.data_root
        self.output_folder = cfg.outputs.vgicp_pgo_map
        os.makedirs(self.output_folder, exist_ok=True)

    def find_target_folders(self):
        depth3 = glob.glob(os.path.join(self.root_path, "*", "*", "20*"))
        depth2 = glob.glob(os.path.join(self.root_path, "*", "2024*"))
        folders = sorted(set(depth3 + depth2))
        return [f for f in folders if os.path.exists(os.path.join(f, "ouster2"))]

    def _process_lidar(self, lidar_path, image, ext_lidar):
        if lidar_path is None or not os.path.exists(lidar_path):
            return None
        pcd = o3d.io.read_point_cloud(lidar_path)
        if pcd.is_empty():
            return None
        pcd = PointCloudProcessor.range_filter(pcd, distance_threshold=self.distance_threshold)
        _, colorized = PointCloudProcessor.project_to_image(
            pcd, image, ext_lidar, self.calib.intrinsic, self.calib.distortion,
            self.calib.fisheye,
            max_range=self.projection_distance,
            mask_params=self.calib.mask_params(),
            cutoff_y=self.calib.cutoff_y(),
            draw_overlay=False,
        )
        return colorized

    def _save_submap(self, submap_t, folder_name, part_idx):
        save_path = os.path.join(
            self.output_folder, f"{folder_name}_{self.output_prefix}_part{part_idx}.pcd"
        )
        try:
            o3d.t.io.write_point_cloud(save_path, submap_t, write_ascii=False)
        except Exception:
            o3d.io.write_point_cloud(save_path, submap_t.to_legacy(), write_ascii=False)
        return save_path

    def process_folder(self, folder_path: str) -> None:
        folder_name = os.path.basename(folder_path)
        check_path = os.path.join(self.output_folder, f"{folder_name}_{self.output_prefix}_part0.pcd")
        if self.cfg.colorize.submap.skip_processed and os.path.exists(check_path):
            print(f"Skipping {folder_name} (Already Exists)")
            return

        print(f"\nProcessing Folder: {folder_name}")

        pc_folders = {
            "ouster1": os.path.join(folder_path, "ouster1/points/"),
            "ouster2": os.path.join(folder_path, "ouster2/points/"),
            "ouster3": os.path.join(folder_path, "ouster3/points/"),
        }
        image_folder = os.path.join(folder_path, "blackfly/")
        navigation_file = os.path.join(folder_path, "navigation.csv")

        if not os.path.exists(navigation_file):
            print("  -> Navigation file not found")
            return

        df_nav = pd.read_csv(navigation_file)
        df_nav.columns = df_nav.columns.str.strip()
        time_cols = [c for c in df_nav.columns if "time" in c.lower() or "stamp" in c.lower()]
        if not time_cols:
            print("  -> No timestamp column found")
            return
        nav_timestamps = df_nav[time_cols[0]].values

        image_files = natsort.natsorted(os.listdir(image_folder))
        files_times = {
            name: FileManager.get_files_and_times(folder) for name, folder in pc_folders.items()
        }

        vgicp = VGICPAligner.from_config(self.s5.vgicp) if self.use_vgicp else None
        pgo = PoseGraphOptimizer.from_config(self.s5.pgo) if self.use_pgo else None

        submap_t = None
        submap_count = 0
        is_origin_set = False
        origin = np.zeros(3)
        frame_count = 0
        prev_pose = np.eye(4)
        local_colorized_pcds = []

        sensors_use = {
            "ouster1": self.cfg.colorize.sensors.use_ouster1,
            "ouster2": self.cfg.colorize.sensors.use_ouster2,
            "ouster3": self.cfg.colorize.sensors.use_ouster3,
        }

        for i, image_file in enumerate(image_files):
            if self.frame_skip > 1 and i % self.frame_skip != 0:
                continue
            if i % 10 == 0:
                p = len(submap_t.point["positions"]) if submap_t is not None else 0
                print(f"  -> Frame {i}/{len(image_files)} | Part {submap_count} | GPU Pts: {p}", end="\r")

            if submap_t is not None and i > 0 and i % self.submap_frame_limit == 0:
                self._save_submap(submap_t, folder_name, submap_count)
                del submap_t
                submap_t = None
                submap_count += 1
                gc.collect()
                o3d.core.cuda.release_cache()

            query_time = TimestampParser.parse(image_file)
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

            image_path = os.path.join(image_folder, image_file)
            image = cv2.imread(image_path)
            if image is None:
                continue
            if self.calib.vignetting.get("enabled", False):
                image = PointCloudProcessor.apply_vignetting_correction(
                    image, strength=self.calib.vignetting.get("strength", 0.4)
                )

            lidar_paths = {}
            for name, folder in pc_folders.items():
                files, times = files_times[name]
                fn = FileManager.match_by_time(query_time, files, times, self.time_tolerance)
                lidar_paths[name] = os.path.join(folder, fn) if fn else None

            if not is_origin_set:
                origin = pos.copy()
                is_origin_set = True

            trajectory_gnss = np.eye(4)
            trajectory_gnss[:3, :3] = R.from_quat(ori).as_matrix()
            trajectory_gnss[:3, 3] = pos - origin

            if pgo is not None:
                pgo.add_pose(trajectory_gnss)

            frame_pcds = []
            merged_frame_local = o3d.geometry.PointCloud()
            for name in ("ouster1", "ouster2", "ouster3"):
                if not sensors_use[name]:
                    continue
                pcd_local = self._process_lidar(
                    lidar_paths[name], image, self.calib.lidar_to_camera[name]
                )
                if pcd_local is None:
                    continue
                ext_gps = self.calib.gps_to_lidar[name]
                frame_pcds.append({"pcd": pcd_local, "ext_gps": ext_gps})
                merged_frame_local += copy.deepcopy(pcd_local).transform(ext_gps)

            if frame_pcds:
                local_colorized_pcds.append({"pcds": frame_pcds, "gnss_pose": trajectory_gnss})

            if vgicp is not None and not merged_frame_local.is_empty():
                refined_pose, fitness = vgicp.align(merged_frame_local, trajectory_gnss)
                if pgo is not None and frame_count > 0:
                    rel = np.linalg.inv(prev_pose) @ refined_pose
                    info = np.eye(6) * (1.0 if fitness > 0.5 else 0.1)
                    pgo.add_edge(frame_count - 1, frame_count, rel, information=info)
                prev_pose = refined_pose

            if not merged_frame_local.is_empty():
                try:
                    final_pose = prev_pose if vgicp is not None else trajectory_gnss
                    for pcd_data in frame_pcds:
                        pcd_global = copy.deepcopy(pcd_data["pcd"]).transform(
                            final_pose @ pcd_data["ext_gps"]
                        )
                        pcd_t = o3d.t.geometry.PointCloud.from_legacy(pcd_global, device=self.device)
                        if not pcd_t.is_empty():
                            pcd_t = pcd_t.voxel_down_sample(self.voxel_size)
                        else:
                            continue
                        submap_t = pcd_t if submap_t is None else submap_t.append(pcd_t)
                        del pcd_t, pcd_global
                except RuntimeError as e:
                    print(f"\n[GPU Error] {image_file}: {e}")
                    o3d.core.cuda.release_cache()
                    continue

            frame_count += 1

        if pgo is not None and len(local_colorized_pcds) > 0:
            print(f"\n[PGO] Starting optimization for {folder_name}...")
            pgo.optimize()
            print("[PGO] Applying optimized poses...")

            if submap_t is not None:
                del submap_t
                submap_t = None
                submap_count = 0
                gc.collect()
                o3d.core.cuda.release_cache()

            optimized_poses = pgo.get_optimized_poses()
            for idx, frame_data in enumerate(local_colorized_pcds):
                if idx >= len(optimized_poses):
                    break
                if idx % 10 == 0:
                    print(f"  -> Applying PGO: {idx}/{len(local_colorized_pcds)}", end="\r")
                opt_pose = optimized_poses[idx]
                for pcd_data in frame_data["pcds"]:
                    try:
                        pcd_global = copy.deepcopy(pcd_data["pcd"]).transform(
                            opt_pose @ pcd_data["ext_gps"]
                        )
                        pcd_t = o3d.t.geometry.PointCloud.from_legacy(pcd_global, device=self.device)
                        pcd_t.remove_non_finite_points()
                        pcd_t.remove_duplicated_points()
                        if len(pcd_t.point) > 0:
                            pcd_t = pcd_t.voxel_down_sample(self.voxel_size)
                        submap_t = pcd_t if submap_t is None else submap_t.append(pcd_t)
                        del pcd_t, pcd_global
                        if idx > 0 and idx % self.submap_frame_limit == 0:
                            self._save_submap(submap_t, folder_name, submap_count)
                            del submap_t
                            submap_t = None
                            submap_count += 1
                            gc.collect()
                            o3d.core.cuda.release_cache()
                    except RuntimeError as e:
                        print(f"\n[GPU Error during PGO apply]: {e}")
                        o3d.core.cuda.release_cache()
                        continue

        if submap_t is not None:
            path = self._save_submap(submap_t, folder_name, submap_count)
            print(f"\n  -> Saved Final Submap Part {submap_count}: {path}")

        del submap_t
        gc.collect()
        o3d.core.cuda.release_cache()
        print(f"\n[Complete] Finished processing {folder_name}")
        print("-" * 60)

    def run(self):
        targets = self.find_target_folders()
        print(f"\n[Automated Processing] Found {len(targets)} folders")
        print(f"VGICP: {self.use_vgicp}, PGO: {self.use_pgo}")
        print(f"Output folder: {self.output_folder}\n")
        for f_idx, folder in enumerate(targets):
            print(f"\n[{f_idx + 1}/{len(targets)}] Processing: {os.path.basename(folder)}")
            self.process_folder(folder)
        print("\n[All Complete] Automated processing finished!")


def main():
    parser = argparse.ArgumentParser(description="Step5: VGICP+PGO LiDAR-Camera fusion")
    parser.add_argument("--config", default="default.yaml", help="config 파일 (configs/ 기준)")
    parser.add_argument("--no-vgicp", action="store_true")
    parser.add_argument("--no-pgo", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.no_vgicp:
        cfg.colorize.step5_vgicp_pgo.vgicp.enabled = False
    if args.no_pgo:
        cfg.colorize.step5_vgicp_pgo.pgo.enabled = False

    AutomatedLiDARCameraFusion(cfg).run()


if __name__ == "__main__":
    main()
