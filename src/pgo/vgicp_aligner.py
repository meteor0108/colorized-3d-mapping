"""Sliding-window 기반 VGICP Aligner.

GNSS를 강한 prior로 사용, drift_threshold를 넘으면 GNSS로 fallback.
"""
from __future__ import annotations

import copy
from collections import deque

import numpy as np
import open3d as o3d
import pygicp


class VGICPAligner:
    def __init__(
        self,
        voxel_size: float = 0.3,
        window_size: int = 20,
        max_correspondence_distance: float = 0.5,
        drift_threshold: float = 1.8,
        num_threads: int = 4,
        transformation_epsilon: float = 1e-6,
    ):
        self.voxel_size = voxel_size
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.local_frames = deque(maxlen=window_size)

        self.gicp = pygicp.FastVGICP()
        self.gicp.set_max_correspondence_distance(max_correspondence_distance)
        self.gicp.set_resolution(voxel_size)
        # 일부 빌드는 아래 API를 제공하지 않음 — 조심스럽게 시도.
        for setter, value in [
            ("set_num_threads", num_threads),
            ("set_transformation_epsilon", transformation_epsilon),
        ]:
            fn = getattr(self.gicp, setter, None)
            if callable(fn):
                try:
                    fn(value)
                except Exception:
                    pass

    @classmethod
    def from_config(cls, vgicp_cfg) -> "VGICPAligner":
        return cls(
            voxel_size=vgicp_cfg.voxel_size,
            window_size=vgicp_cfg.window_size,
            max_correspondence_distance=vgicp_cfg.max_correspondence_distance,
            drift_threshold=vgicp_cfg.drift_threshold,
            num_threads=vgicp_cfg.get("num_threads", 4),
            transformation_epsilon=vgicp_cfg.get("transformation_epsilon", 1e-6),
        )

    def _prepare(self, pcd):
        return pcd.voxel_down_sample(self.voxel_size)

    def update_local_map(self, pcd_global):
        self.local_frames.append(self._prepare(pcd_global))

    def get_current_map(self):
        if not self.local_frames:
            return None
        merged = o3d.geometry.PointCloud()
        for f in self.local_frames:
            merged += f
        return merged.voxel_down_sample(self.voxel_size)

    def align(self, current_pcd_local, initial_guess):
        if not self.local_frames:
            first = copy.deepcopy(current_pcd_local).transform(initial_guess)
            self.update_local_map(first)
            return initial_guess, 1.0

        source = self._prepare(current_pcd_local)
        target = self.get_current_map()

        self.gicp.set_input_target(np.asarray(target.points).astype(np.float64))
        self.gicp.set_input_source(np.asarray(source.points).astype(np.float64))
        corrected = self.gicp.align(initial_guess)

        drift = float(np.linalg.norm(corrected[:3, 3] - initial_guess[:3, 3]))
        final_pose = corrected
        if drift > self.drift_threshold:
            print(f"[VGICP] GNSS 차이 큼({drift:.2f}m). GNSS 포즈 사용.")
            final_pose = initial_guess

        self.update_local_map(copy.deepcopy(current_pcd_local).transform(final_pose))
        return final_pose, 1.0
