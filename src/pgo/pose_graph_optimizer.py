"""GNSS unary + odometry binary factor를 갖는 단순 Pose Graph Optimization.

원본: 5_pointcloud_colorlize_show_vgicp_pgo_gpu.py.
ConfigDict.colorize.step5_vgicp_pgo.pgo 를 받아 파라미터화.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation as R


class PoseGraphOptimizer:
    def __init__(
        self,
        gnss_weight: float = 1000.0,
        odom_weight: float = 1.0,
        max_iterations: int = 50,
        tolerance: float = 1e-6,
        rotation_weight_scale: float = 0.1,
        delta_clip: float = 1.0,
    ):
        self.gnss_weight = gnss_weight
        self.odom_weight = odom_weight
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.rotation_weight_scale = rotation_weight_scale
        self.delta_clip = delta_clip

        self.poses_gnss: list[np.ndarray] = []
        self.poses_optimized: list[np.ndarray] = []
        self.edges: list[dict] = []
        self.vgicp_transforms: list[np.ndarray] = []

    @classmethod
    def from_config(cls, pgo_cfg) -> "PoseGraphOptimizer":
        return cls(
            gnss_weight=pgo_cfg.gnss_weight,
            odom_weight=pgo_cfg.odom_weight,
            max_iterations=pgo_cfg.max_iterations,
            tolerance=pgo_cfg.tolerance,
            rotation_weight_scale=pgo_cfg.rotation_weight_scale,
            delta_clip=pgo_cfg.delta_clip,
        )

    def add_pose(self, gnss_pose: np.ndarray) -> None:
        self.poses_gnss.append(gnss_pose.copy())
        self.poses_optimized.append(gnss_pose.copy())

    def add_edge(self, i: int, j: int, relative_transform: np.ndarray, information=None) -> None:
        if information is None:
            information = np.eye(6) * self.odom_weight
        self.edges.append(
            {"i": i, "j": j, "transform": relative_transform, "information": information}
        )
        self.vgicp_transforms.append(relative_transform)

    @staticmethod
    def _pose_to_vec(pose):
        x, y, z = pose[:3, 3]
        roll, pitch, yaw = R.from_matrix(pose[:3, :3]).as_euler("xyz")
        return np.array([x, y, z, roll, pitch, yaw])

    @staticmethod
    def _vec_to_pose(vec):
        out = np.eye(4)
        out[:3, 3] = vec[:3]
        out[:3, :3] = R.from_euler("xyz", vec[3:6]).as_matrix()
        return out

    def _relative_error(self, pose_i, pose_j, measured):
        estimated = np.linalg.inv(pose_i) @ pose_j
        err = np.linalg.inv(measured) @ estimated
        return self._pose_to_vec(err)

    def optimize(self) -> None:
        if len(self.poses_gnss) < 2:
            print("[PGO] 포즈가 부족합니다")
            return

        print(f"\n[PGO] 최적화 시작: {len(self.poses_gnss)} 포즈, {len(self.edges)} 에지")
        n = len(self.poses_gnss)

        for it in range(self.max_iterations):
            H = lil_matrix((n * 6, n * 6))
            b = np.zeros(n * 6)
            total_err = 0.0

            for i in range(n):
                err = self._pose_to_vec(self.poses_optimized[i]) - self._pose_to_vec(self.poses_gnss[i])
                info = np.eye(6)
                info[:3, :3] *= self.gnss_weight
                info[3:, 3:] *= self.gnss_weight * self.rotation_weight_scale
                J = np.eye(6)
                idx = i * 6
                H[idx:idx + 6, idx:idx + 6] += J.T @ info @ J
                b[idx:idx + 6] += J.T @ info @ err
                total_err += float(np.sum(err**2))

            for edge in self.edges:
                i, j = edge["i"], edge["j"]
                err = self._relative_error(
                    self.poses_optimized[i], self.poses_optimized[j], edge["transform"]
                )
                info = edge["information"]
                J_i, J_j = -np.eye(6), np.eye(6)
                ii, jj = i * 6, j * 6
                H[ii:ii + 6, ii:ii + 6] += J_i.T @ info @ J_i
                H[ii:ii + 6, jj:jj + 6] += J_i.T @ info @ J_j
                H[jj:jj + 6, ii:ii + 6] += J_j.T @ info @ J_i
                H[jj:jj + 6, jj:jj + 6] += J_j.T @ info @ J_j
                b[ii:ii + 6] += J_i.T @ info @ err
                b[jj:jj + 6] += J_j.T @ info @ err
                total_err += float(np.sum(err**2))

            delta = spsolve(H.tocsr(), -b)

            max_d = 0.0
            for i in range(n):
                di = np.clip(delta[i * 6:(i + 1) * 6], -self.delta_clip, self.delta_clip)
                cur = self._pose_to_vec(self.poses_optimized[i])
                self.poses_optimized[i] = self._vec_to_pose(cur + di)
                max_d = max(max_d, float(np.linalg.norm(di)))

            if it % 5 == 0:
                print(f"[PGO] Iter {it}: Error={total_err:.6f}, Max Delta={max_d:.6f}")
            if max_d < self.tolerance:
                print(f"[PGO] 수렴 완료 at iteration {it}")
                break

        print("[PGO] 최적화 완료")

    def get_optimized_poses(self):
        return self.poses_optimized
