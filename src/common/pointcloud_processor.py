"""포인트클라우드 공통 처리 함수 (range filter, undistortion, projection)."""
from __future__ import annotations

import copy

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


class PointCloudProcessor:
    @staticmethod
    def range_filter(pcd, distance_threshold=30, query_point=(0, 0, 0)):
        points = np.asarray(pcd.points)
        distances = np.linalg.norm(points - np.asarray(query_point), axis=1)
        idx = np.where(distances <= distance_threshold)[0]
        return pcd.select_by_index(idx)

    @staticmethod
    def undistortion(x, y, distortion, r, fisheye=False):
        if not fisheye and distortion.shape[0] == 5:
            radial_x = x * (1 + distortion[0] * r**2 + distortion[1] * r**4 + distortion[4] * r**6)
            radial_y = y * (1 + distortion[0] * r**2 + distortion[1] * r**4 + distortion[4] * r**6)
            tang_x = 2 * distortion[2] * x * y + distortion[3] * (r**2 + 2 * x**2)
            tang_y = distortion[2] * (r**2 + 2 * y**2) + 2 * distortion[3] * x * y
            return radial_x + tang_x, radial_y + tang_y

        if not fisheye and distortion.shape[0] == 4:
            radial_x = x * (1 + distortion[0] * r**2 + distortion[1] * r**4)
            radial_y = y * (1 + distortion[0] * r**2 + distortion[1] * r**4)
            tang_x = 2 * distortion[2] * x * y + distortion[3] * (r**2 + 2 * x**2)
            tang_y = distortion[2] * (r**2 + 2 * y**2) + 2 * distortion[3] * x * y
            return radial_x + tang_x, radial_y + tang_y

        if fisheye:
            theta = np.arctan(r)
            theta_d = theta * (
                1
                + distortion[0] * theta**2
                + distortion[1] * theta**4
                + distortion[2] * theta**6
                + distortion[3] * theta**8
            )
            return (theta_d / r) * x, (theta_d / r) * y

        raise ValueError(f"unsupported distortion shape {distortion.shape}")

    @staticmethod
    def project_to_image(
        pointcloud,
        image_,
        extrinsic,
        intrinsic,
        distortion,
        fisheye,
        *,
        point_size=3,
        alpha=0.1,
        max_range=15,
        mask_params=None,
        cutoff_y=None,
        draw_overlay=True,
    ):
        """LiDAR 포인트를 이미지에 투영하고 컬러라이즈된 포인트클라우드 반환.

        Args:
            mask_params: (cx, cy, r) 원형 마스크. None이면 비활성.
            cutoff_y: 차량 하단 y픽셀. None이면 비활성.
            draw_overlay: True면 jet colormap을 이미지에 overlay.
        """
        lidar_pts = np.asarray(pointcloud.points)
        if len(lidar_pts) == 0:
            return image_, o3d.geometry.PointCloud()

        image = copy.deepcopy(image_)
        height, width = image.shape[:2]
        length = np.linalg.norm(lidar_pts, 2, axis=1)

        homo = extrinsic @ np.vstack((lidar_pts.T, np.ones(lidar_pts.shape[0])))
        x = homo[0, :] / homo[2, :]
        y = homo[1, :] / homo[2, :]
        r = np.sqrt(x**2 + y**2)

        xd, yd = PointCloudProcessor.undistortion(x, y, distortion, r, fisheye)
        pixel = intrinsic @ np.vstack((xd, yd, np.ones(xd.shape[0])))

        valid = (pixel[0, :] > 0) & (pixel[0, :] < width) & \
                (pixel[1, :] > 0) & (pixel[1, :] < height) & \
                (homo[2, :] > 0)

        if mask_params is not None:
            mcx, mcy, mr = mask_params
            dist_c = np.sqrt((pixel[0, :] - mcx) ** 2 + (pixel[1, :] - mcy) ** 2)
            valid &= dist_c < mr

        if cutoff_y is not None:
            valid &= pixel[1, :] < cutoff_y

        idx = np.where(valid)[0]
        if len(idx) == 0:
            return image, o3d.geometry.PointCloud()

        pixels = pixel.T[idx, :2].astype(int)
        projection_color = image_[pixels[:, 1], pixels[:, 0], :]

        if draw_overlay:
            cmap = plt.cm.get_cmap("jet", 256)
            cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
            color_idx = (255 * length[idx] / max_range).astype(int)
            color_idx[color_idx > 255] = 0
            color_map = cmap[color_idx]

            padding = 5
            image = cv2.copyMakeBorder(
                image, padding, padding, padding, padding,
                cv2.BORDER_CONSTANT, value=[0, 0, 0],
            )
            image[padding + pixels[:, 1], padding + pixels[:, 0], :] = color_map
            for i in range(-point_size, point_size + 1):
                for j in range(-point_size, point_size + 1):
                    try:
                        image[padding + pixels[:, 1] + i, padding + pixels[:, 0] + j, :] = (
                            image[padding + pixels[:, 1] + i, padding + pixels[:, 0] + j, :] * (1 - alpha)
                            + color_map * alpha
                        ).astype(np.uint8)
                    except IndexError:
                        pass
            image = image[padding:-padding, padding:-padding]

        out = o3d.geometry.PointCloud()
        out.points = o3d.utility.Vector3dVector(lidar_pts[idx])
        colors = np.flip(projection_color, axis=1) / 255.0
        out.colors = o3d.utility.Vector3dVector(colors)
        return image, out

    @staticmethod
    def apply_vignetting_correction(image, strength=0.5):
        """간이 다항식 모델 비네팅 보정."""
        height, width = image.shape[:2]
        X, Y = np.meshgrid(np.arange(width), np.arange(height))
        cx, cy = width / 2, height / 2
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        max_dist = np.sqrt(cx**2 + cy**2)
        mask = 1 + strength * (dist / max_dist) ** 2
        out = image.astype(np.float32)
        for i in range(3):
            out[:, :, i] *= mask
        return np.clip(out, 0, 255).astype(np.uint8)
