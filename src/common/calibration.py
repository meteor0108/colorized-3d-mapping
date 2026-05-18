"""센서 캘리브레이션 헬퍼.

ConfigDict 의 sensor_calibration 영역을 한 데로 묶어 자주 쓰는 행렬과 옵션을
하나의 객체로 노출. 각 파이프라인 스크립트가 동일한 캘리브레이션을 공유한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from .config_loader import ConfigDict, build_gps_to_lidar_extrinsics


@dataclass
class Calibration:
    intrinsic: np.ndarray
    distortion: np.ndarray
    fisheye: bool
    lidar_to_camera: Dict[str, np.ndarray]
    gps_to_lidar: Dict[str, np.ndarray]
    vignetting: dict = field(default_factory=dict)
    ros_topics: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: ConfigDict) -> "Calibration":
        return cls(
            intrinsic=cfg.camera.intrinsic,
            distortion=cfg.camera.distortion,
            fisheye=bool(cfg.camera.fisheye),
            lidar_to_camera=dict(cfg.lidar_to_camera),
            gps_to_lidar=build_gps_to_lidar_extrinsics(cfg),
            vignetting=dict(cfg.camera.vignetting),
            ros_topics=dict(cfg.get("ros_topics", {})),
        )

    def mask_params(self):
        if not self.vignetting.get("enabled", False):
            return None
        return self.vignetting.get("mask_params")

    def cutoff_y(self):
        if not self.vignetting.get("enabled", False):
            return None
        return self.vignetting.get("vehicle_cutoff_y")
