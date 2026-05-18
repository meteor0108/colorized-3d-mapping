"""YAML 기반 Config 로더.

사용 예:
    from src.common import load_config
    cfg = load_config("configs/default.yaml")
    voxel = cfg.colorize.processing.voxel_size
    intrinsic = cfg.camera.intrinsic  # numpy array로 자동 변환
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # data_show/
CONFIGS_DIR = REPO_ROOT / "configs"


class ConfigDict(dict):
    """dict처럼 동작하되 cfg.key.subkey 점 접근을 허용."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        del self[key]


def _is_numeric_matrix(node: list) -> bool:
    """2D 수치 리스트인지."""
    if not node or not all(isinstance(row, list) for row in node):
        return False
    return all(all(isinstance(x, (int, float)) for x in row) for row in node)


def _is_numeric_vector(node: list) -> bool:
    """1D 수치 리스트인지 (단, 비어있지 않고 string/dict 없음)."""
    return bool(node) and all(isinstance(x, (int, float)) for x in node)


def _to_config_dict(node: Any) -> Any:
    if isinstance(node, dict):
        return ConfigDict({k: _to_config_dict(v) for k, v in node.items()})
    if isinstance(node, list):
        if _is_numeric_matrix(node):
            return np.array(node, dtype=np.float64)
        if _is_numeric_vector(node):
            return np.array(node, dtype=np.float64)
        return [_to_config_dict(x) for x in node]
    return node


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve(path: str | os.PathLike) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = CONFIGS_DIR / p
    return p


def load_config(path: str | os.PathLike = "default.yaml") -> ConfigDict:
    """yaml 파일 로드. `includes:` 키를 통한 다중 파일 합성을 지원."""
    p = _resolve(path)
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    includes = data.pop("includes", [])
    merged: dict = {}
    for inc in includes:
        sub = load_config(inc)
        merged = _merge(merged, sub)
    merged = _merge(merged, data)

    return _to_config_dict(merged)


def build_gps_to_lidar_extrinsics(cfg: ConfigDict) -> dict:
    """sensor_calibration.yaml로부터 GPS->각 LiDAR 변환행렬 계산.

    원본 코드의 R_fix 적용 + y_translation 부호 반전 로직을 한 곳에 통합.
    """
    extrinsic_gps2ouster2 = cfg.gps_to_ouster2_raw.copy()
    R_fix = cfg.gps_axis_fix
    extrinsic_gps2ouster2 = extrinsic_gps2ouster2 @ R_fix
    extrinsic_gps2ouster2[1, 3] = -extrinsic_gps2ouster2[1, 3]

    ouster2 = cfg.lidar_to_camera["ouster2"]
    T_L2_to_Cam = np.linalg.inv(ouster2)

    return {
        "ouster1": extrinsic_gps2ouster2 @ T_L2_to_Cam @ cfg.lidar_to_camera["ouster1"],
        "ouster2": extrinsic_gps2ouster2,
        "ouster3": extrinsic_gps2ouster2 @ T_L2_to_Cam @ cfg.lidar_to_camera["ouster3"],
    }
