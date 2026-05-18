"""Sensor 파일 + timestamp 추출 유틸."""
from __future__ import annotations

import os

import natsort
import numpy as np

from .timestamp_parser import TimestampParser


class FileManager:
    @staticmethod
    def get_files_and_times(folder: str, ext: str = ".pcd"):
        if not os.path.exists(folder):
            return [], np.array([])

        files = natsort.natsorted(os.listdir(folder))
        valid_files, times = [], []
        for f in files:
            if not f.endswith(ext):
                continue
            ts = TimestampParser.parse(f)
            if ts is not None:
                valid_files.append(f)
                times.append(ts)
        return valid_files, np.array(times)

    @staticmethod
    def match_by_time(query_time: float, files, times, tolerance: float = 0.2):
        if len(times) == 0:
            return None
        idx = int(np.abs(times - query_time).argmin())
        if abs(times[idx] - query_time) < tolerance:
            return files[idx]
        return None
