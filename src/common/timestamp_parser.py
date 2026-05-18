"""파일명 → Unix timestamp 파싱.

여러 포맷을 지원하며 Unix timestamp 문자열(예: 1710823801290584411)도 처리.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Optional


class TimestampParser:
    FORMATS = [
        "%Y-%m-%d_%H-%M-%S.%f",
        "%Y-%m-%d-%H-%M-%S_%f",
        "%Y-%m-%d-%H-%M-%S.%f",
        "%Y-%m-%d_%H-%M-%S",
        "%Y-%m-%d-%H-%M-%S",
    ]

    @staticmethod
    def parse(filename: str) -> Optional[float]:
        name_body = os.path.splitext(filename)[0]

        try:
            return float(name_body)
        except ValueError:
            pass

        for fmt in TimestampParser.FORMATS:
            try:
                dt = datetime.strptime(name_body, fmt)
                return dt.timestamp()
            except ValueError:
                continue

        return None
