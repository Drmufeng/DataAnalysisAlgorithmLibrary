"""科学计算值到严格 JSON 值的底层转换。"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time
from enum import Enum

import numpy as np
import pandas as pd
from pydantic import JsonValue


def to_json_compatible(value: object) -> JsonValue:
    """递归转换常见科学计算类型，并把非有限数值转换为 null。"""

    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return to_json_compatible(value.item())
    if isinstance(value, np.ndarray):
        return [to_json_compatible(item) for item in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, pd.Timedelta):
        return value.isoformat()
    if isinstance(value, (pd.Interval, pd.Period)):
        return str(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Enum):
        return to_json_compatible(value.value)
    if isinstance(value, Mapping):
        converted: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON 对象的键必须是字符串")
            converted[key] = to_json_compatible(item)
        return converted
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_json_compatible(item) for item in value]
    raise TypeError(f"结果包含不能序列化的类型：{type(value).__name__}")
