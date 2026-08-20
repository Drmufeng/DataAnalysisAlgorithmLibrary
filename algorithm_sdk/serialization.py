"""把算法结果转换为严格 JSON 数据。"""

from __future__ import annotations

import json

from pydantic import JsonValue

from algorithm_sdk.json_compat import to_json_compatible
from algorithm_sdk.models import AlgorithmResult

__all__ = ["result_to_payload", "to_json_compatible"]


def result_to_payload(result: AlgorithmResult) -> dict[str, JsonValue]:
    """移除 DataFrame 后生成可以被严格 JSON 编码的结果。"""

    raw = result.model_dump(mode="python", exclude={"data"})
    converted = to_json_compatible(raw)
    if not isinstance(converted, dict):
        raise TypeError("算法结果必须转换为 JSON 对象")
    # allow_nan=False 是最后一道检查，避免 NaN 或 Infinity 泄漏到结果文件。
    json.dumps(converted, ensure_ascii=False, allow_nan=False)
    return converted
