"""公共模型和严格 JSON 转换测试。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from algorithm_sdk.models import AlgorithmRequest, AlgorithmResult, Metric
from algorithm_sdk.serialization import result_to_payload, to_json_compatible
from tests.helpers import make_overview_request


def test_request_rejects_invalid_operation_key() -> None:
    """组合编号不能使用中文名称或缺少方法编号。"""

    payload = make_overview_request().model_dump(mode="json")
    payload["operation_key"] = "数据概览"
    with pytest.raises(ValidationError):
        AlgorithmRequest.model_validate(payload)


def test_result_serialization_converts_scientific_values() -> None:
    """NumPy 标量、时间和非有限值必须转换为严格 JSON。"""

    result = AlgorithmResult(
        metrics=[Metric(id="count", display_name="数量", value=np.int64(3))],
        metadata={
            "missing": np.nan,
            "positive_infinity": np.inf,
            "time": pd.Timestamp("2026-08-18 12:00:00"),
        },
    )
    payload = result_to_payload(result)
    assert payload["metadata"] == {
        "missing": None,
        "positive_infinity": None,
        "time": "2026-08-18T12:00:00",
    }
    json.dumps(payload, allow_nan=False)


def test_unknown_result_type_is_rejected() -> None:
    """路径等未声明类型不能被悄悄转换为字符串。"""

    with pytest.raises(TypeError, match="不能序列化"):
        to_json_compatible(object())
