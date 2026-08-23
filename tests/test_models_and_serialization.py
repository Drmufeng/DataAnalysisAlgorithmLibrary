"""公共模型和严格 JSON 转换测试。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from algorithm_sdk.models import (
    AlgorithmRequest,
    AlgorithmResult,
    ChartSpec,
    Metric,
    ReportBlockSpec,
    ReportPresentationSpec,
)
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


def test_report_presentation_serializes_and_validates_references() -> None:
    """报告排版只引用结构化结果，并保持与具体图表前端无关。"""

    chart = ChartSpec(
        id="trend",
        display_name="趋势图",
        chart_type="line",
        data={"x": [1, 2], "y": [3, 4]},
    )
    result = AlgorithmResult(
        metrics=[Metric(id="count", display_name="样本量", value=2)],
        charts=[chart],
        presentation=ReportPresentationSpec(
            title="分析结果",
            blocks=[
                ReportBlockSpec(
                    id="summary",
                    block_type="metrics",
                    reference_ids=["count"],
                    width="full",
                    order=0,
                ),
                ReportBlockSpec(
                    id="trend_chart",
                    block_type="chart",
                    reference_ids=["trend"],
                    width="full",
                    order=1,
                ),
            ],
        ),
    )
    payload = result_to_payload(result)
    presentation = payload["presentation"]
    assert isinstance(presentation, dict)
    blocks = presentation["blocks"]
    assert isinstance(blocks, list)
    trend_block = blocks[1]
    assert isinstance(trend_block, dict)
    assert trend_block["reference_ids"] == ["trend"]

    with pytest.raises(ValidationError, match="不存在的结果"):
        AlgorithmResult(
            presentation=ReportPresentationSpec(
                blocks=[
                    ReportBlockSpec(
                        id="missing_chart",
                        block_type="chart",
                        reference_ids=["not_found"],
                        order=0,
                    )
                ]
            )
        )
