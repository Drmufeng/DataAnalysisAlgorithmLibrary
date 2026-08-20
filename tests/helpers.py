"""测试使用的固定表格和协议请求。"""

from __future__ import annotations

import pandas as pd

from algorithm_sdk.models import AlgorithmRequest


def make_sample_data() -> pd.DataFrame:
    """创建覆盖数值、类别、缺失和时间字段的固定数据。"""

    return pd.DataFrame(
        {
            "编号": [1, 2, 3, 4, 4],
            "组别": ["甲", "乙", "甲", "乙", "乙"],
            "成绩": [80.0, 90.0, None, 70.0, 70.0],
            "提交时间": pd.to_datetime(
                ["2026-08-01", "2026-08-02", "2026-08-03", "2026-08-04", "2026-08-04"]
            ),
        }
    )


def make_overview_request() -> AlgorithmRequest:
    """创建数据概览的完整字段快照请求。"""

    return AlgorithmRequest.model_validate(
        {
            "operation_key": "data_overview.basic",
            "slots": {"columns": ["编号", "组别", "成绩", "提交时间"]},
            "parameters": {"preview_rows": 3},
            "context": {
                "run_id": "test-run-001",
                "step_id": "step-1",
                "locale": "zh-CN",
                "random_seed": 42,
                "protocol_version": "1.0",
                "field_metadata": {
                    "编号": {
                        "data_type": "integer",
                        "analysis_type": "categorical",
                        "semantic_role": "identifier",
                    },
                    "组别": {
                        "data_type": "string",
                        "analysis_type": "categorical",
                    },
                    "成绩": {
                        "data_type": "number",
                        "analysis_type": "quantitative",
                    },
                    "提交时间": {
                        "data_type": "datetime",
                        "analysis_type": "categorical",
                        "semantic_role": "time",
                    },
                },
            },
        }
    )
