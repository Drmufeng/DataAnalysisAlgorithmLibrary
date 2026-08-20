"""数据概览算法结果与边界测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from algorithm_sdk.serialization import result_to_payload
from runner_contract.runner import run_operation
from tests.helpers import make_overview_request, make_sample_data

PROJECT_ROOT = Path(__file__).parents[1]
PACKAGE_DIR = PROJECT_ROOT / "packages" / "descriptive_analysis"


def test_data_overview_returns_expected_summary_without_mutation() -> None:
    """概览结果应可复核，并保持输入表不变。"""

    data = make_sample_data()
    original = data.copy(deep=True)
    result = run_operation(PACKAGE_DIR, data, make_overview_request())
    pd.testing.assert_frame_equal(data, original)

    metrics = {metric.id: metric.value for metric in result.metrics}
    assert metrics == {
        "row_count": 5,
        "column_count": 4,
        "selected_column_count": 4,
        "duplicate_row_count": 1,
    }
    assert result.data is None
    assert result.metadata["preview_rows"] == 3

    field_table = next(table for table in result.tables if table.id == "field_overview")
    score_row = next(row for row in field_table.rows if row["field"] == "成绩")
    assert score_row["missing_count"] == 1
    assert score_row["missing_rate"] == 0.2
    assert score_row["minimum"] == 70.0
    assert score_row["maximum"] == 90.0

    payload = result_to_payload(result)
    json.dumps(payload, ensure_ascii=False, allow_nan=False)


def test_empty_table_returns_warning_instead_of_crashing() -> None:
    """有字段但零行的数据仍可概览，并明确提示空数据。"""

    empty_data = make_sample_data().iloc[0:0].copy()
    result = run_operation(PACKAGE_DIR, empty_data, make_overview_request())
    assert any(warning.code == "EMPTY_DATA" for warning in result.warnings)
    metrics = {metric.id: metric.value for metric in result.metrics}
    assert metrics["row_count"] == 0
