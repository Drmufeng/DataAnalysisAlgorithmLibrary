"""数据概览算法。"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
from pydantic import JsonValue

from algorithm_sdk import (
    AlgorithmResult,
    DataType,
    Metric,
    ResultTable,
    TableColumn,
    WarningItem,
    to_json_compatible,
)
from algorithm_sdk.validation import infer_data_type


def run(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """生成字段质量摘要和数据预览，不修改传入表格。"""

    columns = list(slots.get("columns", []))
    preview_rows = _integer_parameter(parameters, "preview_rows", 10)
    include_missing = _boolean_parameter(parameters, "include_missing", True)
    include_unique = _boolean_parameter(parameters, "include_unique", True)
    include_duplicates = _boolean_parameter(parameters, "include_duplicates", True)

    selected = data.loc[:, columns]
    field_rows: list[dict[str, JsonValue]] = []
    warnings: list[WarningItem] = []
    for column in columns:
        series = selected[column]
        missing_count = int(series.isna().sum())
        valid_count = int(series.notna().sum())
        actual_type = infer_data_type(series)
        minimum, maximum = _numeric_range(series, actual_type)
        field_rows.append(
            {
                "field": column,
                "data_type": actual_type.value,
                "analysis_type": _analysis_type(context, column),
                "valid_count": valid_count,
                "missing_count": missing_count if include_missing else None,
                "missing_rate": (
                    missing_count / len(series.index)
                    if include_missing and len(series.index) > 0
                    else None
                ),
                "unique_count": int(series.nunique(dropna=True)) if include_unique else None,
                "minimum": minimum,
                "maximum": maximum,
            }
        )
        if len(series.index) > 0 and missing_count / len(series.index) >= 0.5:
            warnings.append(
                WarningItem(
                    code="HIGH_MISSING_RATE",
                    message=f"字段 {column} 的缺失比例达到 50% 或以上",
                    details={"column": column, "missing_rate": missing_count / len(series.index)},
                )
            )

    metrics = [
        Metric(id="row_count", display_name="总行数", value=len(data.index), unit="行"),
        Metric(id="column_count", display_name="总字段数", value=len(data.columns), unit="个"),
        Metric(
            id="selected_column_count",
            display_name="概览字段数",
            value=len(columns),
            unit="个",
        ),
    ]
    duplicate_rows: int | None = None
    if include_duplicates:
        duplicate_rows = int(selected.duplicated(keep="first").sum())
        metrics.append(
            Metric(
                id="duplicate_row_count",
                display_name="重复行数",
                value=duplicate_rows,
                unit="行",
            )
        )
    if data.empty:
        warnings.append(
            WarningItem(
                code="EMPTY_DATA",
                message="当前数据版本包含字段，但没有数据行",
            )
        )

    preview = selected.head(preview_rows)
    preview_rows_payload = [
        {column: to_json_compatible(value) for column, value in row.items()}
        for row in preview.to_dict(orient="records")
    ]
    return AlgorithmResult(
        metrics=metrics,
        tables=[
            ResultTable(
                id="field_overview",
                display_name="字段概览",
                columns=_field_overview_columns(),
                rows=field_rows,
            ),
            ResultTable(
                id="data_preview",
                display_name=f"前 {min(preview_rows, len(data.index))} 行预览",
                columns=[
                    TableColumn(
                        id=column,
                        display_name=column,
                        data_type=infer_data_type(selected[column]),
                    )
                    for column in columns
                ],
                rows=preview_rows_payload,
            ),
        ],
        warnings=warnings,
        metadata={
            "method": "data_overview.basic",
            "rows_in": len(data.index),
            "rows_out": len(data.index),
            "columns_in": len(data.columns),
            "selected_columns": columns,
            "preview_rows": len(preview.index),
            "duplicate_rows": duplicate_rows,
        },
    )


def _integer_parameter(
    parameters: Mapping[str, JsonValue],
    name: str,
    default: int,
) -> int:
    """读取已由执行器校验过的整数参数。"""

    value = parameters.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"参数 {name} 必须是整数")
    return value


def _boolean_parameter(
    parameters: Mapping[str, JsonValue],
    name: str,
    default: bool,
) -> bool:
    """读取已由执行器校验过的布尔参数。"""

    value = parameters.get(name, default)
    if not isinstance(value, bool):
        raise TypeError(f"参数 {name} 必须是布尔值")
    return value


def _analysis_type(context: Mapping[str, JsonValue], column: str) -> str:
    """从运行快照读取用户当时选择的定类或定量标签。"""

    field_metadata = context.get("field_metadata")
    if not isinstance(field_metadata, dict):
        return "unknown"
    metadata = field_metadata.get(column)
    if not isinstance(metadata, dict):
        return "unknown"
    value = metadata.get("analysis_type")
    return value if isinstance(value, str) else "unknown"


def _numeric_range(series: pd.Series, data_type: DataType) -> tuple[JsonValue, JsonValue]:
    """只为数值字段计算范围，避免对类别文本做误导性大小比较。"""

    if data_type not in {DataType.INTEGER, DataType.NUMBER}:
        return None, None
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return None, None
    return to_json_compatible(valid.min()), to_json_compatible(valid.max())


def _field_overview_columns() -> list[TableColumn]:
    """返回字段概览表的稳定列定义。"""

    return [
        TableColumn(id="field", display_name="字段", data_type=DataType.STRING),
        TableColumn(id="data_type", display_name="实际类型", data_type=DataType.STRING),
        TableColumn(id="analysis_type", display_name="分析标签", data_type=DataType.STRING),
        TableColumn(id="valid_count", display_name="有效数", data_type=DataType.INTEGER),
        TableColumn(id="missing_count", display_name="缺失数", data_type=DataType.INTEGER),
        TableColumn(id="missing_rate", display_name="缺失比例", data_type=DataType.NUMBER),
        TableColumn(id="unique_count", display_name="唯一值数", data_type=DataType.INTEGER),
        TableColumn(id="minimum", display_name="最小值", data_type=DataType.NUMBER),
        TableColumn(id="maximum", display_name="最大值", data_type=DataType.NUMBER),
    ]
