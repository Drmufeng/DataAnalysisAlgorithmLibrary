"""单变量时序滑窗特征与监督学习样本转换。"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from pydantic import JsonValue

from algorithm_sdk import AlgorithmResult, Metric, WarningItem
from algorithm_sdk.algorithm_utils import (
    get_bool_parameter,
    get_int_parameter,
    get_single_slot,
    get_slot,
    get_str_parameter,
    table_from_frame,
)
from algorithm_sdk.errors import ErrorCode, ProtocolError


def run(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """按稳定顺序把历史窗口展开为特征和可选目标。"""

    del context
    value_column = get_single_slot(slots, "value_column")
    order_columns = get_slot(slots, "order_column")
    window_size = get_int_parameter(parameters, "window_size", 3)
    step_size = get_int_parameter(parameters, "step_size", 1)
    output_mode = get_str_parameter(parameters, "output_mode", "supervised")
    forecast_horizon = get_int_parameter(parameters, "forecast_horizon", 1)
    include_incomplete = get_bool_parameter(parameters, "include_incomplete", False)
    flatten_output = get_bool_parameter(parameters, "flatten_output", True)
    if not flatten_output:
        raise ProtocolError(ErrorCode.PARAMETER_ERROR, "第一版只支持 flatten_output=true")
    ordered = data.copy(deep=True)
    warnings: list[WarningItem] = []
    order_column = order_columns[0] if order_columns else None
    if order_column:
        duplicate_policy = get_str_parameter(parameters, "duplicate_order", "error")
        if ordered[order_column].duplicated().any():
            if duplicate_policy == "error":
                raise ProtocolError(ErrorCode.DATA_ERROR, f"顺序字段 {order_column} 存在重复值")
            if duplicate_policy == "last":
                ordered = ordered.drop_duplicates(subset=order_column, keep="last")
            else:
                ordered = ordered.drop_duplicates(subset=order_column, keep="first")
        try:
            ordered = ordered.sort_values(order_column, kind="stable")
        except TypeError as exc:
            raise ProtocolError(ErrorCode.DATA_ERROR, f"顺序字段 {order_column} 无法排序") from exc
    else:
        warnings.append(
            WarningItem(code="ROW_ORDER_USED", message="未提供顺序字段，使用当前数据版本行顺序")
        )
    numeric = pd.to_numeric(ordered[value_column], errors="coerce")
    missing_count = int(numeric.isna().sum())
    if missing_count:
        valid_mask = numeric.notna()
        ordered = ordered.loc[valid_mask].copy()
        numeric = numeric.loc[valid_mask]
        warnings.append(
            WarningItem(
                code="MISSING_VALUES_DROPPED",
                message=f"滑窗前删除了 {missing_count} 条缺失记录",
            )
        )
    values = numeric.to_numpy(dtype=float)
    minimum_required = window_size + (forecast_horizon if output_mode == "supervised" else 0)
    if len(values) < minimum_required:
        raise ProtocolError(
            ErrorCode.DATA_ERROR,
            f"有效样本不足，至少需要 {minimum_required} 条记录",
        )
    rows: list[dict[str, object]] = []
    start = 0
    while start < len(values):
        end = start + window_size
        window = values[start:end]
        if len(window) < window_size and not include_incomplete:
            break
        padded = np.full(window_size, np.nan)
        padded[: len(window)] = window
        row: dict[str, object] = {
            "window_start": int(start),
            "window_end": int(min(end, len(values)) - 1),
        }
        for lag_index, value in enumerate(padded, start=1):
            row[f"{value_column}_lag_{lag_index}"] = value
        if order_column:
            row["order_start"] = ordered.iloc[start][order_column]
            row["order_end"] = ordered.iloc[min(end, len(values)) - 1][order_column]
        if output_mode == "supervised":
            target_index = end - 1 + forecast_horizon
            if target_index >= len(values):
                break
            row[f"{value_column}_target"] = values[target_index]
        rows.append(row)
        start += step_size
    if not rows:
        raise ProtocolError(ErrorCode.DATA_ERROR, "当前窗口和预测步数组合没有生成有效样本")
    output = pd.DataFrame(rows)
    return AlgorithmResult(
        data=output,
        metrics=[
            Metric(
                id="window_count", display_name="生成窗口数", value=len(output.index), unit="个"
            ),
            Metric(
                id="feature_count", display_name="输出字段数", value=len(output.columns), unit="个"
            ),
        ],
        tables=[table_from_frame("window_preview", "滑窗结果预览", output.head(20))],
        warnings=warnings,
        metadata={
            "method": "time_series_window.sliding",
            "rows_in": len(data.index),
            "rows_out": len(output.index),
            "window_size": window_size,
            "step_size": step_size,
            "forecast_horizon": forecast_horizon,
            "changed_columns": output.columns.tolist(),
        },
    )
