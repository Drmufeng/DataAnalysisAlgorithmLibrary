"""周期选样、分块汇总和滤波后降采样。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np
import pandas as pd
from pydantic import JsonValue
from scipy import signal

from algorithm_sdk import AlgorithmResult, Metric, WarningItem
from algorithm_sdk.algorithm_utils import (
    get_bool_parameter,
    get_int_parameter,
    get_slot,
    get_str_parameter,
    table_from_frame,
)
from algorithm_sdk.errors import ErrorCode, ProtocolError


def run_periodic_sample(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """按稳定顺序每隔固定数量保留一行。"""

    del context
    ordered, warnings = _ordered_data(data, slots)
    factor = get_int_parameter(parameters, "factor", 2)
    offset = get_int_parameter(parameters, "offset", 0)
    if offset >= factor:
        raise ProtocolError(ErrorCode.PARAMETER_ERROR, "offset 必须小于 factor")
    output = ordered.iloc[offset::factor].copy()
    return _result(output, data, "data_downsampling.periodic_sample", factor, warnings)


def run_block_aggregate(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """按连续样本块汇总定量字段。"""

    del context
    ordered, warnings = _ordered_data(data, slots)
    value_columns = get_slot(slots, "value_columns")
    order_columns = get_slot(slots, "order_column")
    factor = get_int_parameter(parameters, "factor", 2)
    aggregation = get_str_parameter(parameters, "aggregation", "mean")
    other_policy = get_str_parameter(parameters, "other_columns", "drop")
    group_key = np.arange(len(ordered.index)) // factor
    output_values = (
        ordered[value_columns].groupby(group_key).agg(aggregation).reset_index(drop=True)
    )
    result_parts = [output_values]
    if order_columns:
        order_aggregation = "first" if other_policy != "last" else "last"
        order_values = (
            ordered[order_columns].groupby(group_key).agg(order_aggregation).reset_index(drop=True)
        )
        result_parts.insert(0, order_values)
    if other_policy in {"first", "last"}:
        excluded = set(value_columns + order_columns)
        other_columns = [column for column in ordered.columns if column not in excluded]
        if other_columns:
            others = (
                ordered[other_columns].groupby(group_key).agg(other_policy).reset_index(drop=True)
            )
            result_parts.append(others)
    output = pd.concat(result_parts, axis=1)
    return _result(output, data, "data_downsampling.block_aggregate", factor, warnings)


def run_filtered_decimate(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """低通滤波后按整数因子降低等间隔信号采样率。"""

    del context
    ordered, warnings = _ordered_data(data, slots, require_order=True)
    value_columns = get_slot(slots, "value_columns")
    order_column = get_slot(slots, "order_column")[0]
    factor = get_int_parameter(parameters, "factor", 2)
    filter_type = get_str_parameter(parameters, "filter_type", "iir")
    zero_phase = get_bool_parameter(parameters, "zero_phase", True)
    order_values = ordered[order_column]
    _ensure_equal_interval(order_values)
    numeric = ordered[value_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not bool(np.isfinite(numeric.to_numpy(dtype=float)).all()):
        raise ProtocolError(ErrorCode.DATA_ERROR, "滤波后降采样不允许缺失值或无穷值")
    try:
        decimated = {
            column: signal.decimate(
                numeric[column].to_numpy(dtype=float),
                factor,
                ftype=filter_type,
                zero_phase=zero_phase,
            )
            for column in value_columns
        }
    except ValueError as exc:
        raise ProtocolError(ErrorCode.DATA_ERROR, "样本数不足以执行当前滤波降采样参数") from exc
    output_length = len(next(iter(decimated.values())))
    sampled_order = order_values.iloc[::factor].head(output_length).reset_index(drop=True)
    output = pd.DataFrame({order_column: sampled_order, **decimated})
    return _result(output, data, "data_downsampling.filtered_decimate", factor, warnings)


def _ordered_data(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    *,
    require_order: bool = False,
) -> tuple[pd.DataFrame, list[WarningItem]]:
    """按可选顺序字段稳定排序。"""

    order_columns = get_slot(slots, "order_column")
    if require_order and not order_columns:
        raise ProtocolError(ErrorCode.COLUMN_ERROR, "滤波后降采样必须提供顺序或时间字段")
    if order_columns:
        try:
            return data.sort_values(order_columns[0], kind="stable").copy(), []
        except TypeError as exc:
            raise ProtocolError(ErrorCode.DATA_ERROR, "顺序字段无法排序") from exc
    return data.copy(), [
        WarningItem(code="ROW_ORDER_USED", message="未提供顺序字段，使用当前行顺序")
    ]


def _ensure_equal_interval(series: pd.Series) -> None:
    """验证数值或时间顺序字段近似等间隔。"""

    differences = series.diff().dropna()
    if len(differences.index) < 2:
        raise ProtocolError(ErrorCode.DATA_ERROR, "等间隔信号至少需要 3 个时间点")
    if pd.api.types.is_timedelta64_dtype(differences.dtype):
        timedelta_values = cast(list[pd.Timedelta], differences.tolist())
        numeric = np.asarray(
            [item.total_seconds() for item in timedelta_values],
            dtype=np.float64,
        )
    else:
        numeric = pd.to_numeric(differences, errors="coerce").to_numpy(dtype=float)
    if not bool(np.isfinite(numeric).all()) or not bool(
        np.allclose(numeric, numeric[0], rtol=1e-5, atol=1e-8)
    ):
        raise ProtocolError(ErrorCode.DATA_ERROR, "顺序字段不是近似等间隔，不能滤波后降采样")


def _result(
    output: pd.DataFrame,
    input_data: pd.DataFrame,
    method: str,
    factor: int,
    warnings: list[WarningItem],
) -> AlgorithmResult:
    """构造三种降采样方法共享的结果。"""

    return AlgorithmResult(
        data=output,
        metrics=[
            Metric(id="rows_in", display_name="处理前行数", value=len(input_data.index), unit="行"),
            Metric(id="rows_out", display_name="处理后行数", value=len(output.index), unit="行"),
        ],
        tables=[table_from_frame("downsample_preview", "降采样结果预览", output.head(20))],
        warnings=warnings,
        metadata={
            "method": method,
            "factor": factor,
            "rows_in": len(input_data.index),
            "rows_out": len(output.index),
            "changed_columns": output.columns.tolist(),
        },
    )
