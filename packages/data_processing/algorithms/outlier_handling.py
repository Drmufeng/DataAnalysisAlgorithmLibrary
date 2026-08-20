"""多种异常值识别和可复现处置。"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from pydantic import JsonValue
from scipy import stats

from algorithm_sdk import AlgorithmResult, Metric, WarningItem
from algorithm_sdk.algorithm_utils import (
    get_dict_parameter,
    get_float_parameter,
    get_slot,
    get_str_parameter,
    random_seed_from_context,
    table_from_frame,
)
from algorithm_sdk.errors import ErrorCode, ProtocolError


def run_auto(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """按偏度为每个字段自动选择 IQR 或 MAD。"""

    return _run(data, slots, parameters, context, "auto")


def run_custom(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """按用户提供上下限识别异常值。"""

    return _run(data, slots, parameters, context, "custom")


def run_mad(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """使用稳健中位数绝对偏差识别异常值。"""

    return _run(data, slots, parameters, context, "mad")


def run_iqr(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """使用四分位距识别异常值。"""

    return _run(data, slots, parameters, context, "iqr")


def run_three_sigma(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """使用均值加减标准差倍数识别异常值。"""

    return _run(data, slots, parameters, context, "three_sigma")


def _run(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
    method: str,
) -> AlgorithmResult:
    """统一完成边界计算、异常掩码和处置。"""

    columns = get_slot(slots, "x_columns")
    action = get_str_parameter(parameters, "action", "set_null")
    output = data.copy(deep=True)
    rng = np.random.default_rng(random_seed_from_context(context))
    custom_bounds = get_dict_parameter(parameters, "bounds")
    bound_rows: list[dict[str, JsonValue]] = []
    warnings: list[WarningItem] = []
    total_outliers = 0
    for column in columns:
        numeric = pd.to_numeric(data[column], errors="coerce").astype(float)
        finite_mask = np.isfinite(numeric.to_numpy()) | numeric.isna().to_numpy()
        if not bool(finite_mask.all()):
            raise ProtocolError(ErrorCode.DATA_ERROR, f"字段 {column} 含有无穷值")
        valid = numeric.dropna()
        if valid.empty:
            raise ProtocolError(ErrorCode.DATA_ERROR, f"字段 {column} 全部为空")
        if valid.nunique() <= 1:
            raise ProtocolError(ErrorCode.DATA_ERROR, f"字段 {column} 为常数列")
        actual_method = method
        if method == "auto":
            actual_method = "mad" if abs(float(stats.skew(valid, bias=False))) > 1 else "iqr"
        lower, upper = _bounds_for_column(valid, column, actual_method, parameters, custom_bounds)
        outlier_mask = numeric.lt(lower) | numeric.gt(upper)
        outlier_count = int(outlier_mask.sum())
        total_outliers += outlier_count
        _apply_action(output, column, numeric, outlier_mask, lower, upper, action, parameters, rng)
        indices: list[JsonValue] = [str(value) for value in data.index[outlier_mask][:20].tolist()]
        bound_rows.append(
            {
                "field": column,
                "actual_method": actual_method,
                "lower_bound": lower,
                "upper_bound": upper,
                "outlier_count": outlier_count,
                "index_preview": indices,
            }
        )
        if method == "auto":
            warnings.append(
                WarningItem(
                    code="AUTO_OUTLIER_METHOD",
                    message=f"字段 {column} 自动选择 {actual_method.upper()} 识别异常值",
                )
            )
    return AlgorithmResult(
        data=output,
        metrics=[
            Metric(id="outlier_count", display_name="异常值数量", value=total_outliers, unit="个")
        ],
        tables=[table_from_frame("outlier_bounds", "异常值边界", pd.DataFrame(bound_rows))],
        warnings=warnings,
        metadata={
            "method": f"outlier_handling.{method}",
            "action": action,
            "rows_in": len(data.index),
            "rows_out": len(output.index),
            "changed_columns": columns,
            "random_seed": random_seed_from_context(context),
        },
    )


def _bounds_for_column(
    valid: pd.Series,
    column: str,
    method: str,
    parameters: Mapping[str, JsonValue],
    custom_bounds: Mapping[str, JsonValue],
) -> tuple[float, float]:
    """计算单字段识别上下界。"""

    if method == "custom":
        raw = custom_bounds.get(column)
        if not isinstance(raw, dict):
            raise ProtocolError(ErrorCode.PARAMETER_ERROR, f"字段 {column} 缺少自定义 bounds")
        lower_value = raw.get("minimum")
        upper_value = raw.get("maximum")
        lower = float(lower_value) if isinstance(lower_value, (int, float)) else float("-inf")
        upper = float(upper_value) if isinstance(upper_value, (int, float)) else float("inf")
        if lower >= upper:
            raise ProtocolError(ErrorCode.PARAMETER_ERROR, f"字段 {column} 的下限必须小于上限")
        return lower, upper
    if method == "mad":
        threshold = get_float_parameter(parameters, "threshold", 3.5)
        median = float(valid.median())
        mad = float(np.median(np.abs(valid.to_numpy() - median)))
        if mad == 0:
            raise ProtocolError(ErrorCode.DATA_ERROR, f"字段 {column} 的 MAD 为 0，无法使用 MAD")
        distance = threshold * mad / 0.6744897501960817
        return median - distance, median + distance
    if method == "three_sigma":
        k_value = get_float_parameter(parameters, "k", 3.0)
        mean = float(valid.mean())
        std = float(valid.std(ddof=1))
        return mean - k_value * std, mean + k_value * std
    k_value = get_float_parameter(parameters, "k", 1.5)
    q1, q3 = float(valid.quantile(0.25)), float(valid.quantile(0.75))
    iqr = q3 - q1
    return q1 - k_value * iqr, q3 + k_value * iqr


def _apply_action(
    output: pd.DataFrame,
    column: str,
    numeric: pd.Series,
    mask: pd.Series,
    lower: float,
    upper: float,
    action: str,
    parameters: Mapping[str, JsonValue],
    rng: np.random.Generator,
) -> None:
    """仅修改识别为异常的单元格。"""

    valid_values = numeric[~mask & numeric.notna()]
    if action == "winsorize":
        output.loc[mask, column] = numeric[mask].clip(lower=lower, upper=upper)
    elif action == "set_null":
        output.loc[mask, column] = np.nan
    elif action == "custom_value":
        value = parameters.get("replacement_value")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ProtocolError(ErrorCode.PARAMETER_ERROR, "replacement_value 必须是数值")
        output.loc[mask, column] = value
    elif action == "mean":
        output.loc[mask, column] = float(valid_values.mean())
    elif action == "median":
        output.loc[mask, column] = float(valid_values.median())
    elif action == "mode":
        mode = valid_values.mode()
        if mode.empty:
            raise ProtocolError(ErrorCode.DATA_ERROR, f"字段 {column} 没有可用众数")
        output.loc[mask, column] = float(mode.iloc[0])
    elif action == "zero":
        output.loc[mask, column] = 0
    elif action == "random_valid_value":
        if valid_values.empty:
            raise ProtocolError(ErrorCode.DATA_ERROR, f"字段 {column} 没有可抽样的正常值")
        output.loc[mask, column] = rng.choice(valid_values.to_numpy(), size=int(mask.sum()))
    else:
        raise ProtocolError(ErrorCode.PARAMETER_ERROR, f"不支持的异常值处置方式 {action}")
