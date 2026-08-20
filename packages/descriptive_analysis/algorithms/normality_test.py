"""Shapiro、D'Agostino 和 Anderson 正态性检验。"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from pydantic import JsonValue
from scipy import stats

from algorithm_sdk import AlgorithmResult, Metric, WarningItem
from algorithm_sdk.algorithm_utils import get_float_parameter, get_slot, table_from_frame
from algorithm_sdk.errors import ErrorCode, ProtocolError
from algorithm_sdk.serialization import to_json_compatible


def run_auto(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """按样本数自动选择 Shapiro 或 D'Agostino K²。"""

    return _run_tests(data, slots, parameters, context, None)


def run_shapiro(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """执行 Shapiro-Wilk 检验。"""

    return _run_tests(data, slots, parameters, context, "shapiro")


def run_dagostino(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """执行 D'Agostino K² 检验。"""

    return _run_tests(data, slots, parameters, context, "dagostino")


def run_anderson(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """执行 Anderson-Darling 正态分布检验。"""

    return _run_tests(data, slots, parameters, context, "anderson")


def _run_tests(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
    fixed_method: str | None,
) -> AlgorithmResult:
    """完成每个字段的方法选择、前提检查和结果整理。"""

    del context
    alpha = get_float_parameter(parameters, "alpha", 0.05)
    columns = get_slot(slots, "x_columns")
    rows: list[dict[str, JsonValue]] = []
    warnings: list[WarningItem] = []
    for column in columns:
        valid = (
            pd.to_numeric(data[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        )
        sample_count = len(valid.index)
        if valid.nunique() <= 1:
            raise ProtocolError(ErrorCode.DATA_ERROR, f"字段 {column} 为常数列，无法检验正态性")
        method = fixed_method or ("shapiro" if sample_count < 5000 else "dagostino")
        if method == "shapiro":
            if sample_count < 3:
                raise ProtocolError(ErrorCode.DATA_ERROR, f"字段 {column} 至少需要 3 个有效样本")
            statistic, p_value = stats.shapiro(valid)
            reason = (
                "有效样本数小于 5000，自动选择 Shapiro-Wilk" if fixed_method is None else "用户指定"
            )
            if sample_count > 5000:
                warnings.append(
                    WarningItem(
                        code="SHAPIRO_LARGE_SAMPLE",
                        message=f"字段 {column} 超过 5000 个样本，Shapiro P 值可能不够准确",
                    )
                )
            rows.append(
                _p_value_row(column, method, sample_count, statistic, p_value, alpha, reason)
            )
        elif method == "dagostino":
            if sample_count < 8:
                raise ProtocolError(ErrorCode.DATA_ERROR, f"字段 {column} 至少需要 8 个有效样本")
            statistic, p_value = stats.normaltest(valid)
            reason = (
                "有效样本数较大，自动选择 D'Agostino K²" if fixed_method is None else "用户指定"
            )
            rows.append(
                _p_value_row(column, method, sample_count, statistic, p_value, alpha, reason)
            )
        else:
            if sample_count < 3:
                raise ProtocolError(ErrorCode.DATA_ERROR, f"字段 {column} 至少需要 3 个有效样本")
            try:
                result = stats.anderson(valid, dist="norm", method="interpolate")
                p_value = float(result.pvalue)
                critical_payload: JsonValue = None
                reject = p_value < alpha
            except TypeError:
                # SciPy 1.13—1.16 尚无 method 参数，使用其固定临界值表兼容执行。
                result = stats.anderson(valid, dist="norm")
                critical = {
                    str(level): float(value)
                    for level, value in zip(
                        result.significance_level, result.critical_values, strict=True
                    )
                }
                target_level = min(critical, key=lambda value: abs(float(value) / 100 - alpha))
                reject = float(result.statistic) > critical[target_level]
                p_value = None
                critical_payload = to_json_compatible(critical)
            rows.append(
                {
                    "field": column,
                    "method": "anderson",
                    "sample_count": sample_count,
                    "statistic": float(result.statistic),
                    "p_value": p_value,
                    "alpha": alpha,
                    "critical_values": critical_payload,
                    "conclusion": "拒绝正态性假设" if reject else "未拒绝正态性假设",
                    "selection_reason": "用户指定",
                }
            )
    return AlgorithmResult(
        metrics=[
            Metric(id="field_count", display_name="检验字段数", value=len(columns), unit="个")
        ],
        tables=[table_from_frame("normality_test", "正态性检验", pd.DataFrame(rows))],
        warnings=warnings,
        metadata={"method": fixed_method or "auto", "alpha": alpha},
    )


def _p_value_row(
    column: str,
    method: str,
    sample_count: int,
    statistic: float,
    p_value: float,
    alpha: float,
    reason: str,
) -> dict[str, JsonValue]:
    """构造使用 P 值的正态性检验结果。"""

    return {
        "field": column,
        "method": method,
        "sample_count": sample_count,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "alpha": alpha,
        "critical_values": None,
        "conclusion": "拒绝正态性假设" if p_value < alpha else "未拒绝正态性假设",
        "selection_reason": reason,
    }
