"""根据变量组合选择合适的相关或关联统计量。"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from pydantic import JsonValue
from scipy import stats

from algorithm_sdk import AlgorithmResult, ChartSpec, Metric, WarningItem
from algorithm_sdk.algorithm_utils import (
    field_analysis_type,
    field_category_order,
    field_measurement_level,
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
    """逐对选择统计方法，并记录选择原因。"""

    columns = get_slot(slots, "analysis_columns")
    preference = get_str_parameter(parameters, "method_preference", "auto")
    missing = get_str_parameter(parameters, "missing", "pairwise")
    listwise = data[columns].dropna() if missing == "listwise" else None
    rows: list[dict[str, JsonValue]] = []
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1 :]:
            pair = listwise[[left, right]] if listwise is not None else data[[left, right]].dropna()
            if len(pair.index) < 3:
                raise ProtocolError(ErrorCode.DATA_ERROR, f"字段 {left} 与 {right} 的有效样本不足")
            method, statistic, p_value, reason = _solve_pair(
                pair,
                left,
                right,
                context,
                preference,
            )
            rows.append(
                {
                    "variable_1": left,
                    "variable_2": right,
                    "method": method,
                    "statistic": statistic,
                    "p_value": p_value,
                    "sample_count": len(pair.index),
                    "selection_reason": reason,
                }
            )
    return AlgorithmResult(
        metrics=[
            Metric(id="pair_count", display_name="完成分析的变量对", value=len(rows), unit="对")
        ],
        tables=[table_from_frame("association_auto", "自动关联分析", pd.DataFrame(rows))],
        charts=[
            ChartSpec(
                id="association_matrix",
                display_name="关联统计量矩阵",
                chart_type="method_matrix",
                data={"values": rows},
            )
        ],
        warnings=[
            WarningItem(
                code="MIXED_STATISTICS",
                message="矩阵中可能包含不同统计量，数值大小不能直接视为同一尺度比较",
            )
        ],
        metadata={"method": "association_auto_solver.auto", "preference": preference},
    )


def _solve_pair(
    pair: pd.DataFrame,
    left: str,
    right: str,
    context: Mapping[str, JsonValue],
    preference: str,
) -> tuple[str, float, float | None, str]:
    """返回方法、统计量、P 值和普通语言选择原因。"""

    left_type = field_analysis_type(context, left)
    right_type = field_analysis_type(context, right)
    if left_type == right_type == "quantitative":
        x = pd.to_numeric(pair[left], errors="coerce")
        y = pd.to_numeric(pair[right], errors="coerce")
        if preference == "nonparametric":
            statistic, p_value = stats.spearmanr(x, y)
            return "spearman", float(statistic), float(p_value), "用户偏好非参数方法"
        statistic, p_value = stats.pearsonr(x, y)
        return "pearson", float(statistic), float(p_value), "两个字段均为定量变量"
    if left_type == right_type == "categorical":
        if (
            field_measurement_level(context, left)
            == field_measurement_level(context, right)
            == "ordinal"
        ):
            x_codes = _ordinal_codes(pair[left], left, context)
            y_codes = _ordinal_codes(pair[right], right, context)
            statistic, p_value = stats.kendalltau(x_codes, y_codes, variant="b")
            return "kendall_tau_b", float(statistic), float(p_value), "两个字段均为有序定类"
        table = pd.crosstab(pair[left], pair[right])
        chi_square, p_value, _, _ = stats.chi2_contingency(table)
        denominator = len(pair.index) * max(min(table.shape) - 1, 1)
        cramer_v = float(np.sqrt(chi_square / denominator))
        return "cramers_v", cramer_v, float(p_value), "两个字段均为无序定类"
    categorical = left if left_type == "categorical" else right
    quantitative = right if categorical == left else left
    categories = pair[categorical].dropna().unique()
    numeric = pd.to_numeric(pair[quantitative], errors="coerce")
    if len(categories) == 2:
        binary = (pair[categorical] == sorted(categories, key=str)[1]).astype(int)
        statistic, p_value = stats.pointbiserialr(binary, numeric)
        return "point_biserial", float(statistic), float(p_value), "二分类变量与定量变量组合"
    grand_mean = float(numeric.mean())
    total_sum = float(np.sum((numeric - grand_mean) ** 2))
    between = 0.0
    for category in categories:
        values = numeric[pair[categorical] == category]
        between += len(values.index) * (float(values.mean()) - grand_mean) ** 2
    eta_squared = between / total_sum if total_sum else 0.0
    groups = [numeric[pair[categorical] == category].dropna() for category in categories]
    _, p_value = stats.f_oneway(*groups)
    return "eta_squared", eta_squared, float(p_value), "多分类变量与定量变量组合"


def _ordinal_codes(
    series: pd.Series,
    column: str,
    context: Mapping[str, JsonValue],
) -> np.ndarray:
    """按平台保存的类别顺序编码有序定类字段。"""

    order = [str(value) for value in field_category_order(context, column)]
    actual = series.astype(str)
    if not order:
        numeric = pd.to_numeric(series, errors="coerce")
        if bool(numeric.notna().all()):
            return numeric.to_numpy(dtype=float)
        raise ProtocolError(
            ErrorCode.COLUMN_ERROR,
            f"有序定类字段 {column} 使用文本类别，必须先在平台设置类别顺序",
        )
    missing = sorted(set(actual.unique()) - set(order))
    if missing:
        raise ProtocolError(
            ErrorCode.COLUMN_ERROR,
            f"字段 {column} 的类别顺序缺少取值：{', '.join(missing)}",
        )
    mapping = {value: index for index, value in enumerate(order)}
    return actual.map(mapping).to_numpy(dtype=float)
