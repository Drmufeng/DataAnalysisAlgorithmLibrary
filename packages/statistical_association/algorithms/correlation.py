"""Pearson、Spearman 和 Kendall tau-b 成对相关分析。"""

from __future__ import annotations

import math
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
    get_float_parameter,
    get_slot,
    get_str_parameter,
    table_from_frame,
)
from algorithm_sdk.errors import ErrorCode, ProtocolError


def run_pearson(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """计算 Pearson 相关、P 值、样本数和 Fisher 置信区间。"""

    return _run(data, slots, parameters, context, "pearson")


def run_spearman(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """计算带并列秩的 Spearman 相关。"""

    for column in get_slot(slots, "x_columns"):
        if (
            field_analysis_type(context, column) == "categorical"
            and field_measurement_level(context, column) != "ordinal"
        ):
            raise ProtocolError(
                ErrorCode.COLUMN_ERROR, f"定类字段 {column} 必须标记为有序才能计算 Spearman"
            )
    return _run(data, slots, parameters, context, "spearman")


def run_kendall(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """计算对并列秩修正的 Kendall tau-b。"""

    for column in get_slot(slots, "x_columns"):
        if (
            field_analysis_type(context, column) == "categorical"
            and field_measurement_level(context, column) != "ordinal"
        ):
            raise ProtocolError(
                ErrorCode.COLUMN_ERROR, f"定类字段 {column} 必须标记为有序才能计算 Kendall tau-b"
            )
    return _run(data, slots, parameters, context, "kendall")


def _run(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
    method: str,
) -> AlgorithmResult:
    """按 pairwise 或 listwise 缺失策略生成成对结果矩阵。"""

    columns = get_slot(slots, "x_columns")
    missing = get_str_parameter(parameters, "missing", "pairwise")
    alternative = get_str_parameter(parameters, "alternative", "two_sided").replace("_", "-")
    confidence = get_float_parameter(parameters, "confidence_level", 0.95)
    listwise_data = data[columns].dropna() if missing == "listwise" else None
    rows: list[dict[str, JsonValue]] = []
    warnings = [
        WarningItem(
            code="CORRELATION_NOT_CAUSATION",
            message="相关关系不能单独证明因果关系",
        )
    ]
    for left_index, left in enumerate(columns):
        for right in columns[left_index:]:
            if left == right:
                source = listwise_data[left] if listwise_data is not None else data[left].dropna()
                x = _analysis_values(source, left, context, method)
                y = x.copy()
            else:
                pair = (
                    listwise_data[[left, right]]
                    if listwise_data is not None
                    else data[[left, right]].dropna()
                )
                x = _analysis_values(pair[left], left, context, method)
                y = _analysis_values(pair[right], right, context, method)
            valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
            x, y = x[valid].astype(float), y[valid].astype(float)
            sample_count = len(x.index)
            if sample_count < 3:
                raise ProtocolError(
                    ErrorCode.DATA_ERROR, f"字段 {left} 与 {right} 的有效配对样本不足 3 个"
                )
            if x.nunique() <= 1 or y.nunique() <= 1:
                raise ProtocolError(ErrorCode.DATA_ERROR, f"字段 {left} 或 {right} 为常数列")
            if method == "pearson":
                statistic, p_value = stats.pearsonr(x, y, alternative=alternative)
                lower, upper = _fisher_interval(float(statistic), sample_count, confidence)
                tie_pairs: int | None = None
                concordant: int | None = None
                discordant: int | None = None
            elif method == "spearman":
                statistic, p_value = stats.spearmanr(x, y, alternative=alternative)
                lower, upper = None, None
                tie_pairs = int(x.duplicated().sum() + y.duplicated().sum())
                concordant, discordant = None, None
            else:
                statistic, p_value = stats.kendalltau(x, y, variant="b", alternative=alternative)
                lower, upper = None, None
                concordant, discordant = _pair_direction_counts(x.to_numpy(), y.to_numpy())
                tie_pairs = None
            rows.append(
                {
                    "variable_1": left,
                    "variable_2": right,
                    "coefficient": float(statistic),
                    "p_value": float(p_value),
                    "sample_count": sample_count,
                    "confidence_lower": lower,
                    "confidence_upper": upper,
                    "tie_summary": tie_pairs,
                    "concordant_pairs": concordant,
                    "discordant_pairs": discordant,
                }
            )
    result_frame = pd.DataFrame(rows)
    return AlgorithmResult(
        metrics=[Metric(id="pair_count", display_name="变量配对数", value=len(rows), unit="对")],
        tables=[
            table_from_frame(f"{method}_correlation", f"{method.title()} 相关结果", result_frame)
        ],
        charts=[
            ChartSpec(
                id="correlation_matrix",
                display_name="相关矩阵",
                chart_type="matrix",
                data={"method": method, "values": rows},
            )
        ],
        warnings=warnings,
        metadata={"method": method, "missing": missing, "alternative": alternative},
    )


def _analysis_values(
    series: pd.Series,
    column: str,
    context: Mapping[str, JsonValue],
    method: str,
) -> pd.Series:
    """将定量或允许的定类字段转换为有明确含义的数值序列。"""

    if field_analysis_type(context, column) != "categorical":
        return pd.to_numeric(series, errors="coerce")

    level = field_measurement_level(context, column)
    actual_categories = sorted(series.dropna().astype(str).unique().tolist())
    if level != "ordinal":
        if method != "pearson" or len(actual_categories) != 2:
            raise ProtocolError(
                ErrorCode.COLUMN_ERROR,
                f"定类字段 {column} 只有二分类时才能用于 Pearson；有序相关请先设置为有序定类",
            )
        category_order = actual_categories
    else:
        configured_order = [str(value) for value in field_category_order(context, column)]
        if configured_order:
            if len(configured_order) != len(set(configured_order)):
                raise ProtocolError(ErrorCode.COLUMN_ERROR, f"字段 {column} 的类别顺序存在重复值")
            missing_categories = sorted(set(actual_categories) - set(configured_order))
            if missing_categories:
                raise ProtocolError(
                    ErrorCode.COLUMN_ERROR,
                    f"字段 {column} 的类别顺序缺少取值：{', '.join(missing_categories)}",
                )
            category_order = configured_order
        else:
            numeric = pd.to_numeric(series, errors="coerce")
            if bool(numeric[series.notna()].notna().all()):
                return numeric
            raise ProtocolError(
                ErrorCode.COLUMN_ERROR,
                f"有序定类字段 {column} 使用文本类别，必须先在平台设置类别顺序",
            )
    mapping = {category: index for index, category in enumerate(category_order)}
    return series.astype("string").map(mapping).astype(float)


def _fisher_interval(
    coefficient: float, sample_count: int, confidence: float
) -> tuple[float | None, float | None]:
    """使用 Fisher z 变换计算 Pearson 相关系数置信区间。"""

    if sample_count <= 3 or abs(coefficient) >= 1:
        return None, None
    z_value = np.arctanh(coefficient)
    critical = stats.norm.ppf(1 - (1 - confidence) / 2)
    error = critical / math.sqrt(sample_count - 3)
    return float(np.tanh(z_value - error)), float(np.tanh(z_value + error))


def _pair_direction_counts(x: np.ndarray, y: np.ndarray) -> tuple[int, int]:
    """统计 Kendall 报告使用的一致对和不一致对。"""

    concordant = 0
    discordant = 0
    for index in range(len(x) - 1):
        products = (x[index] - x[index + 1 :]) * (y[index] - y[index + 1 :])
        concordant += int(np.sum(products > 0))
        discordant += int(np.sum(products < 0))
    return concordant, discordant
