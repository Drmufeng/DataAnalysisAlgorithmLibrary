"""Cochran Q、泊松拟合优度和游程检验。"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd
from pydantic import JsonValue
from scipy import stats
from statsmodels.stats.contingency_tables import cochrans_q

from algorithm_sdk import AlgorithmResult, ChartSpec, Metric, WarningItem
from algorithm_sdk.algorithm_utils import (
    get_float_parameter,
    get_single_slot,
    get_slot,
    get_str_parameter,
    table_from_frame,
)
from algorithm_sdk.errors import ErrorCode, ProtocolError


def run_cochran_q(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """检验三个及以上相关二分类条件的阳性率是否相同。"""

    del context
    columns = get_slot(slots, "condition_columns")
    positive_value = parameters.get("positive_value", 1)
    alpha = get_float_parameter(parameters, "alpha", 0.05)
    frame = data[columns].dropna()
    binary = pd.DataFrame(index=frame.index)
    for column in columns:
        if frame[column].nunique() != 2:
            raise ProtocolError(ErrorCode.DATA_ERROR, f"字段 {column} 必须恰好包含两个有效类别")
        binary[column] = (frame[column].astype(str) == str(positive_value)).astype(int)
    if not bool((binary.sum(axis=0) > 0).any()):
        raise ProtocolError(ErrorCode.DATA_ERROR, "positive_value 未匹配任何有效值")
    result = cochrans_q(binary.to_numpy())
    rates = pd.DataFrame(
        {
            "condition": columns,
            "positive_count": binary.sum(axis=0).to_numpy(),
            "positive_rate": binary.mean(axis=0).to_numpy(),
        }
    )
    return AlgorithmResult(
        metrics=[
            Metric(id="q_statistic", display_name="Q 统计量", value=float(result.statistic)),
            Metric(id="p_value", display_name="P 值", value=float(result.pvalue)),
            Metric(
                id="sample_count", display_name="完整样本数", value=len(binary.index), unit="个"
            ),
        ],
        tables=[table_from_frame("condition_rates", "各条件阳性率", rates)],
        metadata={
            "method": "cochran_q.related_binary",
            "degrees_of_freedom": len(columns) - 1,
            "alpha": alpha,
        },
    )


def run_poisson(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """使用合并低期望频数后的卡方统计量检验泊松拟合。"""

    del context
    count_column = get_single_slot(slots, "count_column")
    weight_columns = get_slot(slots, "frequency_weight")
    lambda_mode = get_str_parameter(parameters, "lambda_mode", "estimate")
    min_expected = get_float_parameter(parameters, "min_expected_count", 5.0)
    alpha = get_float_parameter(parameters, "alpha", 0.05)
    columns = [count_column, *weight_columns]
    frame = data[columns].dropna()
    counts = pd.to_numeric(frame[count_column], errors="coerce")
    if counts.isna().any() or (counts < 0).any() or not bool(np.equal(counts % 1, 0).all()):
        raise ProtocolError(ErrorCode.DATA_ERROR, "计数字段必须是非负整数")
    weights = (
        pd.to_numeric(frame[weight_columns[0]], errors="coerce")
        if weight_columns
        else pd.Series(1.0, index=frame.index)
    )
    if weights.isna().any() or (weights < 0).any():
        raise ProtocolError(ErrorCode.DATA_ERROR, "频数权重必须是非负数")
    total = float(weights.sum())
    if total <= 0:
        raise ProtocolError(ErrorCode.DATA_ERROR, "总频数必须大于 0")
    if lambda_mode == "fixed":
        lambda_value = get_float_parameter(parameters, "lambda_value", 1.0)
        estimated_parameters = 0
    else:
        lambda_value = float(np.average(counts, weights=weights))
        estimated_parameters = 1
    if lambda_value <= 0:
        raise ProtocolError(ErrorCode.DATA_ERROR, "泊松分布参数 lambda 必须大于 0")
    max_count = max(int(counts.max()), int(math.ceil(lambda_value + 6 * math.sqrt(lambda_value))))
    observed = np.array([float(weights[counts == value].sum()) for value in range(max_count + 1)])
    expected = stats.poisson.pmf(np.arange(max_count + 1), lambda_value) * total
    expected[-1] += max(total - float(expected.sum()), 0.0)
    labels = [str(value) for value in range(max_count)] + [f">={max_count}"]
    while len(expected) > 2 and expected[-1] < min_expected:
        expected[-2] += expected[-1]
        observed[-2] += observed[-1]
        expected = expected[:-1]
        observed = observed[:-1]
        labels[-2] = f">={labels[-2]}"
        labels = labels[:-1]
    statistic = float(np.sum((observed - expected) ** 2 / expected))
    degrees = len(expected) - 1 - estimated_parameters
    if degrees <= 0:
        raise ProtocolError(ErrorCode.DATA_ERROR, "合并期望频数后自由度不足，无法执行卡方检验")
    p_value = float(stats.chi2.sf(statistic, degrees))
    table = pd.DataFrame({"count": labels, "observed": observed, "expected": expected})
    warnings = []
    if bool((expected < min_expected).any()):
        warnings.append(
            WarningItem(
                code="LOW_EXPECTED_COUNT", message="仍有期望频数低于设定阈值，请谨慎解释卡方近似"
            )
        )
    return AlgorithmResult(
        metrics=[
            Metric(id="lambda", display_name="Lambda", value=lambda_value),
            Metric(id="chi_square", display_name="卡方统计量", value=statistic),
            Metric(id="p_value", display_name="P 值", value=p_value),
        ],
        tables=[table_from_frame("poisson_fit", "泊松拟合频数", table)],
        charts=[
            ChartSpec(
                id="poisson_fit_chart",
                display_name="观测与期望频数",
                chart_type="grouped_bar",
                data={"categories": labels, "observed": observed, "expected": expected},
            )
        ],
        warnings=warnings,
        metadata={
            "method": "poisson_goodness_of_fit.chi_square",
            "degrees_of_freedom": degrees,
            "alpha": alpha,
        },
    )


def run_runs_median(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """以中位数为切点执行游程检验。"""

    return _run_runs(data, slots, parameters, context, "median_cut")


def run_runs_custom(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """以用户阈值为切点执行游程检验。"""

    return _run_runs(data, slots, parameters, context, "custom_cut")


def run_runs_binary(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """对原始二分类序列执行游程检验。"""

    return _run_runs(data, slots, parameters, context, "binary")


def _run_runs(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
    method: str,
) -> AlgorithmResult:
    """按明确顺序将每个字段转成二分类并计算正态近似。"""

    del context
    columns = get_slot(slots, "x_columns")
    order_columns = get_slot(slots, "order_column")
    ordered = data.sort_values(order_columns[0], kind="stable") if order_columns else data
    ties = get_str_parameter(parameters, "ties", "drop")
    alternative = get_str_parameter(parameters, "alternative", "two_sided")
    rows: list[dict[str, JsonValue]] = []
    warnings = []
    if not order_columns:
        warnings.append(
            WarningItem(code="ROW_ORDER_USED", message="未提供顺序字段，使用当前数据行顺序")
        )
    for column in columns:
        series = ordered[column].dropna()
        if method == "binary":
            categories = sorted(series.astype(str).unique())
            if len(categories) != 2:
                raise ProtocolError(ErrorCode.DATA_ERROR, f"字段 {column} 必须恰好包含两个类别")
            binary = (series.astype(str) == categories[1]).astype(int).to_numpy()
            cut_value: JsonValue = categories[0]
        else:
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            cut = (
                float(numeric.median())
                if method == "median_cut"
                else get_float_parameter(parameters, "cut_value", 0.0)
            )
            if ties == "drop":
                numeric = numeric[numeric != cut]
                binary = (numeric > cut).astype(int).to_numpy()
            elif ties == "lower":
                binary = (numeric > cut).astype(int).to_numpy()
            else:
                binary = (numeric >= cut).astype(int).to_numpy()
            cut_value = cut
        n1, n0 = int(np.sum(binary == 1)), int(np.sum(binary == 0))
        if n1 == 0 or n0 == 0:
            raise ProtocolError(ErrorCode.DATA_ERROR, f"字段 {column} 二分后必须同时包含两类")
        runs = int(1 + np.sum(binary[1:] != binary[:-1]))
        expected = 1 + 2 * n1 * n0 / (n1 + n0)
        variance = 2 * n1 * n0 * (2 * n1 * n0 - n1 - n0) / (((n1 + n0) ** 2) * (n1 + n0 - 1))
        z_value = (runs - expected) / math.sqrt(variance)
        if alternative == "greater":
            p_value = float(stats.norm.sf(z_value))
        elif alternative == "less":
            p_value = float(stats.norm.cdf(z_value))
        else:
            p_value = float(2 * stats.norm.sf(abs(z_value)))
        rows.append(
            {
                "field": column,
                "runs": runs,
                "class_0_count": n0,
                "class_1_count": n1,
                "z_statistic": z_value,
                "p_value": p_value,
                "cut_value": cut_value,
            }
        )
    return AlgorithmResult(
        metrics=[
            Metric(id="field_count", display_name="检验字段数", value=len(columns), unit="个")
        ],
        tables=[table_from_frame("runs_test", "游程检验", pd.DataFrame(rows))],
        warnings=warnings,
        metadata={"method": f"runs_test.{method}", "alternative": alternative},
    )
