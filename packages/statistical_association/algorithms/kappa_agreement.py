"""Cohen、加权 Cohen 和 Fleiss Kappa 一致性检验。"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd
from pydantic import JsonValue
from scipy import stats
from sklearn.metrics import cohen_kappa_score

from algorithm_sdk import AlgorithmResult, Metric, WarningItem
from algorithm_sdk.algorithm_utils import (
    field_category_order,
    get_float_parameter,
    get_slot,
    get_str_parameter,
    table_from_frame,
)
from algorithm_sdk.errors import ErrorCode, ProtocolError


def run_cohen(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """计算两位评价者的普通 Cohen Kappa。"""

    return _run(data, slots, parameters, context, "cohen")


def run_linear(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """计算两位评价者的线性加权 Kappa。"""

    return _run(data, slots, parameters, context, "linear_weighted")


def run_quadratic(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """计算两位评价者的平方加权 Kappa。"""

    return _run(data, slots, parameters, context, "quadratic_weighted")


def run_fleiss(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """计算三位及以上评价者的 Fleiss Kappa。"""

    return _run(data, slots, parameters, context, "fleiss")


def _run(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
    method: str,
) -> AlgorithmResult:
    """统一处理缺失、类别集合、Kappa 和近似区间。"""

    columns = get_slot(slots, "rater_columns")
    weight_columns = get_slot(slots, "frequency_weight")
    missing = get_str_parameter(parameters, "missing", "drop_row")
    confidence = get_float_parameter(parameters, "confidence_level", 0.95)
    frame = data[[*columns, *weight_columns]].copy()
    if missing == "treat_as_category":
        if weight_columns:
            frame = frame.dropna(subset=weight_columns)
        frame.loc[:, columns] = (
            frame[columns].astype(object).where(frame[columns].notna(), "（缺失）")
        )
    else:
        frame = frame.dropna()
    if len(frame.index) < 2:
        raise ProtocolError(ErrorCode.DATA_ERROR, "Kappa 至少需要两个完整评价对象")
    rater_frame = frame[columns].astype(str)
    sample_weights = (
        pd.to_numeric(frame[weight_columns[0]], errors="coerce").to_numpy(dtype=float)
        if weight_columns
        else np.ones(len(frame.index), dtype=float)
    )
    if (
        not bool(np.isfinite(sample_weights).all())
        or bool((sample_weights < 0).any())
        or float(sample_weights.sum()) <= 0
    ):
        raise ProtocolError(ErrorCode.DATA_ERROR, "频数权重必须是总和大于 0 的有限非负数")
    categories = (
        _ordered_categories(rater_frame, columns, context)
        if method in {"linear_weighted", "quadratic_weighted"}
        else sorted({str(value) for value in rater_frame.to_numpy().ravel()}, key=str)
    )
    if len(categories) < 2:
        raise ProtocolError(ErrorCode.DATA_ERROR, "评价结果至少需要两个类别")
    if method == "fleiss":
        if len(columns) < 3:
            raise ProtocolError(ErrorCode.COLUMN_ERROR, "Fleiss Kappa 至少需要三位评价者")
        count_table = np.zeros((len(frame.index), len(categories)), dtype=int)
        category_index = {category: index for index, category in enumerate(categories)}
        for row_index, values in enumerate(rater_frame.to_numpy()):
            for value in values:
                count_table[row_index, category_index[value]] += 1
        kappa = _fleiss_kappa_with_weights(count_table, sample_weights)
        agreement_table = pd.DataFrame(count_table, columns=categories)
        agreement_table.insert(0, "frequency_weight", sample_weights)
    else:
        if len(columns) != 2:
            raise ProtocolError(ErrorCode.COLUMN_ERROR, "Cohen 和加权 Kappa 必须恰好选择两位评价者")
        weights = (
            None
            if method == "cohen"
            else ("linear" if method == "linear_weighted" else "quadratic")
        )
        left = rater_frame[columns[0]]
        right = rater_frame[columns[1]]
        kappa = float(
            cohen_kappa_score(
                left,
                right,
                labels=categories,
                weights=weights,
                sample_weight=sample_weights,
            )
        )
        agreement_table = (
            pd.crosstab(
                left,
                right,
                values=sample_weights,
                aggfunc="sum",
                dropna=False,
            )
            .fillna(0)
            .reset_index()
        )
    effective_count = float(sample_weights.sum() ** 2 / np.sum(sample_weights**2))
    standard_error = math.sqrt(max(1 - kappa**2, 0.0) / max(effective_count - 1, 1))
    critical = float(stats.norm.ppf(1 - (1 - confidence) / 2))
    lower, upper = (
        max(-1.0, kappa - critical * standard_error),
        min(1.0, kappa + critical * standard_error),
    )
    p_value = float(2 * stats.norm.sf(abs(kappa / standard_error))) if standard_error > 0 else 0.0
    return AlgorithmResult(
        metrics=[
            Metric(id="kappa", display_name="Kappa", value=kappa),
            Metric(id="p_value", display_name="P 值", value=p_value),
            Metric(id="sample_count", display_name="评价对象数", value=len(frame.index), unit="个"),
        ],
        tables=[table_from_frame("agreement_table", "一致性表", agreement_table)],
        warnings=[
            WarningItem(
                code="APPROXIMATE_KAPPA_INTERVAL",
                message="Kappa 的标准误、P 值和置信区间为近似结果，应结合样本设计解释",
            )
        ],
        metadata={
            "method": f"kappa_agreement.{method}",
            "standard_error": standard_error,
            "effective_sample_count": effective_count,
            "frequency_weight_sum": float(sample_weights.sum()),
            "confidence_lower": lower,
            "confidence_upper": upper,
            "categories": categories,
        },
    )


def _ordered_categories(
    frame: pd.DataFrame,
    columns: list[str],
    context: Mapping[str, JsonValue],
) -> list[str]:
    """加权 Kappa 必须使用两位评价者共享的明确类别顺序。"""

    configured = [
        [str(value) for value in field_category_order(context, column)] for column in columns
    ]
    nonempty = [order for order in configured if order]
    if nonempty:
        order = nonempty[0]
        if any(candidate != order for candidate in nonempty[1:]):
            raise ProtocolError(ErrorCode.COLUMN_ERROR, "两位评价者的类别顺序设置不一致")
    else:
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        if not bool(numeric.notna().all().all()):
            raise ProtocolError(
                ErrorCode.COLUMN_ERROR,
                "线性或平方加权 Kappa 使用文本类别时，必须先在平台设置类别顺序",
            )
        actual_values = {str(value) for value in frame.to_numpy().ravel()}
        order = sorted(actual_values, key=float)
    actual = {str(value) for value in frame.to_numpy().ravel()}
    missing = sorted(actual - set(order))
    if missing:
        raise ProtocolError(
            ErrorCode.COLUMN_ERROR,
            f"类别顺序缺少实际取值：{', '.join(missing)}",
        )
    return order


def _fleiss_kappa_with_weights(counts: np.ndarray, weights: np.ndarray) -> float:
    """按评价对象频数权重计算 Fleiss Kappa。"""

    rater_count = int(counts.sum(axis=1)[0])
    if rater_count < 2 or not bool((counts.sum(axis=1) == rater_count).all()):
        raise ProtocolError(ErrorCode.DATA_ERROR, "每个评价对象必须具有相同数量的有效评价")
    total_weight = float(weights.sum())
    category_proportions = np.sum(counts * weights[:, None], axis=0) / (total_weight * rater_count)
    object_agreement = (np.sum(counts**2, axis=1) - rater_count) / (rater_count * (rater_count - 1))
    observed = float(np.average(object_agreement, weights=weights))
    expected = float(np.sum(category_proportions**2))
    if math.isclose(expected, 1.0):
        raise ProtocolError(ErrorCode.DATA_ERROR, "类别边际分布导致 Fleiss Kappa 无法计算")
    return (observed - expected) / (1 - expected)
