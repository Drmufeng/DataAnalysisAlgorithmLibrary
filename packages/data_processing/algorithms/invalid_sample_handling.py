"""通用表格无效样本规则检查、标记和删除。"""

from __future__ import annotations

import re
from collections.abc import Mapping
from operator import eq, ge, gt, le, lt, ne

import pandas as pd
from pydantic import JsonValue

from algorithm_sdk import AlgorithmResult, Metric, WarningItem
from algorithm_sdk.algorithm_utils import (
    get_float_parameter,
    get_list_parameter,
    get_slot,
    get_str_parameter,
    table_from_frame,
    unique_column_name,
)
from algorithm_sdk.errors import ErrorCode, ProtocolError


def run_rule_based(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """执行一个界面预设的通用有效性规则。"""

    rules = get_list_parameter(parameters, "rules")
    if len(rules) != 1:
        raise ProtocolError(ErrorCode.PARAMETER_ERROR, "rule_based 必须且只能提供一条规则")
    return _run(data, slots, parameters, context, rules, "rule_based")


def run_custom_rules(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """组合多条受限规则，不执行任意 Python 表达式。"""

    rules = get_list_parameter(parameters, "rules")
    if not rules:
        raise ProtocolError(ErrorCode.PARAMETER_ERROR, "custom_rules 至少需要一条规则")
    return _run(data, slots, parameters, context, rules, "custom_rules")


def _run(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
    rules: list[JsonValue],
    method: str,
) -> AlgorithmResult:
    """计算各规则命中掩码并按组合策略处理。"""

    del context
    check_columns = get_slot(slots, "check_columns")
    combination = get_str_parameter(parameters, "combination", "any")
    action = get_str_parameter(parameters, "action", "flag")
    masks: list[pd.Series] = []
    detail_rows: list[dict[str, JsonValue]] = []
    weights: list[float] = []
    for index, raw_rule in enumerate(rules):
        if not isinstance(raw_rule, dict):
            raise ProtocolError(ErrorCode.PARAMETER_ERROR, f"第 {index + 1} 条规则必须是对象")
        mask = _evaluate_rule(data, check_columns, raw_rule)
        masks.append(mask)
        weight_raw = raw_rule.get("weight", 1.0)
        weight = float(weight_raw) if isinstance(weight_raw, (int, float)) else 1.0
        weights.append(weight)
        detail_rows.append(
            {
                "rule_index": index + 1,
                "rule_type": raw_rule.get("type", "unknown"),
                "matched_rows": int(mask.sum()),
                "weight": weight,
            }
        )
    combined = _combine_masks(masks, weights, combination, parameters)
    output = data.copy(deep=True)
    if action == "remove_rows":
        output = output.loc[~combined].copy()
        changed_columns: list[str] = []
    else:
        preferred = get_str_parameter(parameters, "flag_column_name", "invalid_sample")
        flag_column = unique_column_name(output, preferred)
        output[flag_column] = combined.astype(bool)
        changed_columns = [flag_column]
    invalid_count = int(combined.sum())
    return AlgorithmResult(
        data=output,
        metrics=[
            Metric(
                id="invalid_sample_count", display_name="无效样本数", value=invalid_count, unit="行"
            ),
            Metric(
                id="valid_sample_count",
                display_name="有效样本数",
                value=len(data.index) - invalid_count,
                unit="行",
            ),
        ],
        tables=[table_from_frame("rule_details", "规则命中明细", pd.DataFrame(detail_rows))],
        warnings=[
            WarningItem(
                code="BUSINESS_RULE_CONFIRMATION",
                message="无效样本取决于业务规则，请在删除前核对规则定义",
            )
        ],
        metadata={
            "method": f"invalid_sample_handling.{method}",
            "action": action,
            "combination": combination,
            "rows_in": len(data.index),
            "rows_out": len(output.index),
            "changed_columns": changed_columns,
        },
    )


def _evaluate_rule(
    data: pd.DataFrame,
    default_columns: list[str],
    rule: Mapping[str, JsonValue],
) -> pd.Series:
    """执行一条白名单规则并返回命中行为 True 的布尔掩码。"""

    rule_type = rule.get("type")
    raw_columns = rule.get("columns", default_columns)
    columns = (
        [value for value in raw_columns if isinstance(value, str)]
        if isinstance(raw_columns, list)
        else default_columns
    )
    if not columns or any(column not in data.columns for column in columns):
        raise ProtocolError(ErrorCode.COLUMN_ERROR, "规则引用了不存在的字段")
    if rule_type == "missing_ratio":
        threshold = _numeric_rule_value(rule, "threshold", 0.5)
        return data[columns].isna().mean(axis=1) > threshold
    if rule_type == "value_range":
        column = columns[0]
        numeric = pd.to_numeric(data[column], errors="coerce")
        lower = rule.get("minimum")
        upper = rule.get("maximum")
        mask = pd.Series(False, index=data.index)
        if isinstance(lower, (int, float)):
            mask |= numeric < float(lower)
        if isinstance(upper, (int, float)):
            mask |= numeric > float(upper)
        return mask | (numeric.isna() & data[column].notna())
    if rule_type == "allowed_values":
        allowed = rule.get("values", [])
        if not isinstance(allowed, list):
            raise ProtocolError(ErrorCode.PARAMETER_ERROR, "allowed_values.values 必须是数组")
        return ~data[columns[0]].isin(allowed) & data[columns[0]].notna()
    if rule_type == "data_type":
        expected = rule.get("expected", "number")
        converted = pd.to_numeric(data[columns[0]], errors="coerce")
        return (
            converted.isna() & data[columns[0]].notna()
            if expected == "number"
            else pd.Series(False, index=data.index)
        )
    if rule_type == "unique_key":
        return data.duplicated(subset=columns, keep=False)
    if rule_type == "duplicate_row":
        return data.duplicated(subset=columns, keep=False)
    if rule_type == "cross_field_rule":
        if len(columns) != 2:
            raise ProtocolError(ErrorCode.COLUMN_ERROR, "cross_field_rule 必须指定两个字段")
        operator_name = rule.get("operator", "le")
        operators = {"lt": lt, "le": le, "gt": gt, "ge": ge, "eq": eq, "ne": ne}
        operation = operators.get(str(operator_name))
        if operation is None:
            raise ProtocolError(ErrorCode.PARAMETER_ERROR, "不支持的跨字段比较符")
        valid_relation = pd.Series(
            operation(data[columns[0]], data[columns[1]]),
            index=data.index,
            dtype=bool,
        )
        return (~valid_relation) & data[columns].notna().all(axis=1)
    if rule_type == "pattern":
        pattern_value = rule.get("pattern")
        if not isinstance(pattern_value, str) or len(pattern_value) > 256:
            raise ProtocolError(
                ErrorCode.PARAMETER_ERROR, "pattern 必须是不超过 256 字符的正则表达式"
            )
        try:
            pattern = re.compile(pattern_value)
        except re.error as exc:
            raise ProtocolError(ErrorCode.PARAMETER_ERROR, "pattern 不是合法正则表达式") from exc
        series = data[columns[0]].astype("string")
        matched = series.str.fullmatch(pattern, na=True)
        return pd.Series(~matched, index=data.index, dtype=bool)
    raise ProtocolError(ErrorCode.PARAMETER_ERROR, f"不支持的规则类型 {rule_type}")


def _combine_masks(
    masks: list[pd.Series],
    weights: list[float],
    combination: str,
    parameters: Mapping[str, JsonValue],
) -> pd.Series:
    """按 any、all 或加权阈值组合规则。"""

    if combination == "all":
        return pd.concat(masks, axis=1).all(axis=1)
    if combination == "weighted_score":
        threshold = get_float_parameter(parameters, "score_threshold", 1.0)
        score = pd.Series(0.0, index=masks[0].index)
        for mask, weight in zip(masks, weights, strict=True):
            score += mask.astype(float) * weight
        return score >= threshold
    return pd.concat(masks, axis=1).any(axis=1)


def _numeric_rule_value(rule: Mapping[str, JsonValue], name: str, default: float) -> float:
    """读取规则内的数值。"""

    value = rule.get(name, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ProtocolError(ErrorCode.PARAMETER_ERROR, f"规则参数 {name} 必须是数值")
    return float(value)
