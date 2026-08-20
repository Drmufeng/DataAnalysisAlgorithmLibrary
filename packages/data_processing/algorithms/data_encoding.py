"""手工映射、范围编码和自动分箱。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from numbers import Real

import pandas as pd
from pydantic import JsonValue

from algorithm_sdk import AlgorithmResult, Metric, WarningItem
from algorithm_sdk.algorithm_utils import (
    get_bool_parameter,
    get_dict_parameter,
    get_int_parameter,
    get_list_parameter,
    get_single_slot,
    get_str_parameter,
    table_from_frame,
    unique_column_name,
)
from algorithm_sdk.errors import ErrorCode, ProtocolError
from algorithm_sdk.serialization import to_json_compatible


def run_manual(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """按明确的原值到新值映射进行编码。"""

    del context
    mapping = get_dict_parameter(parameters, "mapping")
    if not mapping:
        raise ProtocolError(ErrorCode.PARAMETER_ERROR, "manual 编码必须提供非空 mapping")

    def encoder(value: object) -> tuple[bool, object]:
        key = str(value)
        return (key in mapping, mapping.get(key))

    return _apply_encoding(data, slots, parameters, encoder, "data_encoding.manual", mapping)


def run_range(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """按互不重叠的数值范围编码。"""

    del context
    raw_ranges = get_list_parameter(parameters, "ranges")
    ranges: list[tuple[float | None, float | None, JsonValue]] = []
    for raw in raw_ranges:
        if not isinstance(raw, dict) or "value" not in raw:
            raise ProtocolError(ErrorCode.PARAMETER_ERROR, "ranges 每项必须包含 value")
        lower_raw = raw.get("minimum")
        upper_raw = raw.get("maximum")
        lower = float(lower_raw) if isinstance(lower_raw, (int, float)) else None
        upper = float(upper_raw) if isinstance(upper_raw, (int, float)) else None
        if lower is None and upper is None:
            raise ProtocolError(ErrorCode.PARAMETER_ERROR, "范围至少需要 minimum 或 maximum")
        if lower is not None and upper is not None and lower >= upper:
            raise ProtocolError(ErrorCode.PARAMETER_ERROR, "范围 minimum 必须小于 maximum")
        ranges.append((lower, upper, raw["value"]))
    _ensure_non_overlapping_ranges(ranges)

    def encoder(value: object) -> tuple[bool, object]:
        if not isinstance(value, Real) or isinstance(value, bool):
            return False, None
        number = float(value)
        for lower, upper, target in ranges:
            if (lower is None or number >= lower) and (upper is None or number < upper):
                return True, target
        return False, None

    mapping_description = [
        {"minimum": lower, "maximum": upper, "value": target} for lower, upper, target in ranges
    ]
    return _apply_encoding(
        data,
        slots,
        parameters,
        encoder,
        "data_encoding.range",
        {"ranges": mapping_description},
    )


def run_auto_bin(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """使用等宽或等频方式自动分组。"""

    del context
    source = get_single_slot(slots, "source_column")
    bin_method = get_str_parameter(parameters, "bin_method", "equal_width")
    bin_count = get_int_parameter(parameters, "bin_count", 5)
    numeric = pd.to_numeric(data[source], errors="coerce")
    try:
        if bin_method == "equal_frequency":
            codes, edges = pd.qcut(
                numeric, q=bin_count, labels=False, retbins=True, duplicates="drop"
            )
        else:
            codes, edges = pd.cut(
                numeric,
                bins=bin_count,
                labels=False,
                retbins=True,
                include_lowest=True,
                duplicates="drop",
            )
    except ValueError as exc:
        raise ProtocolError(ErrorCode.DATA_ERROR, "当前字段无法按指定数量自动分组") from exc
    output = data.copy(deep=True)
    target_column = _target_column(output, source, parameters)
    encoded = codes.add(1).astype("Int64")
    output[target_column] = encoded
    changed = int(encoded.notna().sum())
    edge_frame = pd.DataFrame(
        {
            "group": range(1, len(edges)),
            "minimum": edges[:-1],
            "maximum": edges[1:],
        }
    )
    return AlgorithmResult(
        data=output,
        metrics=[
            Metric(id="changed_count", display_name="已编码数量", value=changed, unit="个"),
            Metric(id="group_count", display_name="实际分组数", value=len(edges) - 1, unit="组"),
        ],
        tables=[table_from_frame("bin_mapping", "自动分组边界", edge_frame)],
        metadata={
            "method": "data_encoding.auto_bin",
            "source_column": source,
            "output_column": target_column,
            "rows_in": len(data.index),
            "rows_out": len(output.index),
            "changed_columns": [target_column],
        },
    )


def _apply_encoding(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    encoder: Callable[[object], tuple[bool, object]],
    method: str,
    mapping_description: Mapping[str, object],
) -> AlgorithmResult:
    """执行通用匹配、未匹配策略和输出字段处理。"""

    source = get_single_slot(slots, "source_column")
    unmatched_policy = get_str_parameter(parameters, "unmatched", "keep")
    include_missing = get_bool_parameter(parameters, "include_missing", False)
    output = data.copy(deep=True)
    encoded_values: list[object] = []
    matched_count = 0
    unmatched_count = 0
    for value in data[source].tolist():
        if pd.isna(value) and not include_missing:
            encoded_values.append(value)
            continue
        matched, encoded = encoder(value)
        if matched:
            encoded_values.append(encoded)
            matched_count += 1
        else:
            unmatched_count += 1
            if unmatched_policy == "error":
                raise ProtocolError(
                    ErrorCode.DATA_ERROR,
                    f"字段 {source} 存在未匹配值",
                    details={"value": to_json_compatible(value)},
                )
            encoded_values.append(None if unmatched_policy == "null" else value)
    target_column = _target_column(output, source, parameters)
    output[target_column] = pd.Series(encoded_values, index=output.index, dtype="object")
    warnings = []
    if unmatched_count:
        warnings.append(
            WarningItem(
                code="UNMATCHED_VALUES",
                message=f"共有 {unmatched_count} 个值未匹配编码规则",
            )
        )
    mapping_frame = pd.DataFrame([{"mapping": to_json_compatible(mapping_description)}])
    return AlgorithmResult(
        data=output,
        metrics=[
            Metric(id="changed_count", display_name="已编码数量", value=matched_count, unit="个"),
            Metric(
                id="unmatched_count", display_name="未匹配数量", value=unmatched_count, unit="个"
            ),
        ],
        tables=[table_from_frame("encoding_mapping", "编码规则", mapping_frame)],
        warnings=warnings,
        metadata={
            "method": method,
            "source_column": source,
            "output_column": target_column,
            "rows_in": len(data.index),
            "rows_out": len(output.index),
            "changed_columns": [target_column],
        },
    )


def _target_column(
    output: pd.DataFrame,
    source: str,
    parameters: Mapping[str, JsonValue],
) -> str:
    """根据替换或新字段模式确定输出列名。"""

    output_mode = get_str_parameter(parameters, "output_mode", "new_column")
    if output_mode == "replace":
        return source
    preferred = get_str_parameter(parameters, "output_column_name", f"{source}_encoded")
    return unique_column_name(output, preferred)


def _ensure_non_overlapping_ranges(
    ranges: list[tuple[float | None, float | None, JsonValue]],
) -> None:
    """拒绝任意两个相交区间。"""

    normalized = sorted(ranges, key=lambda item: float("-inf") if item[0] is None else item[0])
    previous_upper: float | None = None
    for index, (lower, upper, _) in enumerate(normalized):
        if index and (previous_upper is None or lower is None or lower < previous_upper):
            raise ProtocolError(ErrorCode.PARAMETER_ERROR, "ranges 不能重叠")
        previous_upper = upper
