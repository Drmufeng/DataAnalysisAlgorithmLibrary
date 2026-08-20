"""算法运行前的字段槽位、数据条件和参数校验。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy

import numpy as np
import pandas as pd
from jsonschema import Draft202012Validator
from pydantic import JsonValue

from algorithm_sdk.errors import ErrorCode, ProtocolError
from algorithm_sdk.models import (
    AlgorithmRequest,
    DataType,
    InputConstraint,
    InputSlot,
    MethodManifest,
)


def infer_data_type(series: pd.Series) -> DataType:
    """根据 pandas dtype 判断字段的实际数据类型。"""

    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return DataType.CATEGORY
    if pd.api.types.is_bool_dtype(dtype):
        return DataType.BOOLEAN
    if pd.api.types.is_integer_dtype(dtype):
        return DataType.INTEGER
    if pd.api.types.is_numeric_dtype(dtype):
        return DataType.NUMBER
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return DataType.DATETIME
    if pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
        return DataType.STRING
    return DataType.UNKNOWN


def normalize_parameters_schema(schema: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """把协议扩展参数类型转换为 JSON Schema 2020-12 可校验结构。"""

    normalized = deepcopy(dict(schema))

    def visit(node: object) -> None:
        if not isinstance(node, dict):
            return
        if "additional_properties" in node:
            node["additionalProperties"] = node.pop("additional_properties")
        custom_type = node.get("type")
        if custom_type == "enum":
            node.pop("type")
        elif custom_type == "enum[]":
            enum_values = node.pop("enum", [])
            node["type"] = "array"
            node["items"] = {"enum": enum_values}
        elif custom_type == "scalar":
            node["type"] = ["string", "number", "integer", "boolean", "null"]
        elif custom_type == "column" or custom_type == "date_format":
            node["type"] = "string"
        elif custom_type == "columns":
            node["type"] = "array"
            node.setdefault("items", {"type": "string"})

        properties = node.get("properties")
        if isinstance(properties, dict):
            for property_schema in properties.values():
                visit(property_schema)
        items = node.get("items")
        visit(items)
        for key in ("allOf", "anyOf", "oneOf"):
            branches = node.get(key)
            if isinstance(branches, list):
                for branch in branches:
                    visit(branch)

    visit(normalized)
    return normalized


def validate_parameters(parameters: Mapping[str, JsonValue], method: MethodManifest) -> None:
    """按算法方法声明的参数 Schema 校验实际参数。"""

    schema = normalize_parameters_schema(method.parameters_schema)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(dict(parameters)), key=lambda item: list(item.path))
    if not errors:
        return
    first = errors[0]
    path = ".".join(str(item) for item in first.path) or "parameters"
    raise ProtocolError(
        ErrorCode.PARAMETER_ERROR,
        f"参数 {path} 不符合要求：{first.message}",
        details={"path": path},
    )


def resolve_parameters(
    parameters: Mapping[str, JsonValue],
    method: MethodManifest,
) -> dict[str, JsonValue]:
    """补齐清单中的默认值并返回通过校验的新参数字典。"""

    resolved = deepcopy(dict(parameters))
    properties = method.parameters_schema.get("properties")
    if isinstance(properties, dict):
        for name, raw_property in properties.items():
            if (
                isinstance(name, str)
                and name not in resolved
                and isinstance(raw_property, dict)
                and "default" in raw_property
            ):
                resolved[name] = deepcopy(raw_property["default"])
    validate_parameters(resolved, method)
    return resolved


def validate_request(
    data: pd.DataFrame,
    request: AlgorithmRequest,
    method: MethodManifest,
) -> None:
    """统一校验表格、字段槽位、字段元数据、组合约束和参数。"""

    _validate_table(data)
    slot_definitions = {slot.id: slot for slot in method.input.slots}
    unknown_slots = sorted(set(request.slots) - set(slot_definitions))
    if unknown_slots:
        raise ProtocolError(
            ErrorCode.COLUMN_ERROR,
            f"请求包含算法未声明的字段槽位：{', '.join(unknown_slots)}",
            details={"slots": unknown_slots},
        )

    for slot in method.input.slots:
        selected = request.slots.get(slot.id, [])
        _validate_slot_count(slot, selected)
        _validate_selected_columns(data, request, slot, selected)

    for constraint in method.input.constraints:
        _validate_constraint(data, request, constraint)
    validate_parameters(request.parameters, method)


def _validate_table(data: pd.DataFrame) -> None:
    """检查算法协议需要的基本表格条件。"""

    if not all(isinstance(column, str) for column in data.columns):
        raise ProtocolError(ErrorCode.COLUMN_ERROR, "所有字段名都必须是字符串")
    if data.columns.has_duplicates:
        duplicates = data.columns[data.columns.duplicated()].unique().tolist()
        raise ProtocolError(
            ErrorCode.COLUMN_ERROR,
            f"数据包含重复字段名：{', '.join(duplicates)}",
            details={"columns": duplicates},
        )


def _validate_slot_count(slot: InputSlot, selected: list[str]) -> None:
    """检查一个字段角色的数量和重复选择。"""

    if len(selected) != len(set(selected)):
        raise ProtocolError(
            ErrorCode.COLUMN_ERROR,
            f"字段槽位 {slot.id} 中存在重复字段",
            details={"slot": slot.id},
        )
    if len(selected) < slot.min_items:
        raise ProtocolError(
            ErrorCode.COLUMN_ERROR,
            f"字段槽位 {slot.display_name} 至少需要 {slot.min_items} 个字段",
            details={"slot": slot.id, "min_items": slot.min_items},
        )
    if slot.max_items is not None and len(selected) > slot.max_items:
        raise ProtocolError(
            ErrorCode.COLUMN_ERROR,
            f"字段槽位 {slot.display_name} 最多允许 {slot.max_items} 个字段",
            details={"slot": slot.id, "max_items": slot.max_items},
        )


def _validate_selected_columns(
    data: pd.DataFrame,
    request: AlgorithmRequest,
    slot: InputSlot,
    selected: list[str],
) -> None:
    """检查每个所选字段的存在性、实际类型、标签和取值条件。"""

    for column in selected:
        if column not in data.columns:
            raise ProtocolError(
                ErrorCode.COLUMN_ERROR,
                f"字段 {column} 不存在于当前数据版本",
                details={"slot": slot.id, "column": column},
            )
        metadata = request.context.field_metadata.get(column)
        if metadata is None:
            raise ProtocolError(
                ErrorCode.COLUMN_ERROR,
                f"字段 {column} 缺少运行时元数据快照",
                details={"slot": slot.id, "column": column},
            )

        series = data[column]
        actual_type = infer_data_type(series)
        if not _reported_type_matches_actual(metadata.data_type, actual_type):
            raise ProtocolError(
                ErrorCode.COLUMN_ERROR,
                f"字段 {column} 的登记类型与实际数据不一致",
                details={
                    "column": column,
                    "reported_type": metadata.data_type.value,
                    "actual_type": actual_type.value,
                },
            )
        if not _data_type_allowed(actual_type, slot.allowed_data_types):
            allowed = [item.value for item in slot.allowed_data_types]
            raise ProtocolError(
                ErrorCode.COLUMN_ERROR,
                f"字段 {column} 的实际类型 {actual_type.value} 不适用于该字段槽位",
                details={"column": column, "allowed_data_types": allowed},
            )
        if metadata.analysis_type not in slot.allowed_analysis_types:
            allowed_analysis = [item.value for item in slot.allowed_analysis_types]
            raise ProtocolError(
                ErrorCode.COLUMN_ERROR,
                f"字段 {column} 当前标记为 {metadata.analysis_type.value}，不适用于该字段槽位",
                details={"column": column, "allowed_analysis_types": allowed_analysis},
            )
        if (
            slot.allowed_semantic_roles is not None
            and metadata.semantic_role not in slot.allowed_semantic_roles
        ):
            raise ProtocolError(
                ErrorCode.COLUMN_ERROR,
                f"字段 {column} 的业务角色不适用于该字段槽位",
                details={"column": column, "semantic_role": metadata.semantic_role},
            )
        _validate_series_values(series, slot, column)


def _reported_type_matches_actual(reported: DataType, actual: DataType) -> bool:
    """数值型声明可以兼容实际整数，其他类型必须一致。"""

    return reported == actual or (reported == DataType.NUMBER and actual == DataType.INTEGER)


def _data_type_allowed(actual: DataType, allowed: list[DataType]) -> bool:
    """JSON 数值语义下 number 同时允许整数和浮点数。"""

    return actual in allowed or (actual == DataType.INTEGER and DataType.NUMBER in allowed)


def _validate_series_values(series: pd.Series, slot: InputSlot, column: str) -> None:
    """检查缺失、唯一值、有限值和可选数值范围。"""

    if not slot.allow_missing and bool(series.isna().any()):
        raise ProtocolError(
            ErrorCode.DATA_ERROR,
            f"字段 {column} 含有缺失值，该算法不允许缺失值",
            details={"column": column, "missing_count": int(series.isna().sum())},
        )
    unique_count = int(series.nunique(dropna=True))
    if slot.min_unique_values is not None and unique_count < slot.min_unique_values:
        raise ProtocolError(
            ErrorCode.DATA_ERROR,
            f"字段 {column} 的有效不同值数量不足",
            details={"column": column, "unique_count": unique_count},
        )
    if slot.max_unique_values is not None and unique_count > slot.max_unique_values:
        raise ProtocolError(
            ErrorCode.DATA_ERROR,
            f"字段 {column} 的不同值数量超过限制",
            details={"column": column, "unique_count": unique_count},
        )
    if slot.unique_values and bool(series.dropna().duplicated().any()):
        raise ProtocolError(
            ErrorCode.DATA_ERROR,
            f"字段 {column} 必须保持唯一，但存在重复值",
            details={"column": column},
        )

    if slot.finite_only or slot.value_range is not None:
        numeric = pd.to_numeric(series, errors="coerce")
        valid_numeric = numeric.dropna().astype(float)
        if slot.finite_only and not bool(np.isfinite(valid_numeric.to_numpy()).all()):
            raise ProtocolError(
                ErrorCode.DATA_ERROR,
                f"字段 {column} 含有无穷值",
                details={"column": column},
            )
        if slot.value_range is not None:
            _validate_value_range(valid_numeric, slot, column)


def _validate_value_range(series: pd.Series, slot: InputSlot, column: str) -> None:
    """根据清单声明检查字段数值范围。"""

    value_range = slot.value_range
    if value_range is None or series.empty:
        return
    minimum = float(series.min())
    maximum = float(series.max())
    below_minimum = value_range.minimum is not None and (
        minimum <= value_range.minimum
        if value_range.exclusive_minimum
        else minimum < value_range.minimum
    )
    above_maximum = value_range.maximum is not None and (
        maximum >= value_range.maximum
        if value_range.exclusive_maximum
        else maximum > value_range.maximum
    )
    if below_minimum or above_maximum:
        raise ProtocolError(
            ErrorCode.DATA_ERROR,
            f"字段 {column} 存在超出允许范围的数值",
            details={"column": column, "minimum": minimum, "maximum": maximum},
        )


def _validate_constraint(
    data: pd.DataFrame,
    request: AlgorithmRequest,
    constraint: InputConstraint,
) -> None:
    """执行协议已经定义的跨字段槽位约束。"""

    selected_by_slot = [request.slots.get(slot, []) for slot in constraint.slots]
    total_items = sum(len(items) for items in selected_by_slot)
    if constraint.type == "total_min_items":
        expected = _constraint_int_value(constraint)
        if total_items < expected:
            raise ProtocolError(
                ErrorCode.COLUMN_ERROR,
                f"字段槽位合计至少需要 {expected} 个字段",
                details={"slots": constraint.slots, "min_items": expected},
            )
    elif constraint.type == "total_max_items":
        expected = _constraint_int_value(constraint)
        if total_items > expected:
            raise ProtocolError(
                ErrorCode.COLUMN_ERROR,
                f"字段槽位合计最多允许 {expected} 个字段",
                details={"slots": constraint.slots, "max_items": expected},
            )
    elif constraint.type == "slots_disjoint":
        flattened = [column for items in selected_by_slot for column in items]
        duplicates = sorted({column for column in flattened if flattened.count(column) > 1})
        if duplicates:
            raise ProtocolError(
                ErrorCode.COLUMN_ERROR,
                f"互斥字段槽位重复使用了字段：{', '.join(duplicates)}",
                details={"columns": duplicates},
            )
    elif constraint.type == "requires_slot":
        if len(selected_by_slot) >= 2 and selected_by_slot[0] and not selected_by_slot[1]:
            raise ProtocolError(
                ErrorCode.COLUMN_ERROR,
                f"选择 {constraint.slots[0]} 后必须同时选择 {constraint.slots[1]}",
                details={"slots": constraint.slots[:2]},
            )
    elif constraint.type == "exact_category_count":
        expected = _constraint_int_value(constraint)
        for items in selected_by_slot:
            for column in items:
                actual = int(data[column].nunique(dropna=True))
                if actual != expected:
                    raise ProtocolError(
                        ErrorCode.DATA_ERROR,
                        f"字段 {column} 必须刚好有 {expected} 个有效类别",
                        details={"column": column, "category_count": actual},
                    )
    elif constraint.type == "min_rows":
        expected = _constraint_int_value(constraint)
        if len(data.index) < expected:
            raise ProtocolError(
                ErrorCode.DATA_ERROR,
                f"算法至少需要 {expected} 条数据",
                details={"rows": len(data.index), "min_rows": expected},
            )
    elif constraint.type == "time_order":
        _validate_time_order(data, selected_by_slot, constraint)
    elif constraint.type == "equal_interval":
        _validate_equal_interval(data, selected_by_slot)


def _constraint_int_value(constraint: InputConstraint) -> int:
    """读取必须为正整数的约束值。"""

    value = constraint.value
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProtocolError(
            ErrorCode.PARAMETER_ERROR,
            f"约束 {constraint.type} 缺少有效整数 value",
        )
    return value


def _first_constraint_column(selected_by_slot: list[list[str]]) -> str | None:
    """取得时序约束使用的第一个字段。"""

    for items in selected_by_slot:
        if items:
            return items[0]
    return None


def _validate_time_order(
    data: pd.DataFrame,
    selected_by_slot: list[list[str]],
    constraint: InputConstraint,
) -> None:
    """检查时间字段是否按照指定方向单调排列。"""

    column = _first_constraint_column(selected_by_slot)
    if column is None:
        return
    series = data[column].dropna()
    direction = constraint.value or "increasing"
    valid = (
        series.is_monotonic_increasing
        if direction == "increasing"
        else series.is_monotonic_decreasing
    )
    if not valid:
        raise ProtocolError(
            ErrorCode.DATA_ERROR,
            f"时间字段 {column} 未按 {direction} 顺序排列",
            details={"column": column, "direction": str(direction)},
        )


def _validate_equal_interval(data: pd.DataFrame, selected_by_slot: list[list[str]]) -> None:
    """检查时间或数值索引是否为等间隔。"""

    column = _first_constraint_column(selected_by_slot)
    if column is None:
        return
    series = data[column].dropna()
    if len(series.index) < 3:
        return
    differences = series.diff().dropna()
    first = differences.iloc[0]
    if not bool(differences.eq(first).all()):
        raise ProtocolError(
            ErrorCode.DATA_ERROR,
            f"字段 {column} 不是等间隔序列",
            details={"column": column},
        )


def is_finite_number(value: object) -> bool:
    """供算法结果合理性检查复用的有限数值判断。"""

    return (
        isinstance(value, (int, float, np.number))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
