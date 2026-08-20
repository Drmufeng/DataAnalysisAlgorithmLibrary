"""算法实现复用的参数、字段、数值和结果辅助函数。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from pydantic import JsonValue

from algorithm_sdk.errors import ErrorCode, ProtocolError
from algorithm_sdk.models import ResultTable, TableColumn
from algorithm_sdk.serialization import to_json_compatible
from algorithm_sdk.validation import infer_data_type


def get_int_parameter(
    parameters: Mapping[str, JsonValue],
    name: str,
    default: int,
) -> int:
    """读取已通过 Schema 校验的整数参数。"""

    value = parameters.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ProtocolError(ErrorCode.PARAMETER_ERROR, f"参数 {name} 必须是整数")
    return value


def get_float_parameter(
    parameters: Mapping[str, JsonValue],
    name: str,
    default: float,
) -> float:
    """读取已通过 Schema 校验的数值参数。"""

    value = parameters.get(name, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ProtocolError(ErrorCode.PARAMETER_ERROR, f"参数 {name} 必须是数值")
    return float(value)


def get_bool_parameter(
    parameters: Mapping[str, JsonValue],
    name: str,
    default: bool,
) -> bool:
    """读取布尔参数。"""

    value = parameters.get(name, default)
    if not isinstance(value, bool):
        raise ProtocolError(ErrorCode.PARAMETER_ERROR, f"参数 {name} 必须是布尔值")
    return value


def get_str_parameter(
    parameters: Mapping[str, JsonValue],
    name: str,
    default: str,
) -> str:
    """读取字符串或枚举参数。"""

    value = parameters.get(name, default)
    if not isinstance(value, str):
        raise ProtocolError(ErrorCode.PARAMETER_ERROR, f"参数 {name} 必须是字符串")
    return value


def get_list_parameter(
    parameters: Mapping[str, JsonValue],
    name: str,
    default: Sequence[JsonValue] = (),
) -> list[JsonValue]:
    """读取数组参数并返回副本。"""

    value = parameters.get(name, list(default))
    if not isinstance(value, list):
        raise ProtocolError(ErrorCode.PARAMETER_ERROR, f"参数 {name} 必须是数组")
    return list(value)


def get_dict_parameter(
    parameters: Mapping[str, JsonValue],
    name: str,
) -> dict[str, JsonValue]:
    """读取对象参数。"""

    value = parameters.get(name, {})
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ProtocolError(ErrorCode.PARAMETER_ERROR, f"参数 {name} 必须是对象")
    return dict(value)


def get_slot(slots: Mapping[str, list[str]], name: str) -> list[str]:
    """读取字段槽位并返回副本。"""

    return list(slots.get(name, []))


def get_single_slot(slots: Mapping[str, list[str]], name: str) -> str:
    """读取已经过数量校验的单字段槽位。"""

    values = slots.get(name, [])
    if len(values) != 1:
        raise ProtocolError(ErrorCode.COLUMN_ERROR, f"字段槽位 {name} 必须恰好包含一个字段")
    return values[0]


def field_metadata(
    context: Mapping[str, JsonValue],
    column: str,
) -> dict[str, JsonValue]:
    """取得一次运行保存的字段元数据。"""

    all_metadata = context.get("field_metadata")
    if not isinstance(all_metadata, dict):
        return {}
    value = all_metadata.get(column)
    return dict(value) if isinstance(value, dict) else {}


def field_analysis_type(context: Mapping[str, JsonValue], column: str) -> str:
    """读取字段当前定类或定量标签。"""

    value = field_metadata(context, column).get("analysis_type")
    return value if isinstance(value, str) else "unknown"


def field_measurement_level(context: Mapping[str, JsonValue], column: str) -> str | None:
    """读取 nominal、ordinal 或 scale 测量水平。"""

    value = field_metadata(context, column).get("measurement_level")
    return value if isinstance(value, str) else None


def field_category_order(context: Mapping[str, JsonValue], column: str) -> list[JsonValue]:
    """读取有序定类字段的类别顺序快照。"""

    value = field_metadata(context, column).get("category_order")
    return list(value) if isinstance(value, list) else []


def random_seed_from_context(context: Mapping[str, JsonValue]) -> int:
    """所有随机算法统一使用运行上下文中的种子。"""

    value = context.get("random_seed", 42)
    return value if isinstance(value, int) and not isinstance(value, bool) else 42


def complete_cases(data: pd.DataFrame, columns: Sequence[str]) -> tuple[pd.DataFrame, int]:
    """按指定字段整行删除缺失值，并返回删除数量。"""

    selected = data.loc[:, list(columns)]
    valid_mask = selected.notna().all(axis=1)
    return data.loc[valid_mask].copy(), int((~valid_mask).sum())


def numeric_frame(
    data: pd.DataFrame,
    columns: Sequence[str],
    *,
    allow_constant: bool = False,
) -> pd.DataFrame:
    """转换数值字段并检查非有限值与常数列。"""

    frame = pd.DataFrame(
        data.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").astype(float),
        index=data.index,
        columns=list(columns),
    )
    values = frame.to_numpy(dtype=float)
    if not bool(np.isfinite(values).all()):
        raise ProtocolError(ErrorCode.DATA_ERROR, "参与计算的定量字段包含缺失值或无穷值")
    constant = [str(column) for column in frame.columns if frame[column].nunique(dropna=True) <= 1]
    if constant and not allow_constant:
        raise ProtocolError(
            ErrorCode.DATA_ERROR,
            f"参与计算的字段包含常数列：{', '.join(constant)}",
            details={"columns": constant},
        )
    return frame


def unique_column_name(data: pd.DataFrame, preferred: str) -> str:
    """为派生字段生成不会覆盖原字段的稳定名称。"""

    if preferred not in data.columns:
        return preferred
    index = 2
    while f"{preferred}_{index}" in data.columns:
        index += 1
    return f"{preferred}_{index}"


def table_from_frame(
    table_id: str,
    display_name: str,
    frame: pd.DataFrame,
) -> ResultTable:
    """把小型 pandas 表转换为与前端无关的结果表。"""

    normalized = frame.copy()
    normalized.columns = [str(column) for column in normalized.columns]
    rows = [
        {column: to_json_compatible(value) for column, value in row.items()}
        for row in normalized.to_dict(orient="records")
    ]
    return ResultTable(
        id=table_id,
        display_name=display_name,
        columns=[
            TableColumn(
                id=column,
                display_name=column,
                data_type=infer_data_type(normalized[column]),
            )
            for column in normalized.columns
        ],
        rows=rows,
    )


def listwise_model_data(
    data: pd.DataFrame,
    y_column: str,
    x_columns: Sequence[str],
) -> tuple[pd.DataFrame, int]:
    """为监督模型生成包含 Y 和 X 的完整样本。"""

    columns = [y_column, *x_columns]
    return complete_cases(data, columns)


def encode_features(
    frame: pd.DataFrame,
    x_columns: Sequence[str],
    context: Mapping[str, JsonValue],
    *,
    drop_first: bool = True,
) -> tuple[pd.DataFrame, dict[str, JsonValue]]:
    """按字段标签对定类 X 做独热编码，并保留映射和参考类别。"""

    parts: list[pd.DataFrame] = []
    encoding: dict[str, JsonValue] = {}
    for column in x_columns:
        if field_analysis_type(context, column) == "categorical":
            categories = sorted(frame[column].astype(str).unique().tolist())
            if len(categories) > 100:
                raise ProtocolError(
                    ErrorCode.DATA_ERROR,
                    f"定类字段 {column} 的类别数超过 100，不适合直接独热编码",
                    details={"column": column, "category_count": len(categories)},
                )
            categorical = pd.Categorical(frame[column].astype(str), categories=categories)
            dummy = pd.get_dummies(
                categorical, prefix=column, prefix_sep="=", drop_first=drop_first
            )
            dummy.index = frame.index
            parts.append(dummy.astype(float))
            encoding[column] = {
                "categories": categories,
                "reference": categories[0] if drop_first and categories else None,
            }
        else:
            numeric = pd.to_numeric(frame[column], errors="coerce").astype(float)
            if not bool(np.isfinite(numeric.to_numpy()).all()):
                raise ProtocolError(ErrorCode.DATA_ERROR, f"定量字段 {column} 包含非有限值")
            parts.append(numeric.to_frame(column))
    if not parts:
        raise ProtocolError(ErrorCode.DATA_ERROR, "模型没有可用自变量")
    features = pd.concat(parts, axis=1)
    if features.shape[1] == 0:
        raise ProtocolError(ErrorCode.DATA_ERROR, "定类字段编码后没有可用特征")
    return features, encoding


def check_sample_parameter_count(sample_count: int, parameter_count: int) -> None:
    """回归模型必须保留正的残差自由度。"""

    if sample_count <= parameter_count:
        raise ProtocolError(
            ErrorCode.DATA_ERROR,
            "有效样本数必须大于模型估计参数数",
            details={"sample_count": sample_count, "parameter_count": parameter_count},
        )


def finite_or_none(value: float | np.floating[Any]) -> float | None:
    """统计量不可计算时统一返回 null。"""

    number = float(value)
    return number if np.isfinite(number) else None
