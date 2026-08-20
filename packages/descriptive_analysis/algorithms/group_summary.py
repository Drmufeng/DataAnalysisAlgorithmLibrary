"""多字段分类汇总。"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
from pydantic import JsonValue

from algorithm_sdk import AlgorithmResult, Metric
from algorithm_sdk.algorithm_utils import (
    get_bool_parameter,
    get_int_parameter,
    get_list_parameter,
    get_slot,
    table_from_frame,
)
from algorithm_sdk.errors import ErrorCode, ProtocolError


def run(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """按一个或多个定类字段汇总定量字段。"""

    del context
    group_columns = get_slot(slots, "group_columns")
    value_columns = get_slot(slots, "value_columns")
    aggregation_values = get_list_parameter(parameters, "aggregations", ["count"])
    aggregations = [value for value in aggregation_values if isinstance(value, str)]
    drop_missing_groups = get_bool_parameter(parameters, "drop_missing_groups", True)
    round_digits = get_int_parameter(parameters, "round_digits", 4)
    if not value_columns and any(name != "count" for name in aggregations):
        raise ProtocolError(
            ErrorCode.PARAMETER_ERROR,
            "未选择汇总定量字段时只能使用 count",
        )
    grouped = data.groupby(group_columns, dropna=drop_missing_groups, sort=True)
    if not value_columns:
        output = grouped.size().reset_index(name="count")
    else:
        named_aggregations: dict[str, pd.NamedAgg] = {}
        for column in value_columns:
            for name in aggregations:
                if name == "count":
                    named_aggregations[f"{column}_count"] = pd.NamedAgg(
                        column=column, aggfunc="count"
                    )
                else:
                    named_aggregations[f"{column}_{name}"] = pd.NamedAgg(
                        column=column, aggfunc=name
                    )
        output = grouped.agg(**named_aggregations).reset_index()
    numeric_columns = output.select_dtypes(include="number").columns
    output.loc[:, numeric_columns] = output.loc[:, numeric_columns].round(round_digits)
    return AlgorithmResult(
        metrics=[
            Metric(id="group_count", display_name="分组数量", value=len(output.index), unit="组")
        ],
        tables=[table_from_frame("group_summary", "分类汇总", output)],
        metadata={
            "method": "group_summary.aggregate",
            "group_columns": group_columns,
            "value_columns": value_columns,
        },
    )
