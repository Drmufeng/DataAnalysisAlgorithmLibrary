"""精确频数和分箱频数分析。"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
from pydantic import JsonValue

from algorithm_sdk import AlgorithmResult, Metric, WarningItem
from algorithm_sdk.algorithm_utils import (
    field_analysis_type,
    get_bool_parameter,
    get_int_parameter,
    get_list_parameter,
    get_slot,
    get_str_parameter,
    table_from_frame,
)
from algorithm_sdk.errors import ErrorCode, ProtocolError


def run_exact(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """按字段原值计算频数、百分比和累计百分比。"""

    columns = get_slot(slots, "columns")
    include_missing = get_bool_parameter(parameters, "include_missing", False)
    sort_by = get_str_parameter(parameters, "sort_by", "count_desc")
    top_n = get_int_parameter(parameters, "top_n", 0)
    tables = []
    warnings: list[WarningItem] = []
    for index, column in enumerate(columns):
        series = data[column]
        valid = series if include_missing else series.dropna()
        if include_missing:
            valid = valid.astype(object).where(valid.notna(), "（缺失）")
        counts = valid.value_counts(dropna=False, sort=False)
        frequency = counts.rename_axis("value").reset_index(name="frequency")
        if sort_by == "count_desc":
            frequency = frequency.sort_values(
                ["frequency", "value"], ascending=[False, True], kind="stable"
            )
        else:
            try:
                frequency = frequency.sort_values("value", kind="stable")
            except TypeError:
                frequency["_sort"] = frequency["value"].astype(str)
                frequency = frequency.sort_values("_sort", kind="stable").drop(columns="_sort")
        if top_n > 0:
            frequency = frequency.head(top_n)
        denominator = max(len(valid.index), 1)
        frequency["percentage"] = frequency["frequency"] / denominator
        frequency["cumulative_frequency"] = frequency["frequency"].cumsum()
        frequency["cumulative_percentage"] = frequency["percentage"].cumsum()
        frequency.insert(0, "field", column)
        tables.append(table_from_frame(f"frequency_{index}", f"{column} 频数表", frequency))
        unique_ratio = series.nunique(dropna=True) / max(int(series.notna().sum()), 1)
        if field_analysis_type(context, column) == "quantitative" and unique_ratio > 0.5:
            warnings.append(
                WarningItem(
                    code="HIGH_CARDINALITY_EXACT_FREQUENCY",
                    message=f"定量字段 {column} 的不同值比例较高，分箱频数通常更容易解释",
                    details={"column": column, "unique_ratio": unique_ratio},
                )
            )
    return AlgorithmResult(
        metrics=[
            Metric(id="field_count", display_name="分析字段数", value=len(columns), unit="个")
        ],
        tables=tables,
        warnings=warnings,
        metadata={"method": "frequency_analysis.exact", "rows_in": len(data.index)},
    )


def run_binned(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """对连续定量字段分箱后计算频数。"""

    del context
    columns = get_slot(slots, "columns")
    include_missing = get_bool_parameter(parameters, "include_missing", False)
    bin_method = get_str_parameter(parameters, "bin_method", "equal_width")
    bin_count = get_int_parameter(parameters, "bin_count", 10)
    custom_edges = get_list_parameter(parameters, "custom_edges")
    tables = []
    warnings: list[WarningItem] = []
    for index, column in enumerate(columns):
        numeric = pd.to_numeric(data[column], errors="coerce")
        if int(numeric.notna().sum()) == 0:
            raise ProtocolError(ErrorCode.DATA_ERROR, f"字段 {column} 没有可分箱的有效数值")
        try:
            if bin_method == "equal_frequency":
                binned = pd.qcut(numeric, q=bin_count, duplicates="drop")
            elif bin_method == "custom":
                edges = [float(value) for value in custom_edges if isinstance(value, (int, float))]
                if len(edges) < 2 or edges != sorted(set(edges)):
                    raise ProtocolError(
                        ErrorCode.PARAMETER_ERROR,
                        "custom_edges 至少包含两个严格递增边界",
                    )
                binned = pd.cut(numeric, bins=edges, include_lowest=True)
            else:
                binned = pd.cut(numeric, bins=bin_count, include_lowest=True, duplicates="drop")
        except ValueError as exc:
            raise ProtocolError(ErrorCode.DATA_ERROR, f"字段 {column} 无法按当前参数分箱") from exc
        if include_missing:
            labels = binned.astype(object).where(numeric.notna(), "（缺失）")
        else:
            labels = binned[numeric.notna()]
        counts = labels.value_counts(dropna=False, sort=False)
        frequency = counts.rename_axis("interval").reset_index(name="frequency")
        denominator = max(len(labels.index), 1)
        frequency["percentage"] = frequency["frequency"] / denominator
        frequency["cumulative_frequency"] = frequency["frequency"].cumsum()
        frequency["cumulative_percentage"] = frequency["percentage"].cumsum()
        frequency.insert(0, "field", column)
        tables.append(
            table_from_frame(f"binned_frequency_{index}", f"{column} 分箱频数", frequency)
        )
        actual_bins = int(binned.dropna().nunique())
        if actual_bins < bin_count and bin_method == "equal_frequency":
            warnings.append(
                WarningItem(
                    code="DUPLICATE_BIN_EDGES",
                    message=f"字段 {column} 因重复值只形成 {actual_bins} 个有效分箱",
                )
            )
    return AlgorithmResult(
        metrics=[
            Metric(id="field_count", display_name="分析字段数", value=len(columns), unit="个")
        ],
        tables=tables,
        warnings=warnings,
        metadata={
            "method": "frequency_analysis.binned",
            "bin_method": bin_method,
            "bin_count": bin_count,
        },
    )
