"""列联表、百分比和卡方独立性检验。"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from pydantic import JsonValue
from scipy import stats

from algorithm_sdk import AlgorithmResult, Metric, WarningItem
from algorithm_sdk.algorithm_utils import (
    field_analysis_type,
    get_bool_parameter,
    get_dict_parameter,
    get_single_slot,
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
    """对一个分组字段和多个 X 字段分别生成交叉分析。"""

    group_column = get_single_slot(slots, "group_column")
    x_columns = get_slot(slots, "x_columns")
    include_missing = get_bool_parameter(parameters, "include_missing", False)
    include_chi_square = get_bool_parameter(parameters, "include_chi_square", True)
    normalize = get_str_parameter(parameters, "normalize", "none")
    binning = get_dict_parameter(parameters, "binning")
    tables = []
    metrics: list[Metric] = []
    warnings: list[WarningItem] = []
    for index, x_column in enumerate(x_columns):
        frame = data[[group_column, x_column]].copy()
        if field_analysis_type(context, x_column) == "quantitative":
            config = binning.get(x_column)
            if not isinstance(config, dict):
                raise ProtocolError(
                    ErrorCode.PARAMETER_ERROR,
                    f"定量字段 {x_column} 必须提供 binning 配置",
                )
            count_value = config.get("bin_count", 5)
            bin_count = int(count_value) if isinstance(count_value, (int, float)) else 5
            frame[x_column] = pd.cut(
                pd.to_numeric(frame[x_column], errors="coerce"),
                bins=bin_count,
                include_lowest=True,
            )
        if include_missing:
            frame = frame.astype(object).where(frame.notna(), "（缺失）")
        else:
            frame = frame.dropna()
        if frame.empty:
            raise ProtocolError(ErrorCode.DATA_ERROR, f"字段 {x_column} 没有可用于列联分析的样本")
        counts = pd.crosstab(frame[group_column], frame[x_column], dropna=False)
        if counts.shape[0] < 2 or counts.shape[1] < 2:
            raise ProtocolError(ErrorCode.DATA_ERROR, f"字段 {x_column} 的列联表至少需要 2×2 类别")
        row_pct = counts.div(counts.sum(axis=1), axis=0)
        col_pct = counts.div(counts.sum(axis=0), axis=1)
        long_rows = []
        for group_value in counts.index:
            for x_value in counts.columns:
                long_rows.append(
                    {
                        group_column: group_value,
                        x_column: x_value,
                        "frequency": int(counts.loc[group_value, x_value]),
                        "row_percentage": float(row_pct.loc[group_value, x_value]),
                        "column_percentage": float(col_pct.loc[group_value, x_value]),
                    }
                )
        result_frame = pd.DataFrame(long_rows)
        if normalize != "none":
            if normalize == "all":
                normalized = pd.crosstab(
                    frame[group_column], frame[x_column], normalize="all", dropna=False
                )
            elif normalize == "columns":
                normalized = pd.crosstab(
                    frame[group_column], frame[x_column], normalize="columns", dropna=False
                )
            else:
                normalized = pd.crosstab(
                    frame[group_column], frame[x_column], normalize="index", dropna=False
                )
            result_frame["normalized_value"] = [
                float(normalized.loc[row[group_column], row[x_column]]) for row in long_rows
            ]
        tables.append(
            table_from_frame(f"contingency_{index}", f"{x_column} × {group_column}", result_frame)
        )
        if include_chi_square:
            statistic, p_value, degrees, expected = stats.chi2_contingency(counts.to_numpy())
            metrics.extend(
                [
                    Metric(
                        id=f"chi_square_{index}",
                        display_name=f"{x_column} 卡方统计量",
                        value=float(statistic),
                    ),
                    Metric(
                        id=f"p_value_{index}", display_name=f"{x_column} P 值", value=float(p_value)
                    ),
                ]
            )
            low_expected = int(np.sum(expected < 5))
            if low_expected:
                warnings.append(
                    WarningItem(
                        code="LOW_EXPECTED_FREQUENCY",
                        message=f"{x_column} 的列联表有 {low_expected} 个期望频数小于 5",
                        details={"degrees_of_freedom": int(degrees)},
                    )
                )
    return AlgorithmResult(
        metrics=metrics,
        tables=tables,
        warnings=warnings,
        metadata={"method": "contingency_analysis.crosstab", "rows_in": len(data.index)},
    )
