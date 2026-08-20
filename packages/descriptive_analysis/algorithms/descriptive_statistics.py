"""定量字段描述性统计。"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from pydantic import JsonValue
from scipy import stats

from algorithm_sdk import AlgorithmResult, ChartSpec, Metric, WarningItem
from algorithm_sdk.algorithm_utils import (
    finite_or_none,
    get_int_parameter,
    get_list_parameter,
    get_slot,
    table_from_frame,
)

DEFAULT_STATISTICS = ["count", "missing", "mean", "std", "min", "q1", "median", "q3", "max"]


def run(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """计算常用集中趋势、离散程度、偏度和峰度。"""

    del context
    columns = get_slot(slots, "x_columns")
    requested_raw = get_list_parameter(parameters, "statistics", DEFAULT_STATISTICS)
    requested = [value for value in requested_raw if isinstance(value, str)]
    round_digits = get_int_parameter(parameters, "round_digits", 4)
    rows: list[dict[str, JsonValue]] = []
    charts: list[ChartSpec] = []
    warnings: list[WarningItem] = []
    for column in columns:
        series = pd.to_numeric(data[column], errors="coerce")
        valid = series.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
        count = len(valid.index)
        mean = float(valid.mean()) if count else np.nan
        std = float(valid.std(ddof=1)) if count >= 2 else np.nan
        minimum = float(valid.min()) if count else np.nan
        maximum = float(valid.max()) if count else np.nan
        first_quartile = float(valid.quantile(0.25)) if count else np.nan
        third_quartile = float(valid.quantile(0.75)) if count else np.nan
        values: dict[str, float | int | None] = {
            "count": count,
            "missing": int(len(series.index) - count),
            "mean": finite_or_none(mean),
            "std": finite_or_none(std),
            "variance": finite_or_none(std**2),
            "min": finite_or_none(minimum),
            "q1": finite_or_none(first_quartile),
            "median": finite_or_none(valid.median()) if count else None,
            "q3": finite_or_none(third_quartile),
            "max": finite_or_none(maximum),
            "range": finite_or_none(maximum - minimum),
            "iqr": finite_or_none(third_quartile - first_quartile),
            "skewness": finite_or_none(stats.skew(valid, bias=False)) if count >= 3 else None,
            "kurtosis": finite_or_none(stats.kurtosis(valid, bias=False)) if count >= 4 else None,
            "coefficient_of_variation": (
                finite_or_none(std / mean) if count >= 2 and mean != 0 else None
            ),
            "standard_error": finite_or_none(std / np.sqrt(count)) if count >= 2 else None,
        }
        row: dict[str, JsonValue] = {"field": column}
        for name in requested:
            value = values.get(name)
            row[name] = round(value, round_digits) if isinstance(value, float) else value
        rows.append(row)
        if count and valid.nunique() <= 1:
            warnings.append(
                WarningItem(
                    code="CONSTANT_COLUMN",
                    message=f"字段 {column} 为常数列，部分离散统计量不可计算",
                )
            )
        hist_counts, hist_edges = np.histogram(valid.to_numpy(), bins="auto") if count else ([], [])
        charts.append(
            ChartSpec(
                id=f"distribution_{len(charts)}",
                display_name=f"{column} 分布",
                chart_type="histogram_boxplot",
                data={
                    "field": column,
                    "histogram_counts": hist_counts,
                    "histogram_edges": hist_edges,
                    "boxplot": [
                        values["min"],
                        values["q1"],
                        values["median"],
                        values["q3"],
                        values["max"],
                    ],
                },
            )
        )
    return AlgorithmResult(
        metrics=[
            Metric(id="field_count", display_name="分析字段数", value=len(columns), unit="个")
        ],
        tables=[table_from_frame("descriptive_statistics", "描述性统计", pd.DataFrame(rows))],
        charts=charts,
        warnings=warnings,
        metadata={"method": "descriptive_statistics.basic", "round_digits": round_digits},
    )
