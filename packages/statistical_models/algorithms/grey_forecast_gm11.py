"""带级比检验、自动平移和后验检验的 GM(1,1)。"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from pydantic import JsonValue

from algorithm_sdk import AlgorithmResult, ChartSpec, Metric, WarningItem
from algorithm_sdk.algorithm_utils import (
    get_float_parameter,
    get_int_parameter,
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
    """拟合 GM(1,1)，输出样本内拟合和未来预测。"""

    del context
    y_column = get_single_slot(slots, "y_column")
    time_columns = get_slot(slots, "time_column")
    forecast_steps = get_int_parameter(parameters, "forecast_steps", 1)
    coefficient = get_float_parameter(parameters, "background_value_coefficient", 0.5)
    translation_mode = get_str_parameter(parameters, "translation_mode", "auto")
    frame = data[[*time_columns, y_column]].dropna().copy()
    if time_columns:
        time_column = time_columns[0]
        if frame[time_column].duplicated().any():
            raise ProtocolError(ErrorCode.DATA_ERROR, "GM(1,1) 时间字段必须唯一")
        frame = frame.sort_values(time_column, kind="stable")
        time_values = frame[time_column].astype(str).tolist()
    else:
        time_values = [str(index + 1) for index in range(len(frame.index))]
    original = pd.to_numeric(frame[y_column], errors="coerce").to_numpy(dtype=float)
    if len(original) < 4 or not bool(np.isfinite(original).all()):
        raise ProtocolError(ErrorCode.DATA_ERROR, "GM(1,1) 至少需要 4 个有限有效样本")
    translation = 0.0
    if bool((original < 0).any()) or not _level_ratio_passes(original):
        if translation_mode == "none":
            raise ProtocolError(ErrorCode.DATA_ERROR, "原序列未通过非负或级比检验，请启用自动平移")
        translation = _find_translation(original)
    sequence = original + translation
    accumulated = np.cumsum(sequence)
    background = coefficient * accumulated[:-1] + (1 - coefficient) * accumulated[1:]
    design = np.column_stack((-background, np.ones(len(background))))
    development, grey_action = np.linalg.lstsq(design, sequence[1:], rcond=None)[0]
    if abs(development) < np.finfo(float).eps:
        raise ProtocolError(ErrorCode.DATA_ERROR, "发展系数过于接近 0，模型数值不稳定")
    total_length = len(sequence) + forecast_steps
    accumulated_prediction = np.array(
        [
            (sequence[0] - grey_action / development) * np.exp(-development * index)
            + grey_action / development
            for index in range(total_length)
        ]
    )
    restored = np.empty(total_length)
    restored[0] = accumulated_prediction[0]
    restored[1:] = np.diff(accumulated_prediction)
    restored -= translation
    fitted = restored[: len(original)]
    residuals = original - fitted
    relative_error = np.abs(residuals) / np.maximum(np.abs(original), np.finfo(float).eps)
    original_std = float(np.std(original, ddof=1))
    residual_std = float(np.std(residuals, ddof=1))
    posterior_ratio = residual_std / original_std if original_std else np.inf
    small_error_probability = float(
        np.mean(np.abs(residuals - residuals.mean()) < 0.6745 * original_std)
    )
    fit_frame = pd.DataFrame(
        {
            "time": time_values,
            "actual": original,
            "fitted": fitted,
            "residual": residuals,
            "relative_error": relative_error,
        }
    )
    forecast_frame = pd.DataFrame(
        {
            "step": range(1, forecast_steps + 1),
            "forecast": restored[len(original) :],
        }
    )
    warnings = []
    if not time_columns:
        warnings.append(
            WarningItem(code="ROW_ORDER_USED", message="未提供时间字段，使用当前数据行顺序")
        )
    if translation:
        warnings.append(
            WarningItem(
                code="SEQUENCE_TRANSLATED",
                message="原序列未通过级比检验，模型已记录并应用自动平移",
                details={"translation": translation},
            )
        )
    return AlgorithmResult(
        metrics=[
            Metric(id="development_coefficient", display_name="发展系数", value=float(development)),
            Metric(id="grey_action", display_name="灰作用量", value=float(grey_action)),
            Metric(id="posterior_ratio", display_name="后验差比", value=posterior_ratio),
            Metric(
                id="small_error_probability",
                display_name="小误差概率",
                value=small_error_probability,
            ),
            Metric(
                id="mean_relative_error",
                display_name="平均相对误差",
                value=float(relative_error.mean()),
            ),
        ],
        tables=[
            table_from_frame("gm11_fit", "GM(1,1) 样本内拟合", fit_frame),
            table_from_frame("gm11_forecast", "GM(1,1) 未来预测", forecast_frame),
        ],
        charts=[
            ChartSpec(
                id="gm11_series",
                display_name="实际、拟合与预测序列",
                chart_type="line",
                data={"actual": original, "fitted": fitted, "forecast": restored[len(original) :]},
            )
        ],
        warnings=warnings,
        metadata={
            "method": "grey_forecast_gm11.basic",
            "sample_count": len(original),
            "forecast_steps": forecast_steps,
            "translation": translation,
            "level_ratio_passed": _level_ratio_passes(sequence),
        },
    )


def _level_ratio_passes(sequence: np.ndarray) -> bool:
    """检查 GM(1,1) 常用级比可容区间。"""

    if bool((sequence <= 0).any()):
        return False
    ratios = sequence[:-1] / sequence[1:]
    lower = np.exp(-2 / (len(sequence) + 1))
    upper = np.exp(2 / (len(sequence) + 1))
    return bool(((ratios >= lower) & (ratios <= upper)).all())


def _find_translation(sequence: np.ndarray) -> float:
    """确定能通过非负和级比检查的最小近似平移量。"""

    translation = max(0.0, -float(sequence.min()) + 1e-8)
    step = max(float(np.std(sequence)), 1.0) * 0.1
    for _ in range(100000):
        if _level_ratio_passes(sequence + translation):
            return translation
        translation += step
    raise ProtocolError(ErrorCode.DATA_ERROR, "无法在合理范围内找到通过级比检验的平移量")
