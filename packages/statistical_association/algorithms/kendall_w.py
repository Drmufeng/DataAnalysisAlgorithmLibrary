"""带并列秩修正的 Kendall 协同系数 W。"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from pydantic import JsonValue
from scipy import stats

from algorithm_sdk import AlgorithmResult, Metric
from algorithm_sdk.algorithm_utils import (
    get_float_parameter,
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
    """计算多位评价者对同一批对象的秩次一致程度。"""

    del context
    columns = get_slot(slots, "rater_columns")
    direction = get_str_parameter(parameters, "rank_direction", "ascending")
    alpha = get_float_parameter(parameters, "alpha", 0.05)
    frame = data[columns].apply(pd.to_numeric, errors="coerce").dropna()
    object_count, rater_count = frame.shape
    if object_count < 2:
        raise ProtocolError(ErrorCode.DATA_ERROR, "Kendall W 至少需要两个完整评价对象")
    ranks = frame.rank(axis=0, method="average", ascending=direction == "ascending")
    rank_sums = ranks.sum(axis=1)
    centered_sum = float(np.sum((rank_sums - rank_sums.mean()) ** 2))
    tie_correction = 0.0
    for column in columns:
        counts = frame[column].value_counts()
        tie_correction += float(np.sum(counts**3 - counts))
    denominator = rater_count**2 * (object_count**3 - object_count) - rater_count * tie_correction
    if denominator <= 0:
        raise ProtocolError(ErrorCode.DATA_ERROR, "并列秩结构导致 Kendall W 无法计算")
    w_value = 12 * centered_sum / denominator
    chi_square = rater_count * (object_count - 1) * w_value
    degrees = object_count - 1
    p_value = float(stats.chi2.sf(chi_square, degrees))
    rank_summary = pd.DataFrame(
        {"object_index": [str(index) for index in frame.index], "rank_sum": rank_sums.to_numpy()}
    )
    return AlgorithmResult(
        metrics=[
            Metric(id="kendall_w", display_name="Kendall's W", value=w_value),
            Metric(id="chi_square", display_name="卡方统计量", value=chi_square),
            Metric(id="p_value", display_name="P 值", value=p_value),
        ],
        tables=[table_from_frame("rank_summary", "秩次汇总", rank_summary)],
        metadata={
            "method": "kendall_w.concordance",
            "degrees_of_freedom": degrees,
            "alpha": alpha,
            "conclusion": "存在显著一致性" if p_value < alpha else "未发现显著一致性",
        },
    )
