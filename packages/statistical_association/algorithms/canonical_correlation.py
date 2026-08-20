"""典型相关分析、载荷、冗余度和 Wilks Lambda 近似检验。"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from pydantic import JsonValue
from scipy import linalg, stats
from sklearn.preprocessing import StandardScaler

from algorithm_sdk import AlgorithmResult, ChartSpec, Metric, WarningItem
from algorithm_sdk.algorithm_utils import (
    complete_cases,
    get_bool_parameter,
    get_float_parameter,
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
    """使用正则化协方差白化和 SVD 计算典型变量。"""

    del context
    x_columns = get_slot(slots, "x_set")
    y_columns = get_slot(slots, "y_set")
    if set(x_columns) & set(y_columns):
        raise ProtocolError(ErrorCode.COLUMN_ERROR, "典型相关的 X 集合和 Y 集合必须互斥")
    frame, dropped = complete_cases(data, [*x_columns, *y_columns])
    sample_count = len(frame.index)
    total_variables = len(x_columns) + len(y_columns)
    if sample_count <= total_variables:
        raise ProtocolError(ErrorCode.DATA_ERROR, "有效样本数必须大于 X、Y 总变量数")
    standardize = get_bool_parameter(parameters, "standardize", True)
    regularization = get_float_parameter(parameters, "regularization", 0.0)
    x = frame[x_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y = frame[y_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not bool(np.isfinite(x).all() and np.isfinite(y).all()):
        raise ProtocolError(ErrorCode.DATA_ERROR, "典型相关输入包含无法转换的数值")
    if standardize:
        x = StandardScaler().fit_transform(x)
        y = StandardScaler().fit_transform(y)
    else:
        x = x - x.mean(axis=0)
        y = y - y.mean(axis=0)
    maximum = min(x.shape[1], y.shape[1])
    component_raw = parameters.get("n_components", "auto")
    if component_raw == "auto":
        component_count = maximum
    elif isinstance(component_raw, int) and not isinstance(component_raw, bool):
        component_count = component_raw
    else:
        raise ProtocolError(ErrorCode.PARAMETER_ERROR, "n_components 必须为 auto 或整数")
    if component_count < 1 or component_count > maximum:
        raise ProtocolError(ErrorCode.PARAMETER_ERROR, f"n_components 必须在 1 到 {maximum} 之间")
    sxx = np.cov(x, rowvar=False) + regularization * np.eye(x.shape[1])
    syy = np.cov(y, rowvar=False) + regularization * np.eye(y.shape[1])
    sxy = np.cov(x, y, rowvar=False)[: x.shape[1], x.shape[1] :]
    try:
        inv_x = _inverse_sqrt(sxx)
        inv_y = _inverse_sqrt(syy)
        left, _, right_transposed = linalg.svd(inv_x @ sxy @ inv_y)
    except linalg.LinAlgError as exc:
        raise ProtocolError(
            ErrorCode.DATA_ERROR,
            "协方差矩阵奇异，请删除共线字段或设置 regularization",
        ) from exc
    x_weights = inv_x @ left[:, :component_count]
    y_weights = inv_y @ right_transposed.T[:, :component_count]
    x_scores = x @ x_weights
    y_scores = y @ y_weights
    correlations = [
        float(np.corrcoef(x_scores[:, index], y_scores[:, index])[0, 1])
        for index in range(component_count)
    ]
    x_loadings = _loadings(x, x_scores)
    y_loadings = _loadings(y, y_scores)
    x_cross = _loadings(x, y_scores)
    y_cross = _loadings(y, x_scores)
    loading_rows: list[dict[str, JsonValue]] = []
    for name, matrix, set_name, loading_type in (
        (x_columns, x_loadings, "X", "loading"),
        (y_columns, y_loadings, "Y", "loading"),
        (x_columns, x_cross, "X", "cross_loading"),
        (y_columns, y_cross, "Y", "cross_loading"),
    ):
        for variable_index, variable in enumerate(name):
            for component_index in range(component_count):
                loading_rows.append(
                    {
                        "set": set_name,
                        "variable": variable,
                        "component": component_index + 1,
                        "type": loading_type,
                        "value": float(matrix[variable_index, component_index]),
                    }
                )
    significance_rows = []
    for start in range(component_count):
        wilks = float(np.prod(1 - np.square(correlations[start:])))
        coefficient = sample_count - 1 - (len(x_columns) + len(y_columns) + 1) / 2
        chi_square = float(-coefficient * np.log(max(wilks, np.finfo(float).tiny)))
        degrees = (len(x_columns) - start) * (len(y_columns) - start)
        significance_rows.append(
            {
                "from_component": start + 1,
                "wilks_lambda": wilks,
                "chi_square": chi_square,
                "degrees_of_freedom": degrees,
                "p_value": float(stats.chi2.sf(chi_square, degrees)),
            }
        )
    correlation_frame = pd.DataFrame(
        {
            "component": range(1, component_count + 1),
            "canonical_correlation": correlations,
            "x_redundancy": np.mean(np.square(x_cross), axis=0),
            "y_redundancy": np.mean(np.square(y_cross), axis=0),
        }
    )
    warnings = []
    if sample_count < 10 * total_variables:
        warnings.append(
            WarningItem(
                code="LIMITED_SAMPLE_SIZE",
                message="有效样本数低于总变量数的 10 倍，典型载荷可能不稳定",
            )
        )
    return AlgorithmResult(
        metrics=[
            Metric(
                id="component_count", display_name="典型变量对数", value=component_count, unit="对"
            ),
            Metric(id="first_correlation", display_name="第一典型相关系数", value=correlations[0]),
        ],
        tables=[
            table_from_frame("canonical_correlations", "典型相关与冗余度", correlation_frame),
            table_from_frame(
                "canonical_loadings", "典型载荷与交叉载荷", pd.DataFrame(loading_rows)
            ),
            table_from_frame("wilks_tests", "Wilks Lambda 检验", pd.DataFrame(significance_rows)),
        ],
        charts=[
            ChartSpec(
                id="canonical_scores",
                display_name="第一典型变量得分",
                chart_type="scatter",
                data={"x": x_scores[:, 0], "y": y_scores[:, 0]},
            )
        ],
        warnings=warnings,
        metadata={
            "method": "canonical_correlation.basic",
            "sample_count": sample_count,
            "dropped_rows": dropped,
            "standardize": standardize,
            "regularization": regularization,
        },
    )


def _inverse_sqrt(matrix: np.ndarray) -> np.ndarray:
    """对称正定矩阵的稳定逆平方根。"""

    eigenvalues, eigenvectors = linalg.eigh(matrix)
    if bool((eigenvalues <= np.finfo(float).eps).any()):
        raise linalg.LinAlgError("matrix is singular")
    return np.asarray(
        eigenvectors @ np.diag(1 / np.sqrt(eigenvalues)) @ eigenvectors.T,
        dtype=float,
    )


def _loadings(values: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """计算原变量与典型得分之间的相关载荷。"""

    output = np.empty((values.shape[1], scores.shape[1]), dtype=float)
    for variable_index in range(values.shape[1]):
        for component_index in range(scores.shape[1]):
            output[variable_index, component_index] = np.corrcoef(
                values[:, variable_index], scores[:, component_index]
            )[0, 1]
    return output
