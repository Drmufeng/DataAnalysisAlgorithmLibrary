"""固定成分数或特征根阈值 PCA，支持 Varimax 旋转。"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from pydantic import JsonValue
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from algorithm_sdk import AlgorithmResult, ChartSpec, Metric, WarningItem
from algorithm_sdk.algorithm_utils import (
    get_bool_parameter,
    get_float_parameter,
    get_int_parameter,
    get_slot,
    get_str_parameter,
    table_from_frame,
)
from algorithm_sdk.errors import ErrorCode, ProtocolError


def run_fixed(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """按固定主成分个数执行 PCA。"""

    return _run(data, slots, parameters, context, "fixed_components")


def run_threshold(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """保留特征根不低于阈值的主成分。"""

    return _run(data, slots, parameters, context, "eigenvalue_threshold")


def _run(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
    method: str,
) -> AlgorithmResult:
    """完成缺失处理、标准化、成分选择、旋转和结果表构造。"""

    del context
    columns = get_slot(slots, "x_columns")
    row_indices = get_slot(slots, "row_index")
    numeric = data[columns].apply(pd.to_numeric, errors="coerce")
    missing = get_str_parameter(parameters, "missing", "listwise")
    if missing == "mean_impute":
        values = SimpleImputer(strategy="mean").fit_transform(numeric)
        index = numeric.index
        dropped = 0
    else:
        complete = numeric.dropna()
        values = complete.to_numpy(dtype=float)
        index = complete.index
        dropped = len(numeric.index) - len(complete.index)
    if values.shape[0] < 3:
        raise ProtocolError(ErrorCode.DATA_ERROR, "PCA 至少需要 3 个完整样本")
    if not bool(np.isfinite(values).all()) or bool((np.std(values, axis=0, ddof=1) == 0).any()):
        raise ProtocolError(ErrorCode.DATA_ERROR, "PCA 不允许非有限值或常数列")
    standardize = get_bool_parameter(parameters, "standardize", True)
    transformed = (
        StandardScaler().fit_transform(values) if standardize else values - values.mean(axis=0)
    )
    full_model = PCA().fit(transformed)
    if method == "fixed_components":
        component_count = get_int_parameter(parameters, "n_components", 2)
    else:
        threshold = get_float_parameter(parameters, "eigenvalue_threshold", 1.0)
        component_count = int(np.sum(full_model.explained_variance_ >= threshold))
        component_count = max(component_count, 1)
    maximum = min(transformed.shape)
    if component_count < 1 or component_count > maximum:
        raise ProtocolError(ErrorCode.PARAMETER_ERROR, f"主成分个数必须在 1 到 {maximum} 之间")
    model = PCA(n_components=component_count).fit(transformed)
    scores = model.transform(transformed)
    loadings = model.components_.T * np.sqrt(model.explained_variance_)
    rotation = get_str_parameter(parameters, "rotation", "none")
    if rotation == "varimax" and component_count > 1:
        rotated_loadings, transform = _varimax(loadings)
        scores = scores @ transform
        loadings = rotated_loadings
    loading_frame = pd.DataFrame(
        loadings,
        index=columns,
        columns=[f"PC{index + 1}" for index in range(component_count)],
    ).reset_index(names="variable")
    score_frame = pd.DataFrame(
        scores, columns=[f"PC{index + 1}" for index in range(component_count)]
    )
    score_frame.insert(
        0,
        "row_index",
        data.loc[index, row_indices[0]].astype(str).to_numpy()
        if row_indices
        else [str(value) for value in index],
    )
    variance_frame = pd.DataFrame(
        {
            "component": range(1, component_count + 1),
            "eigenvalue": model.explained_variance_,
            "explained_variance_ratio": model.explained_variance_ratio_,
            "cumulative_ratio": np.cumsum(model.explained_variance_ratio_),
        }
    )
    warnings = []
    if values.shape[0] < 5 * len(columns):
        warnings.append(
            WarningItem(
                code="LIMITED_SAMPLE_SIZE", message="样本数低于变量数的 5 倍，主成分结构可能不稳定"
            )
        )
    return AlgorithmResult(
        metrics=[
            Metric(
                id="component_count", display_name="保留主成分数", value=component_count, unit="个"
            ),
            Metric(
                id="cumulative_explained_ratio",
                display_name="累计解释方差率",
                value=float(np.sum(model.explained_variance_ratio_)),
            ),
        ],
        tables=[
            table_from_frame("explained_variance", "特征值与解释方差", variance_frame),
            table_from_frame("component_loadings", "主成分载荷", loading_frame),
            table_from_frame("component_scores", "主成分得分", score_frame.head(1000)),
        ],
        charts=[
            ChartSpec(
                id="scree_plot",
                display_name="碎石图",
                chart_type="line",
                data={
                    "component": list(range(1, len(full_model.explained_variance_) + 1)),
                    "eigenvalue": full_model.explained_variance_,
                },
            )
        ],
        warnings=warnings,
        metadata={
            "method": f"principal_component_analysis.{method}",
            "sample_count": values.shape[0],
            "dropped_rows": dropped,
            "standardize": standardize,
            "rotation": rotation,
        },
    )


def _varimax(
    loadings: np.ndarray, gamma: float = 1.0, iterations: int = 50, tolerance: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """正交 Varimax 旋转载荷矩阵。"""

    rows, columns = loadings.shape
    rotation = np.eye(columns)
    previous = 0.0
    for _ in range(iterations):
        transformed = loadings @ rotation
        left, singular, right = np.linalg.svd(
            loadings.T
            @ (
                transformed**3
                - (gamma / rows) * transformed @ np.diag(np.diag(transformed.T @ transformed))
            )
        )
        rotation = left @ right
        current = float(singular.sum())
        if previous and current / previous < 1 + tolerance:
            break
        previous = current
    return loadings @ rotation, rotation
