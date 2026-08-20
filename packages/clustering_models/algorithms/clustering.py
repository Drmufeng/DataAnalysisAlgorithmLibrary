"""K-Means 和 DBSCAN 聚类。"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from pydantic import JsonValue
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from algorithm_sdk import AlgorithmResult, ChartSpec, Metric, WarningItem
from algorithm_sdk.algorithm_utils import (
    complete_cases,
    encode_features,
    get_bool_parameter,
    get_float_parameter,
    get_int_parameter,
    get_slot,
    get_str_parameter,
    random_seed_from_context,
    table_from_frame,
    unique_column_name,
)
from algorithm_sdk.errors import ErrorCode, ProtocolError


def run_kmeans(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """执行可复现 K-Means，并默认生成类别字段。"""

    columns = get_slot(slots, "x_columns")
    frame, dropped = complete_cases(data, columns)
    features, encoding = encode_features(frame, columns, context, drop_first=False)
    numeric = features.to_numpy(dtype=float)
    feature_names = [str(column) for column in features.columns]
    _check_cluster_input(numeric, feature_names)
    n_clusters = get_int_parameter(parameters, "n_clusters", 3)
    if n_clusters >= len(frame.index):
        raise ProtocolError(ErrorCode.PARAMETER_ERROR, "n_clusters 必须小于有效样本数")
    standardize = get_bool_parameter(parameters, "standardize", True)
    scaler = StandardScaler() if standardize else None
    transformed = scaler.fit_transform(numeric) if scaler else numeric
    init = get_str_parameter(parameters, "init", "k-means++")
    model = KMeans(
        n_clusters=n_clusters,
        init=init,
        n_init=get_int_parameter(parameters, "n_init", 10),
        max_iter=get_int_parameter(parameters, "max_iter", 300),
        random_state=random_seed_from_context(context),
    ).fit(transformed)
    labels = model.labels_
    centers = scaler.inverse_transform(model.cluster_centers_) if scaler else model.cluster_centers_
    center_frame = pd.DataFrame(centers, columns=feature_names)
    center_frame.insert(0, "cluster", range(n_clusters))
    count_frame = (
        pd.Series(labels)
        .value_counts()
        .sort_index()
        .rename_axis("cluster")
        .reset_index(name="sample_count")
    )
    silhouette = float(silhouette_score(transformed, labels)) if n_clusters > 1 else None
    output = _attach_labels(data, frame.index, labels, parameters, "cluster_kmeans")
    changed_columns = (
        [column for column in output.columns if column not in data.columns]
        if output is not None
        else []
    )
    visualization = _visualization(transformed, labels, frame.index)
    return AlgorithmResult(
        data=output,
        metrics=[
            Metric(id="inertia", display_name="簇内平方和", value=float(model.inertia_)),
            Metric(id="silhouette", display_name="轮廓系数", value=silhouette),
            Metric(id="cluster_count", display_name="聚类数量", value=n_clusters, unit="类"),
        ],
        tables=[
            table_from_frame("cluster_centers", "聚类中心", center_frame),
            table_from_frame("cluster_counts", "各类样本数", count_frame),
        ],
        charts=[
            ChartSpec(
                id="cluster_projection",
                display_name="聚类二维投影",
                chart_type="scatter",
                data=visualization,
            )
        ],
        warnings=_cluster_warnings(dropped, transformed.shape[1]),
        metadata={
            "method": "kmeans_clustering.standard",
            "sample_count": len(frame.index),
            "dropped_rows": dropped,
            "standardize": standardize,
            "categorical_encoding": encoding,
            "random_seed": random_seed_from_context(context),
            "rows_in": len(data.index),
            "rows_out": len(output.index) if output is not None else len(data.index),
            "changed_columns": changed_columns,
        },
    )


def run_dbscan(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """执行 DBSCAN，噪声标签固定为 -1。"""

    del context
    columns = get_slot(slots, "x_columns")
    frame, dropped = complete_cases(data, columns)
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    _check_cluster_input(numeric, columns)
    standardize = get_bool_parameter(parameters, "standardize", True)
    transformed = StandardScaler().fit_transform(numeric) if standardize else numeric
    metric = get_str_parameter(parameters, "metric", "euclidean")
    metric_name = "cityblock" if metric == "manhattan" else metric
    model = DBSCAN(
        eps=get_float_parameter(parameters, "eps", 0.5),
        min_samples=get_int_parameter(parameters, "min_samples", 5),
        metric=metric_name,
    ).fit(transformed)
    labels = model.labels_
    cluster_labels = sorted(set(labels) - {-1})
    count_frame = (
        pd.Series(labels)
        .value_counts()
        .sort_index()
        .rename_axis("cluster")
        .reset_index(name="sample_count")
    )
    output = _attach_labels(data, frame.index, labels, parameters, "cluster_dbscan")
    changed_columns = (
        [column for column in output.columns if column not in data.columns]
        if output is not None
        else []
    )
    visualization = _visualization(transformed, labels, frame.index)
    warnings = _cluster_warnings(dropped, transformed.shape[1])
    if not cluster_labels:
        warnings.append(
            WarningItem(
                code="NO_CLUSTERS_FOUND", message="当前参数未形成有效簇，所有或大部分样本为噪声点"
            )
        )
    return AlgorithmResult(
        data=output,
        metrics=[
            Metric(
                id="cluster_count", display_name="有效聚类数", value=len(cluster_labels), unit="类"
            ),
            Metric(
                id="noise_count",
                display_name="噪声点数",
                value=int(np.sum(labels == -1)),
                unit="个",
            ),
            Metric(
                id="core_count",
                display_name="核心点数",
                value=len(model.core_sample_indices_),
                unit="个",
            ),
        ],
        tables=[table_from_frame("cluster_counts", "各类样本数", count_frame)],
        charts=[
            ChartSpec(
                id="cluster_projection",
                display_name="聚类二维投影",
                chart_type="scatter",
                data=visualization,
            )
        ],
        warnings=warnings,
        metadata={
            "method": "dbscan_clustering.standard",
            "sample_count": len(frame.index),
            "dropped_rows": dropped,
            "standardize": standardize,
            "metric": metric,
            "rows_in": len(data.index),
            "rows_out": len(output.index) if output is not None else len(data.index),
            "changed_columns": changed_columns,
        },
    )


def _check_cluster_input(values: np.ndarray, columns: list[str]) -> None:
    """聚类输入不能包含非有限值或常数特征。"""

    if len(values) < 2 or not bool(np.isfinite(values).all()):
        raise ProtocolError(ErrorCode.DATA_ERROR, "聚类至少需要两个完整有限样本")
    constant = [
        column for column, std in zip(columns, np.std(values, axis=0), strict=True) if std == 0
    ]
    if constant:
        raise ProtocolError(
            ErrorCode.DATA_ERROR,
            f"聚类字段包含常数特征：{', '.join(constant)}",
            details={"columns": constant},
        )


def _attach_labels(
    data: pd.DataFrame,
    valid_index: pd.Index,
    labels: np.ndarray,
    parameters: Mapping[str, JsonValue],
    preferred: str,
) -> pd.DataFrame | None:
    """默认在数据副本中添加类别标签，也允许用户关闭。"""

    if not get_bool_parameter(parameters, "generate_label_column", True):
        return None
    output = data.copy(deep=True)
    configured = get_str_parameter(parameters, "label_column_name", preferred)
    column = unique_column_name(output, configured)
    output[column] = pd.Series(pd.NA, index=output.index, dtype="Int64")
    output.loc[valid_index, column] = labels
    return output


def _visualization(values: np.ndarray, labels: np.ndarray, index: pd.Index) -> dict[str, JsonValue]:
    """将高维特征投影到二维，只用于展示而不参与聚类。"""

    if values.shape[1] >= 2:
        projected = PCA(n_components=2).fit_transform(values)
    else:
        projected = np.column_stack((values[:, 0], np.zeros(len(values))))
    return {
        "x": projected[:, 0],
        "y": projected[:, 1],
        "label": labels,
        "row_index": [str(value) for value in index],
    }


def _cluster_warnings(dropped: int, dimension_count: int) -> list[WarningItem]:
    """构造缺失删除和高维距离警告。"""

    warnings: list[WarningItem] = []
    if dropped:
        warnings.append(
            WarningItem(code="MISSING_ROWS_DROPPED", message=f"聚类前删除了 {dropped} 条缺失记录")
        )
    if dimension_count > 20:
        warnings.append(
            WarningItem(
                code="HIGH_DIMENSION_DISTANCE", message="特征维度超过 20，距离区分度可能下降"
            )
        )
    return warnings
