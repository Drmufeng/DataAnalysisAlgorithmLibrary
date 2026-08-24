"""全部算法方法共享的固定数据和正常运行用例。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from pydantic import JsonValue

from algorithm_sdk.models import AlgorithmRequest
from algorithm_sdk.validation import infer_data_type


@dataclass(frozen=True, slots=True)
class OperationCase:
    """一个可直接交给执行器的正常算法用例。"""

    package_id: str
    slots: dict[str, list[str]]
    parameters: dict[str, JsonValue]


def make_algorithm_data() -> pd.DataFrame:
    """生成覆盖统计、时序、聚类和分类任务的可复现表格。"""

    size = 90
    row = np.arange(size)
    cluster = row % 3
    x1 = cluster * 4.5 + np.sin(row * 0.37) + (row % 7) * 0.03
    x2 = cluster * 2.7 + np.cos(row * 0.29) + (row % 5) * 0.05
    x3 = (row % 13) + np.sin(row * 0.11) * 0.4
    x4 = (row % 17) * 0.7 + np.cos(row * 0.17) * 0.3
    group_values = np.array(["甲", "乙", "丙"], dtype=object)[cluster]
    ordinal_values = np.array(["低", "中", "高"], dtype=object)[cluster]
    binary_target = np.where((row + row // 3) % 2 == 0, "否", "是")
    multi_target = np.array(["A", "B", "C"], dtype=object)[(row + row // 10) % 3]
    rater_1 = np.array(["低", "中", "高"], dtype=object)[row % 3]
    rater_2 = np.array(["低", "中", "高"], dtype=object)[(row + (row % 10 == 0)) % 3]
    rater_3 = np.array(["低", "中", "高"], dtype=object)[(row + (row % 15 == 0)) % 3]
    outlier = np.sin(row * 0.31) * 2 + row % 5
    outlier[-1] = 100.0
    return pd.DataFrame(
        {
            "id": [f"S{value + 1:03d}" for value in row],
            "time": pd.date_range("2026-01-01", periods=size, freq="D"),
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4,
            "group": group_values,
            "ordinal": ordinal_values,
            "binary_target": binary_target,
            "multi_target": multi_target,
            "condition_a": np.where(row % 2 == 0, "是", "否"),
            "condition_b": np.where((row + row // 4) % 2 == 0, "是", "否"),
            "condition_c": np.where((row + row // 5) % 2 == 0, "是", "否"),
            "rater_1": rater_1,
            "rater_2": rater_2,
            "rater_3": rater_3,
            "rank_1": (row % 15) + 1,
            "rank_2": ((row + row // 8) % 15) + 1,
            "rank_3": ((row + row // 11) % 15) + 1,
            "count": row % 7,
            "weight": (row % 3) + 1,
            "signal": np.sin(2 * np.pi * row / 12) + 0.05 * np.cos(row * 0.7),
            "gm_y": 10 * np.power(1.02, row),
            "outlier_value": outlier,
            "y_cont": 5 + 1.7 * x1 - 0.8 * x2 + 0.35 * x3 + np.sin(row * 0.23),
        }
    )


def make_field_metadata(data: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """为固定表格生成平台侧字段元数据快照。"""

    categorical = {
        "id",
        "time",
        "group",
        "ordinal",
        "binary_target",
        "multi_target",
        "condition_a",
        "condition_b",
        "condition_c",
        "rater_1",
        "rater_2",
        "rater_3",
    }
    ordinal = {"ordinal", "rater_1", "rater_2", "rater_3"}
    metadata: dict[str, dict[str, Any]] = {}
    for column in data.columns:
        is_categorical = column in categorical
        item: dict[str, Any] = {
            "data_type": infer_data_type(data[column]).value,
            "analysis_type": "categorical" if is_categorical else "quantitative",
            "measurement_level": (
                "ordinal" if column in ordinal else ("nominal" if is_categorical else "scale")
            ),
            "analysis_type_source": "user",
        }
        if column in ordinal:
            item["category_order"] = ["低", "中", "高"]
        if column == "time":
            item["semantic_role"] = "time"
        if column == "id":
            item["semantic_role"] = "identifier"
        metadata[column] = item
    return metadata


def make_request(operation_key: str, case: OperationCase, data: pd.DataFrame) -> AlgorithmRequest:
    """根据用例构造包含字段标签快照的请求。"""

    return AlgorithmRequest.model_validate(
        {
            "operation_key": operation_key,
            "slots": case.slots,
            "parameters": case.parameters,
            "context": {
                "run_id": f"test-{operation_key}",
                "locale": "zh-CN",
                "random_seed": 42,
                "protocol_version": "1.0",
                "field_metadata_revision": 1,
                "field_metadata": make_field_metadata(data),
            },
        }
    )


OPERATION_CASES: dict[str, OperationCase] = {
    "data_overview.basic": OperationCase(
        "descriptive_analysis", {"columns": ["x1", "group", "time"]}, {}
    ),
    "frequency_analysis.exact": OperationCase(
        "descriptive_analysis", {"columns": ["group", "x1"]}, {}
    ),
    "frequency_analysis.binned": OperationCase(
        "descriptive_analysis", {"columns": ["x1", "x2"]}, {"bin_count": 6}
    ),
    "contingency_analysis.crosstab": OperationCase(
        "descriptive_analysis",
        {"group_column": ["group"], "x_columns": ["ordinal", "x1"]},
        {"binning": {"x1": {"bin_count": 4}}},
    ),
    "descriptive_statistics.basic": OperationCase(
        "descriptive_analysis", {"x_columns": ["x1", "x2", "x3"]}, {}
    ),
    "group_summary.aggregate": OperationCase(
        "descriptive_analysis",
        {"group_columns": ["group"], "value_columns": ["x1", "x2"]},
        {"aggregations": ["count", "mean", "median", "min", "max", "sum", "std"]},
    ),
    "normality_test.auto": OperationCase("descriptive_analysis", {"x_columns": ["x1"]}, {}),
    "normality_test.shapiro": OperationCase("descriptive_analysis", {"x_columns": ["x1"]}, {}),
    "normality_test.dagostino": OperationCase("descriptive_analysis", {"x_columns": ["x1"]}, {}),
    "normality_test.anderson": OperationCase("descriptive_analysis", {"x_columns": ["x1"]}, {}),
    "data_encoding.manual": OperationCase(
        "data_processing", {"source_column": ["group"]}, {"mapping": {"甲": 1, "乙": 2, "丙": 3}}
    ),
    "data_encoding.range": OperationCase(
        "data_processing",
        {"source_column": ["x3"]},
        {
            "ranges": [
                {"maximum": 4, "value": "低"},
                {"minimum": 4, "maximum": 9, "value": "中"},
                {"minimum": 9, "value": "高"},
            ]
        },
    ),
    "data_encoding.auto_bin": OperationCase(
        "data_processing", {"source_column": ["x3"]}, {"bin_count": 4}
    ),
    "outlier_handling.auto": OperationCase(
        "data_processing", {"x_columns": ["outlier_value"]}, {"action": "set_null"}
    ),
    "outlier_handling.custom": OperationCase(
        "data_processing",
        {"x_columns": ["outlier_value"]},
        {"bounds": {"outlier_value": {"minimum": -10, "maximum": 10}}, "action": "median"},
    ),
    "outlier_handling.mad": OperationCase(
        "data_processing", {"x_columns": ["outlier_value"]}, {"action": "median"}
    ),
    "outlier_handling.iqr": OperationCase(
        "data_processing", {"x_columns": ["outlier_value"]}, {"action": "winsorize"}
    ),
    "outlier_handling.three_sigma": OperationCase(
        "data_processing", {"x_columns": ["outlier_value"]}, {"action": "mean"}
    ),
    "invalid_sample_handling.rule_based": OperationCase(
        "data_processing",
        {"check_columns": ["x1", "x2"]},
        {
            "rules": [{"type": "value_range", "columns": ["x1"], "minimum": -20, "maximum": 20}],
            "action": "flag",
        },
    ),
    "invalid_sample_handling.custom_rules": OperationCase(
        "data_processing",
        {"check_columns": ["x1", "x2"]},
        {
            "rules": [
                {"type": "value_range", "columns": ["x1"], "minimum": -20, "maximum": 20},
                {"type": "cross_field_rule", "columns": ["x1", "x2"], "operator": "ge"},
            ],
            "combination": "any",
            "action": "flag",
        },
    ),
    "time_series_window.sliding": OperationCase(
        "time_signal_processing",
        {"value_column": ["signal"], "order_column": ["time"]},
        {"window_size": 6, "step_size": 2, "forecast_horizon": 1},
    ),
    "data_downsampling.periodic_sample": OperationCase(
        "time_signal_processing",
        {"value_columns": ["signal", "x1"], "order_column": ["time"]},
        {"factor": 3, "offset": 1},
    ),
    "data_downsampling.block_aggregate": OperationCase(
        "time_signal_processing",
        {"value_columns": ["signal", "x1"], "order_column": ["time"]},
        {"factor": 3, "aggregation": "mean"},
    ),
    "data_downsampling.filtered_decimate": OperationCase(
        "time_signal_processing",
        {"value_columns": ["signal"], "order_column": ["time"]},
        {"factor": 3, "filter_type": "iir"},
    ),
    "association_auto_solver.auto": OperationCase(
        "statistical_association", {"analysis_columns": ["x1", "x2", "group"]}, {}
    ),
    "correlation_analysis.pearson": OperationCase(
        "statistical_association", {"x_columns": ["x1", "x2", "binary_target"]}, {}
    ),
    "correlation_analysis.spearman": OperationCase(
        "statistical_association", {"x_columns": ["x1", "ordinal"]}, {}
    ),
    "correlation_analysis.kendall_tau_b": OperationCase(
        "statistical_association", {"x_columns": ["x1", "ordinal"]}, {}
    ),
    "cochran_q.related_binary": OperationCase(
        "statistical_association",
        {"condition_columns": ["condition_a", "condition_b", "condition_c"]},
        {"positive_value": "是"},
    ),
    "kappa_agreement.cohen": OperationCase(
        "statistical_association",
        {"rater_columns": ["rater_1", "rater_2"], "frequency_weight": ["weight"]},
        {},
    ),
    "kappa_agreement.linear_weighted": OperationCase(
        "statistical_association",
        {"rater_columns": ["rater_1", "rater_2"], "frequency_weight": []},
        {},
    ),
    "kappa_agreement.quadratic_weighted": OperationCase(
        "statistical_association",
        {"rater_columns": ["rater_1", "rater_2"], "frequency_weight": []},
        {},
    ),
    "kappa_agreement.fleiss": OperationCase(
        "statistical_association",
        {"rater_columns": ["rater_1", "rater_2", "rater_3"], "frequency_weight": ["weight"]},
        {},
    ),
    "kendall_w.concordance": OperationCase(
        "statistical_association", {"rater_columns": ["rank_1", "rank_2", "rank_3"]}, {}
    ),
    "canonical_correlation.basic": OperationCase(
        "statistical_association",
        {"y_set": ["x3", "x4"], "x_set": ["x1", "x2"]},
        {"regularization": 0.001},
    ),
    "poisson_goodness_of_fit.chi_square": OperationCase(
        "statistical_association",
        {"count_column": ["count"], "frequency_weight": ["weight"]},
        {"lambda_mode": "estimate", "min_expected_count": 3},
    ),
    "runs_test.median_cut": OperationCase(
        "statistical_association", {"x_columns": ["signal"], "order_column": ["time"]}, {}
    ),
    "runs_test.custom_cut": OperationCase(
        "statistical_association",
        {"x_columns": ["signal"], "order_column": ["time"]},
        {"cut_value": 0},
    ),
    "runs_test.binary": OperationCase(
        "statistical_association", {"x_columns": ["binary_target"], "order_column": ["time"]}, {}
    ),
    "linear_regression.ols": OperationCase(
        "statistical_models", {"y_column": ["y_cont"], "x_columns": ["x1", "x2", "group"]}, {}
    ),
    "ridge_regression.fixed_alpha": OperationCase(
        "statistical_models",
        {"y_column": ["y_cont"], "x_columns": ["x1", "x2", "group"]},
        {"alpha": 0.5},
    ),
    "ridge_regression.cross_validated": OperationCase(
        "statistical_models",
        {"y_column": ["y_cont"], "x_columns": ["x1", "x2", "group"]},
        {"alpha_grid": [0.01, 0.1, 1.0], "cv_folds": 3},
    ),
    "hierarchical_regression.blockwise_ols": OperationCase(
        "statistical_models",
        {
            "y_column": ["y_cont"],
            "block_1": ["x1"],
            "block_2": ["x2"],
            "block_3": ["x3"],
            "block_4": [],
            "block_5": [],
        },
        {},
    ),
    "grey_forecast_gm11.basic": OperationCase(
        "statistical_models", {"y_column": ["gm_y"], "time_column": ["time"]}, {"forecast_steps": 3}
    ),
    "principal_component_analysis.fixed_components": OperationCase(
        "statistical_models",
        {"x_columns": ["x1", "x2", "x3", "x4"], "row_index": ["id"]},
        {"n_components": 2, "rotation": "varimax"},
    ),
    "principal_component_analysis.eigenvalue_threshold": OperationCase(
        "statistical_models",
        {"x_columns": ["x1", "x2", "x3", "x4"], "row_index": ["id"]},
        {"eigenvalue_threshold": 0.5},
    ),
    "kmeans_clustering.standard": OperationCase(
        "clustering_models", {"x_columns": ["x1", "x2", "group"]}, {"n_clusters": 3, "n_init": 5}
    ),
    "dbscan_clustering.standard": OperationCase(
        "clustering_models", {"x_columns": ["x1", "x2"]}, {"eps": 0.5, "min_samples": 3}
    ),
    "logistic_regression.binary_gradient_descent": OperationCase(
        "ml_classification",
        {"y_column": ["binary_target"], "x_columns": ["x1", "x2", "group"]},
        {"train_ratio": 0.8, "cv_folds": 3},
    ),
    "naive_bayes_classification.gaussian": OperationCase(
        "ml_classification",
        {"y_column": ["multi_target"], "x_columns": ["x1", "x2", "x3"]},
        {"train_ratio": 0.8},
    ),
    "naive_bayes_classification.categorical": OperationCase(
        "ml_classification",
        {"y_column": ["multi_target"], "x_columns": ["group", "ordinal"]},
        {"train_ratio": 0.8},
    ),
    "naive_bayes_classification.mixed": OperationCase(
        "ml_classification",
        {"y_column": ["multi_target"], "x_columns": ["x1", "x2", "group"]},
        {"train_ratio": 0.8},
    ),
    "lightgbm_classification.standard": OperationCase(
        "lightgbm_classification",
        {"y_column": ["multi_target"], "x_columns": ["x1", "x2", "group"]},
        {"train_ratio": 0.8, "n_estimators": 30, "min_child_samples": 5},
    ),
}


def _load_studio_operation_cases() -> None:
    """加载工作台为新方法生成的最小真实运行用例。"""

    cases_path = Path(__file__).with_name("studio_operation_cases.json")
    if not cases_path.is_file():
        return
    raw_payload: object = json.loads(cases_path.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, dict):
        raise RuntimeError("studio_operation_cases.json 根节点必须是对象")
    payload = cast(dict[str, object], raw_payload)
    for operation_key, raw_case in payload.items():
        if operation_key in OPERATION_CASES:
            raise RuntimeError(f"工作台运行用例重复：{operation_key}")
        if not isinstance(raw_case, dict):
            raise RuntimeError(f"工作台运行用例格式错误：{operation_key}")
        case = cast(dict[str, object], raw_case)
        package_id = case.get("package_id")
        raw_slots = case.get("slots")
        raw_parameters = case.get("parameters")
        if not isinstance(package_id, str) or not isinstance(raw_slots, dict):
            raise RuntimeError(f"工作台运行用例缺少 package_id 或 slots：{operation_key}")
        slots: dict[str, list[str]] = {}
        for slot_id, raw_columns in cast(dict[str, object], raw_slots).items():
            if not isinstance(raw_columns, list) or not all(
                isinstance(column, str) for column in raw_columns
            ):
                raise RuntimeError(f"工作台运行用例槽位格式错误：{operation_key}.{slot_id}")
            slots[slot_id] = cast(list[str], raw_columns)
        if not isinstance(raw_parameters, dict):
            raise RuntimeError(f"工作台运行用例 parameters 格式错误：{operation_key}")
        OPERATION_CASES[operation_key] = OperationCase(
            package_id=package_id,
            slots=slots,
            parameters=cast(dict[str, JsonValue], raw_parameters),
        )


_load_studio_operation_cases()
