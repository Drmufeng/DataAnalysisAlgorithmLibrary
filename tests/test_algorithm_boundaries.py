"""算法字段标签、异常边界、版本语义和可复现性测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from algorithm_sdk.errors import ErrorCode, ProtocolError
from algorithm_sdk.models import AlgorithmResult, FieldMetadata
from algorithm_sdk.serialization import result_to_payload
from runner_contract.runner import run_operation
from tests.operation_cases import (
    OPERATION_CASES,
    OperationCase,
    make_algorithm_data,
    make_request,
)

PROJECT_ROOT = Path(__file__).parents[1]
PACKAGES_ROOT = PROJECT_ROOT / "packages"


def _run(operation_key: str, data: pd.DataFrame | None = None) -> AlgorithmResult:
    """运行固定用例并减少边界测试样板代码。"""

    actual_data = make_algorithm_data() if data is None else data
    case = OPERATION_CASES[operation_key]
    return run_operation(
        PACKAGES_ROOT / case.package_id,
        actual_data,
        make_request(operation_key, case, actual_data),
    )


def test_analysis_type_can_switch_without_changing_physical_type() -> None:
    """同一实际整数列可在平台切换分析标签，物理类型保持不变。"""

    categorical = FieldMetadata.model_validate(
        {
            "data_type": "integer",
            "analysis_type": "categorical",
            "measurement_level": "nominal",
            "analysis_type_source": "user",
        }
    )
    quantitative = FieldMetadata.model_validate(
        {
            "data_type": "integer",
            "analysis_type": "quantitative",
            "measurement_level": "scale",
            "analysis_type_source": "user",
        }
    )
    assert categorical.data_type == quantitative.data_type
    assert categorical.analysis_type != quantitative.analysis_type


def test_inconsistent_measurement_level_is_rejected() -> None:
    """平台不能把定量和 nominal 等互相矛盾的标签写入运行快照。"""

    with pytest.raises(ValueError):
        FieldMetadata.model_validate(
            {
                "data_type": "number",
                "analysis_type": "quantitative",
                "measurement_level": "nominal",
            }
        )


def test_quantitative_slot_rejects_categorical_label() -> None:
    """算法槽位应按用户当前标签拦截不适用字段。"""

    data = make_algorithm_data()
    case = OperationCase("descriptive_analysis", {"x_columns": ["group"]}, {})
    request = make_request("descriptive_statistics.basic", case, data)
    with pytest.raises(ProtocolError) as caught:
        run_operation(PACKAGES_ROOT / case.package_id, data, request)
    assert caught.value.code == ErrorCode.COLUMN_ERROR


def test_string_column_cannot_become_computable_only_by_toggling_label() -> None:
    """文本列即使被标为定量，也必须先真正编码才能进入数值算法。"""

    data = make_algorithm_data()
    case = OperationCase("descriptive_analysis", {"x_columns": ["group"]}, {})
    request = make_request("descriptive_statistics.basic", case, data)
    metadata = request.context.field_metadata.copy()
    group_payload = metadata["group"].model_dump(mode="json")
    metadata["group"] = FieldMetadata.model_validate(
        {**group_payload, "analysis_type": "quantitative", "measurement_level": "scale"}
    )
    request = request.model_copy(
        update={"context": request.context.model_copy(update={"field_metadata": metadata})}
    )
    with pytest.raises(ProtocolError) as caught:
        run_operation(PACKAGES_ROOT / case.package_id, data, request)
    assert caught.value.code == ErrorCode.COLUMN_ERROR
    assert "实际类型" in caught.value.message


def test_text_ordinal_correlation_requires_category_order() -> None:
    """文本有序定类不能按字典序偷偷编码。"""

    data = make_algorithm_data()
    case = OPERATION_CASES["correlation_analysis.spearman"]
    request = make_request("correlation_analysis.spearman", case, data)
    metadata = request.context.field_metadata.copy()
    metadata["ordinal"] = metadata["ordinal"].model_copy(update={"category_order": []})
    request = request.model_copy(
        update={"context": request.context.model_copy(update={"field_metadata": metadata})}
    )
    with pytest.raises(ProtocolError) as caught:
        run_operation(PACKAGES_ROOT / case.package_id, data, request)
    assert caught.value.code == ErrorCode.COLUMN_ERROR
    assert "类别顺序" in caught.value.message


def test_missing_field_metadata_is_rejected_for_every_algorithm() -> None:
    """执行器不允许脱离平台字段快照自行猜测定类或定量。"""

    data = make_algorithm_data()
    case = OPERATION_CASES["linear_regression.ols"]
    request = make_request("linear_regression.ols", case, data)
    request = request.model_copy(
        update={"context": request.context.model_copy(update={"field_metadata": {}})}
    )
    with pytest.raises(ProtocolError) as caught:
        run_operation(PACKAGES_ROOT / case.package_id, data, request)
    assert caught.value.code == ErrorCode.COLUMN_ERROR


def test_unknown_parameter_returns_parameter_error() -> None:
    """拼写错误或过期参数不能被算法静默忽略。"""

    data = make_algorithm_data()
    case = OPERATION_CASES["descriptive_statistics.basic"]
    request = make_request("descriptive_statistics.basic", case, data).model_copy(
        update={"parameters": {"unknown_parameter": 1}}
    )
    with pytest.raises(ProtocolError) as caught:
        run_operation(PACKAGES_ROOT / case.package_id, data, request)
    assert caught.value.code == ErrorCode.PARAMETER_ERROR


def test_non_finite_model_input_returns_data_error() -> None:
    """无穷值应返回稳定数据错误，而不是底层库异常。"""

    data = make_algorithm_data()
    data.loc[0, "x1"] = np.inf
    case = OPERATION_CASES["linear_regression.ols"]
    with pytest.raises(ProtocolError) as caught:
        run_operation(
            PACKAGES_ROOT / case.package_id,
            data,
            make_request("linear_regression.ols", case, data),
        )
    assert caught.value.code == ErrorCode.DATA_ERROR


def test_constant_pca_column_returns_data_error() -> None:
    """PCA 常数列必须在拟合前明确拒绝。"""

    data = make_algorithm_data()
    data["x1"] = 1.0
    case = OPERATION_CASES["principal_component_analysis.fixed_components"]
    with pytest.raises(ProtocolError) as caught:
        run_operation(
            PACKAGES_ROOT / case.package_id,
            data,
            make_request("principal_component_analysis.fixed_components", case, data),
        )
    assert caught.value.code == ErrorCode.DATA_ERROR


def test_small_sample_normality_test_is_rejected() -> None:
    """D'Agostino K² 的最小样本数限制应在统一校验阶段生效。"""

    data = make_algorithm_data().head(7).copy()
    case = OPERATION_CASES["normality_test.dagostino"]
    with pytest.raises(ProtocolError) as caught:
        run_operation(
            PACKAGES_ROOT / case.package_id,
            data,
            make_request("normality_test.dagostino", case, data),
        )
    assert caught.value.code == ErrorCode.DATA_ERROR


def test_clustering_can_return_analysis_without_generating_a_label_version() -> None:
    """关闭类别字段时聚类只返回分析结果，不创建数据版本。"""

    data = make_algorithm_data()
    case = OPERATION_CASES["kmeans_clustering.standard"]
    parameters = {**case.parameters, "generate_label_column": False}
    changed_case = OperationCase(case.package_id, case.slots, parameters)
    result = run_operation(
        PACKAGES_ROOT / case.package_id,
        data,
        make_request("kmeans_clustering.standard", changed_case, data),
    )
    assert result.data is None
    assert result.metadata["changed_columns"] == []


@pytest.mark.parametrize(
    "operation_key",
    [
        "outlier_handling.auto",
        "kmeans_clustering.standard",
        "logistic_regression.binary_gradient_descent",
        "lightgbm_classification.standard",
    ],
)
def test_randomized_operations_are_reproducible(operation_key: str) -> None:
    """固定运行种子后，随机算法的结构化结果必须一致。"""

    data = make_algorithm_data()
    case = OPERATION_CASES[operation_key]
    request = make_request(operation_key, case, data)
    first = run_operation(PACKAGES_ROOT / case.package_id, data, request)
    second = run_operation(PACKAGES_ROOT / case.package_id, data, request)
    assert result_to_payload(first) == result_to_payload(second)
    if first.data is not None and second.data is not None:
        assert_frame_equal(first.data, second.data)


def test_lightgbm_binary_and_multiclass_metrics_are_both_available() -> None:
    """LightGBM 第一版同时支持二分类和多分类，而不是仅支持二分类。"""

    data = make_algorithm_data()
    multiclass = _run("lightgbm_classification.standard", data)
    binary_case = OperationCase(
        "lightgbm_classification",
        {"y_column": ["binary_target"], "x_columns": ["x1", "x2", "group"]},
        {"train_ratio": 0.8, "n_estimators": 30, "min_child_samples": 5},
    )
    binary = run_operation(
        PACKAGES_ROOT / binary_case.package_id,
        data,
        make_request("lightgbm_classification.standard", binary_case, data),
    )
    multiclass_ids = {metric.id for metric in multiclass.metrics}
    binary_ids = {metric.id for metric in binary.metrics}
    assert {"roc_auc_ovr_macro", "roc_auc_ovr_micro", "roc_auc_ovr_weighted"} <= multiclass_ids
    assert {"roc_auc", "pr_auc"} <= binary_ids


def test_processing_result_never_mutates_source_data() -> None:
    """数据处理算法必须基于副本输出，原始数据版本保持不变。"""

    data = make_algorithm_data()
    original = data.copy(deep=True)
    result = _run("outlier_handling.custom", data)
    assert_frame_equal(data, original)
    assert result.data is not None
    assert result.data is not data
