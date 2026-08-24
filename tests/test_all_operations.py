"""全部 53 个算法方法的统一执行和协议测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from algorithm_cli.validator import validate_package
from algorithm_sdk.manifest import load_manifest
from algorithm_sdk.serialization import result_to_payload
from runner_contract.runner import run_operation
from tests.operation_cases import OPERATION_CASES, make_algorithm_data, make_request

PROJECT_ROOT = Path(__file__).parents[1]
PACKAGES_ROOT = PROJECT_ROOT / "packages"
LEGACY_DATA_OPERATION_PREFIXES = (
    "data_encoding.",
    "outlier_handling.",
    "invalid_sample_handling.",
    "time_series_window.",
    "data_downsampling.",
    "kmeans_clustering.",
    "dbscan_clustering.",
)


def _manifest_operations() -> dict[str, bool]:
    """读取全部方法及其是否产生数据版本。"""

    operations: dict[str, bool] = {}
    for package_dir in PACKAGES_ROOT.iterdir():
        if not package_dir.is_dir() or not (package_dir / "manifest.json").is_file():
            continue
        manifest = load_manifest(package_dir)
        for algorithm in manifest.algorithms:
            for method in algorithm.methods:
                operation_key = f"{algorithm.algorithm_id}.{method.method_id}"
                operations[operation_key] = method.output.get(
                    "produces_data_version"
                ) is True or operation_key.startswith(LEGACY_DATA_OPERATION_PREFIXES)
    return operations


def test_every_manifest_operation_has_a_runtime_case() -> None:
    """清单新增方法时必须同步增加运行测试，防止只登记不执行。"""

    assert set(_manifest_operations()) == set(OPERATION_CASES)


@pytest.mark.parametrize(
    "package_id", sorted({case.package_id for case in OPERATION_CASES.values()})
)
def test_every_package_passes_full_validation(package_id: str) -> None:
    """全部算法包应通过文件、Schema、依赖和入口校验。"""

    report = validate_package(PACKAGES_ROOT / package_id)
    assert report.is_valid, [issue.to_payload() for issue in report.issues]


@pytest.mark.parametrize("operation_key", sorted(OPERATION_CASES))
def test_every_operation_runs_and_returns_strict_json(operation_key: str) -> None:
    """每个 operation_key 都应真实执行并返回严格 JSON 结果。"""

    data = make_algorithm_data()
    original = data.copy(deep=True)
    case = OPERATION_CASES[operation_key]
    result = run_operation(
        PACKAGES_ROOT / case.package_id,
        data,
        make_request(operation_key, case, data),
    )
    assert result.status == "success"
    assert result.metrics or result.tables
    assert_frame_equal(data, original)
    payload = result_to_payload(result)
    json.dumps(payload, ensure_ascii=False, allow_nan=False)


@pytest.mark.parametrize(
    "operation_key",
    [key for key, produces_data in sorted(_manifest_operations().items()) if produces_data],
)
def test_data_changing_operations_return_a_new_frame(operation_key: str) -> None:
    """改变数据的操作必须返回新 DataFrame，平台据此创建数据版本。"""

    data = make_algorithm_data()
    case = OPERATION_CASES[operation_key]
    result = run_operation(
        PACKAGES_ROOT / case.package_id,
        data,
        make_request(operation_key, case, data),
    )
    assert isinstance(result.data, pd.DataFrame)
    assert result.data is not data


@pytest.mark.parametrize(
    "operation_key",
    [key for key, produces_data in sorted(_manifest_operations().items()) if not produces_data],
)
def test_analysis_operations_do_not_return_a_data_version(operation_key: str) -> None:
    """纯分析方法只返回结构化结果，不应伪造数据版本。"""

    data = make_algorithm_data()
    case = OPERATION_CASES[operation_key]
    result = run_operation(
        PACKAGES_ROOT / case.package_id,
        data,
        make_request(operation_key, case, data),
    )
    assert result.data is None
