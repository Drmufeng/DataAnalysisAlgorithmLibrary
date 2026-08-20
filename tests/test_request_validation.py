"""字段槽位、数据标签和参数校验测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from algorithm_sdk.errors import ErrorCode, ProtocolError
from algorithm_sdk.manifest import load_manifest
from algorithm_sdk.validation import resolve_parameters, validate_request
from tests.helpers import make_overview_request, make_sample_data

PROJECT_ROOT = Path(__file__).parents[1]
PACKAGE_DIR = PROJECT_ROOT / "packages" / "descriptive_analysis"


def test_parameter_defaults_are_resolved() -> None:
    """简单模式省略的参数应使用清单默认值，并记录最终参数。"""

    manifest = load_manifest(PACKAGE_DIR)
    method = manifest.find_method("data_overview.basic")
    assert method is not None
    resolved = resolve_parameters({"preview_rows": 3}, method)
    assert resolved == {
        "preview_rows": 3,
        "include_missing": True,
        "include_unique": True,
        "include_duplicates": True,
    }


def test_missing_field_metadata_is_rejected() -> None:
    """执行器不能在缺少定类定量快照时自行猜测用户意图。"""

    manifest = load_manifest(PACKAGE_DIR)
    method = manifest.find_method("data_overview.basic")
    assert method is not None
    request = make_overview_request()
    context = request.context.model_copy(update={"field_metadata": {}})
    invalid_request = request.model_copy(update={"context": context})
    with pytest.raises(ProtocolError) as caught:
        validate_request(make_sample_data(), invalid_request, method)
    assert caught.value.code == ErrorCode.COLUMN_ERROR
    assert "元数据快照" in caught.value.message


def test_out_of_range_parameter_is_rejected() -> None:
    """预览行数超过清单上限时返回参数错误。"""

    manifest = load_manifest(PACKAGE_DIR)
    method = manifest.find_method("data_overview.basic")
    assert method is not None
    request = make_overview_request().model_copy(update={"parameters": {"preview_rows": 101}})
    with pytest.raises(ProtocolError) as caught:
        validate_request(make_sample_data(), request, method)
    assert caught.value.code == ErrorCode.PARAMETER_ERROR
