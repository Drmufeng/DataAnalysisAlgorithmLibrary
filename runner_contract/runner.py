"""单个算法步骤的校验、调用和结果归一化。"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from algorithm_sdk.errors import ErrorCode, ProtocolError
from algorithm_sdk.manifest import load_manifest
from algorithm_sdk.models import AlgorithmRequest, AlgorithmResult
from algorithm_sdk.serialization import result_to_payload
from algorithm_sdk.validation import resolve_parameters, validate_request
from runner_contract.loader import load_algorithm_entry

LOGGER = logging.getLogger(__name__)


def run_operation(
    package_dir: Path,
    data: pd.DataFrame,
    request: AlgorithmRequest,
) -> AlgorithmResult:
    """运行一个算法方法；不修改调用方传入的 DataFrame。"""

    manifest = load_manifest(package_dir)
    if (
        request.context.protocol_version.split(".", maxsplit=1)[0]
        != manifest.protocol_version.split(".", maxsplit=1)[0]
    ):
        raise ProtocolError(
            ErrorCode.EXECUTION_ERROR,
            "运行请求与算法包的协议主版本不兼容",
            details={
                "request_protocol": request.context.protocol_version,
                "package_protocol": manifest.protocol_version,
            },
        )
    method = manifest.find_method(request.operation_key)
    if method is None:
        raise ProtocolError(
            ErrorCode.PARAMETER_ERROR,
            f"算法包不包含方法 {request.operation_key}",
            details={"operation_key": request.operation_key},
        )

    resolved_parameters = resolve_parameters(request.parameters, method)
    resolved_request = request.model_copy(update={"parameters": resolved_parameters})
    validate_request(data, resolved_request, method)
    context_payload = resolved_request.context.model_dump(mode="json")

    try:
        with load_algorithm_entry(package_dir, method.entry) as algorithm:
            # 运行副本隔离算法内部的意外原地修改，原始数据版本始终由平台保留。
            raw_result = algorithm(
                data.copy(deep=True),
                resolved_request.slots,
                resolved_parameters,
                context_payload,
            )
        result = _normalize_result(raw_result)
        _validate_result_data(result)
        result_to_payload(result)
        return result
    except ProtocolError:
        raise
    except Exception as exc:
        LOGGER.exception(
            "算法执行出现未处理异常 run_id=%s operation_key=%s",
            request.context.run_id,
            request.operation_key,
        )
        raise ProtocolError(
            ErrorCode.EXECUTION_ERROR,
            "算法运行失败，请根据运行编号联系管理员查看日志",
            details={"run_id": request.context.run_id},
        ) from exc


def _normalize_result(raw_result: AlgorithmResult | Mapping[str, object]) -> AlgorithmResult:
    """允许算法返回模型或普通映射，但统一转为严格结果模型。"""

    if isinstance(raw_result, AlgorithmResult):
        return raw_result
    try:
        return AlgorithmResult.model_validate(raw_result)
    except ValidationError as exc:
        raise ProtocolError(
            ErrorCode.EXECUTION_ERROR,
            "算法返回结果不符合统一结果协议",
        ) from exc


def _validate_result_data(result: AlgorithmResult) -> None:
    """检查数据处理输出的基本表格合理性。"""

    output_data = result.data
    if output_data is None:
        return
    if not all(isinstance(column, str) for column in output_data.columns):
        raise ProtocolError(ErrorCode.EXECUTION_ERROR, "算法输出字段名必须是字符串")
    if output_data.columns.has_duplicates:
        raise ProtocolError(ErrorCode.EXECUTION_ERROR, "算法输出包含重复字段名")
