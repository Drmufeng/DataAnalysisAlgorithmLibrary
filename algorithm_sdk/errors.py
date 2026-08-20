"""算法协议使用的结构化错误。"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum

from pydantic import JsonValue

from algorithm_sdk.json_compat import to_json_compatible


class ErrorCode(StrEnum):
    """平台能够稳定识别的错误类型。"""

    PARAMETER_ERROR = "PARAMETER_ERROR"
    COLUMN_ERROR = "COLUMN_ERROR"
    DATA_ERROR = "DATA_ERROR"
    DEPENDENCY_ERROR = "DEPENDENCY_ERROR"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"
    CANCELLED = "CANCELLED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ProtocolError(Exception):
    """可安全返回给平台的算法协议异常。"""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        converted_details = to_json_compatible(details or {})
        if not isinstance(converted_details, dict):
            raise TypeError("错误 details 必须是 JSON 对象")
        self.details = converted_details

    def to_payload(self) -> dict[str, JsonValue]:
        """生成不包含堆栈和本机路径的错误数据。"""

        return {
            "code": self.code.value,
            "message": self.message,
            "details": self.details,
        }
