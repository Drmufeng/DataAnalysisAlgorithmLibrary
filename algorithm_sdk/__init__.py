"""算法包与执行器共享的公开协议。"""

from algorithm_sdk.errors import ErrorCode, ProtocolError
from algorithm_sdk.manifest import load_library_manifest, load_manifest
from algorithm_sdk.models import (
    AlgorithmRequest,
    AlgorithmResult,
    AnalysisType,
    ChartSpec,
    DataType,
    FieldMetadata,
    LibraryManifest,
    LibraryModuleReference,
    Metric,
    PackageManifest,
    ReportBlockSpec,
    ReportPresentationSpec,
    ResultTable,
    RunContext,
    TableColumn,
    WarningItem,
)
from algorithm_sdk.serialization import result_to_payload, to_json_compatible

__all__ = [
    "AlgorithmRequest",
    "AlgorithmResult",
    "AnalysisType",
    "ChartSpec",
    "DataType",
    "ErrorCode",
    "FieldMetadata",
    "LibraryManifest",
    "LibraryModuleReference",
    "Metric",
    "PackageManifest",
    "ReportBlockSpec",
    "ReportPresentationSpec",
    "ProtocolError",
    "ResultTable",
    "RunContext",
    "TableColumn",
    "WarningItem",
    "load_manifest",
    "load_library_manifest",
    "result_to_payload",
    "to_json_compatible",
]
