"""算法请求、运行结果和算法包清单的数据模型。"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Literal

import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from algorithm_sdk.json_compat import to_json_compatible

IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
OPERATION_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}\.[a-z][a-z0-9_]{0,63}$")
ENTRY_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_.]*:[a-zA-Z_][a-zA-Z0-9_]*$")
SEMVER_PATTERN = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
PROTOCOL_VERSION_PATTERN = re.compile(r"^[1-9]\d*\.\d+$")


class StrictModel(BaseModel):
    """拒绝未知字段的公共模型基类。"""

    model_config = ConfigDict(extra="forbid")


class DataType(StrEnum):
    """平台与算法共同使用的实际数据类型。"""

    INTEGER = "integer"
    NUMBER = "number"
    STRING = "string"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    CATEGORY = "category"
    UNKNOWN = "unknown"


class AnalysisType(StrEnum):
    """用户可以调整、但不会生成新数据版本的分析标签。"""

    CATEGORICAL = "categorical"
    QUANTITATIVE = "quantitative"
    UNKNOWN = "unknown"


class FieldMetadata(StrictModel):
    """一次运行使用的字段元数据快照。"""

    data_type: DataType
    analysis_type: AnalysisType
    measurement_level: Literal["nominal", "ordinal", "scale"] | None = None
    semantic_role: str | None = None
    display_name: str | None = None
    value_labels: dict[str, str] = Field(default_factory=dict)
    category_order: list[str | int | float | bool | None] = Field(default_factory=list)
    unit: str | None = None
    description: str | None = None
    analysis_type_source: Literal["system", "user"] = "system"

    @model_validator(mode="after")
    def validate_measurement_level(self) -> FieldMetadata:
        """定类/定量标签和测量水平必须表达同一种分析含义。"""

        if self.analysis_type == AnalysisType.QUANTITATIVE and self.measurement_level not in {
            None,
            "scale",
        }:
            raise ValueError("定量变量的 measurement_level 必须为 scale")
        if self.analysis_type == AnalysisType.CATEGORICAL and self.measurement_level == "scale":
            raise ValueError("定类变量的 measurement_level 不能为 scale")
        return self


class RunContext(StrictModel):
    """算法运行所需且必须可复现的上下文。"""

    run_id: str = Field(min_length=1, max_length=128)
    step_id: str | None = Field(default=None, max_length=128)
    locale: str = "zh-CN"
    random_seed: int = 42
    protocol_version: str = "1.0"
    field_metadata_revision: int = Field(default=0, ge=0)
    field_metadata: dict[str, FieldMetadata] = Field(default_factory=dict)

    @field_validator("protocol_version")
    @classmethod
    def validate_protocol_version(cls, value: str) -> str:
        """限制为可比较的主版本和次版本格式。"""

        if not PROTOCOL_VERSION_PATTERN.fullmatch(value):
            raise ValueError("协议版本必须使用 '主版本.次版本' 格式")
        return value


class AlgorithmRequest(StrictModel):
    """平台或工作流提交给单个算法步骤的请求。"""

    operation_key: str
    slots: dict[str, list[str]]
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    context: RunContext

    @field_validator("operation_key")
    @classmethod
    def validate_operation_key(cls, value: str) -> str:
        """保证算法和方法组合编号稳定且可寻址。"""

        if not OPERATION_KEY_PATTERN.fullmatch(value):
            raise ValueError("operation_key 必须使用 algorithm_id.method_id 格式")
        return value


JsonScalar = str | int | float | bool | None


class Metric(StrictModel):
    """结果页优先展示的单个关键指标。"""

    id: str
    display_name: str
    value: JsonScalar
    unit: str | None = None

    @field_validator("value", mode="before")
    @classmethod
    def normalize_value(cls, value: object) -> JsonValue:
        """在模型边界接收常见 NumPy 标量和缺失值。"""

        return to_json_compatible(value)


class TableColumn(StrictModel):
    """结构化结果表中的列定义。"""

    id: str
    display_name: str
    data_type: DataType


class ResultTable(StrictModel):
    """算法返回的可序列化统计表。"""

    id: str
    display_name: str
    columns: list[TableColumn]
    rows: list[dict[str, JsonValue]]

    @field_validator("rows", mode="before")
    @classmethod
    def normalize_rows(cls, value: object) -> JsonValue:
        """表格单元格统一转换为严格 JSON 值。"""

        return to_json_compatible(value)


class ChartSpec(StrictModel):
    """与 ECharts 等具体前端库无关的图表数据描述。"""

    id: str
    display_name: str
    chart_type: str
    data: dict[str, JsonValue]
    options: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("data", "options", mode="before")
    @classmethod
    def normalize_chart_values(cls, value: object) -> JsonValue:
        """图表数据不能保留第三方数值对象。"""

        return to_json_compatible(value)


class WarningItem(StrictModel):
    """不会阻止算法成功，但需要用户注意的情况。"""

    code: str
    message: str
    details: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("details", mode="before")
    @classmethod
    def normalize_details(cls, value: object) -> JsonValue:
        """警告明细在进入结果模型时完成 JSON 规范化。"""

        return to_json_compatible(value)


class AlgorithmResult(StrictModel):
    """算法入口必须返回的统一成功结果。"""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    status: Literal["success"] = "success"
    data: pd.DataFrame | None = Field(default=None, exclude=True)
    metrics: list[Metric] = Field(default_factory=list)
    tables: list[ResultTable] = Field(default_factory=list)
    charts: list[ChartSpec] = Field(default_factory=list)
    warnings: list[WarningItem] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def normalize_metadata(cls, value: object) -> JsonValue:
        """运行元数据统一接收时间、NumPy 标量和标准缺失值。"""

        return to_json_compatible(value)


class RuntimeDependency(StrictModel):
    """算法包声明的一个 Python 分发依赖。"""

    name: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=128)


class RuntimeSpec(StrictModel):
    """算法包支持的 Python 和第三方依赖范围。"""

    python: str
    dependencies: list[RuntimeDependency] = Field(default_factory=list)

    @model_validator(mode="after")
    def ensure_unique_dependencies(self) -> RuntimeSpec:
        """同一个依赖只能声明一次，避免版本规则互相覆盖。"""

        normalized = [item.name.lower().replace("_", "-") for item in self.dependencies]
        if len(normalized) != len(set(normalized)):
            raise ValueError("runtime.dependencies 存在重复依赖")
        return self


class ValueRange(StrictModel):
    """字段允许的可选数值范围。"""

    minimum: float | None = None
    maximum: float | None = None
    exclusive_minimum: bool = False
    exclusive_maximum: bool = False

    @model_validator(mode="after")
    def validate_bounds(self) -> ValueRange:
        """最小值不能大于最大值。"""

        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("value_range.minimum 不能大于 maximum")
        return self


class InputSlot(StrictModel):
    """一个算法方法需要用户放入的字段角色。"""

    id: str
    display_name: str
    type: Literal["column", "columns"]
    required: bool
    min_items: int = Field(default=0, ge=0)
    max_items: int | None = Field(default=None, ge=1)
    allowed_data_types: list[DataType]
    allowed_analysis_types: list[AnalysisType]
    allowed_semantic_roles: list[str] | None = None
    min_unique_values: int | None = Field(default=None, ge=0)
    max_unique_values: int | None = Field(default=None, ge=1)
    unique_values: bool = False
    allow_missing: bool = True
    finite_only: bool = False
    value_range: ValueRange | None = None

    @field_validator("id")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        """字段角色编号使用稳定英文标识。"""

        if not IDENTIFIER_PATTERN.fullmatch(value):
            raise ValueError("字段角色 id 格式不正确")
        return value

    @model_validator(mode="after")
    def validate_counts_and_types(self) -> InputSlot:
        """检查数量上下限和允许类型是否完整。"""

        if self.required and self.min_items < 1:
            raise ValueError("required=true 时 min_items 必须至少为 1")
        if self.type == "column" and self.max_items != 1:
            raise ValueError("type=column 时 max_items 必须为 1")
        if self.max_items is not None and self.min_items > self.max_items:
            raise ValueError("min_items 不能大于 max_items")
        if not self.allowed_data_types:
            raise ValueError("allowed_data_types 不能为空")
        if not self.allowed_analysis_types:
            raise ValueError("allowed_analysis_types 不能为空")
        if (
            self.min_unique_values is not None
            and self.max_unique_values is not None
            and self.min_unique_values > self.max_unique_values
        ):
            raise ValueError("min_unique_values 不能大于 max_unique_values")
        return self


ConstraintType = Literal[
    "total_min_items",
    "total_max_items",
    "slots_disjoint",
    "requires_slot",
    "exact_category_count",
    "min_rows",
    "time_order",
    "equal_interval",
]


class InputConstraint(StrictModel):
    """多个字段角色之间的组合限制。"""

    type: ConstraintType
    slots: list[str] = Field(default_factory=list)
    value: int | str | bool | None = None


class InputSpec(StrictModel):
    """一个算法方法的完整表格输入说明。"""

    data_type: Literal["table"] = "table"
    slots: list[InputSlot]
    constraints: list[InputConstraint] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_slots_and_constraints(self) -> InputSpec:
        """检查字段角色唯一性和组合约束引用。"""

        slot_ids = [slot.id for slot in self.slots]
        if len(slot_ids) != len(set(slot_ids)):
            raise ValueError("input.slots 存在重复 id")
        known_slots = set(slot_ids)
        for constraint in self.constraints:
            missing = sorted(set(constraint.slots) - known_slots)
            if missing:
                raise ValueError(f"输入约束引用了不存在的字段角色: {', '.join(missing)}")
        return self


class MethodManifest(StrictModel):
    """可独立执行的算法方法清单。"""

    method_id: str
    display_name: str
    entry: str
    description: str
    input: InputSpec
    parameters_schema: dict[str, JsonValue]
    output: dict[str, JsonValue]
    deterministic: bool

    @field_validator("method_id")
    @classmethod
    def validate_method_id(cls, value: str) -> str:
        """校验稳定方法编号。"""

        if not IDENTIFIER_PATTERN.fullmatch(value):
            raise ValueError("method_id 格式不正确")
        return value

    @field_validator("entry")
    @classmethod
    def validate_entry(cls, value: str) -> str:
        """限制入口为 module:function。"""

        if not ENTRY_PATTERN.fullmatch(value):
            raise ValueError("entry 必须使用 module:function 格式")
        return value


class AlgorithmManifest(StrictModel):
    """一个完整算法功能及其方法列表。"""

    algorithm_id: str
    display_name: str
    category: str
    description: str
    methods: list[MethodManifest] = Field(min_length=1)

    @field_validator("algorithm_id")
    @classmethod
    def validate_algorithm_id(cls, value: str) -> str:
        """校验稳定算法编号。"""

        if not IDENTIFIER_PATTERN.fullmatch(value):
            raise ValueError("algorithm_id 格式不正确")
        return value

    @model_validator(mode="after")
    def ensure_unique_methods(self) -> AlgorithmManifest:
        """同一算法内的方法编号不能重复。"""

        method_ids = [method.method_id for method in self.methods]
        if len(method_ids) != len(set(method_ids)):
            raise ValueError(f"算法 {self.algorithm_id} 存在重复 method_id")
        return self


class PackageManifest(StrictModel):
    """算法包导入、登记和运行使用的完整清单。"""

    protocol_version: str
    package_id: str
    package_name: str
    version: str
    publisher: str
    description: str
    runtime: RuntimeSpec
    algorithms: list[AlgorithmManifest] = Field(min_length=1)

    @field_validator("protocol_version")
    @classmethod
    def validate_protocol_version(cls, value: str) -> str:
        """第一阶段只接受 1.x 协议。"""

        if not PROTOCOL_VERSION_PATTERN.fullmatch(value):
            raise ValueError("protocol_version 必须使用 '主版本.次版本' 格式")
        if value.split(".", maxsplit=1)[0] != "1":
            raise ValueError("当前只支持 1.x 算法协议")
        return value

    @field_validator("package_id")
    @classmethod
    def validate_package_id(cls, value: str) -> str:
        """校验稳定算法包编号。"""

        if not IDENTIFIER_PATTERN.fullmatch(value):
            raise ValueError("package_id 格式不正确")
        return value

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        """发布版本必须使用三段式版本号。"""

        if not SEMVER_PATTERN.fullmatch(value):
            raise ValueError("version 必须使用三段式版本号，例如 1.0.0")
        return value

    @model_validator(mode="after")
    def ensure_unique_algorithms(self) -> PackageManifest:
        """同一包内算法编号不能重复。"""

        algorithm_ids = [algorithm.algorithm_id for algorithm in self.algorithms]
        if len(algorithm_ids) != len(set(algorithm_ids)):
            raise ValueError("algorithms 存在重复 algorithm_id")
        return self

    def find_method(self, operation_key: str) -> MethodManifest | None:
        """按稳定组合编号查找一个运行方法。"""

        for algorithm in self.algorithms:
            for method in algorithm.methods:
                if f"{algorithm.algorithm_id}.{method.method_id}" == operation_key:
                    return method
        return None
