"""为现有模块安全生成算法、方法、入口代码和测试骨架。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from pydantic import ValidationError

from algorithm_cli.validator import validate_library
from algorithm_sdk.models import IDENTIFIER_PATTERN, LibraryManifest, PackageManifest
from algorithm_studio.repository import StudioLibrary, StudioModule

InputMode = Literal["none", "single", "multiple"]
FieldKind = Literal["any", "quantitative", "categorical"]
OutputKind = Literal["analysis", "data"]


class ScaffoldingError(ValueError):
    """脚手架输入或写入结果不满足算法库约束。"""


@dataclass(frozen=True, slots=True)
class MethodDraft:
    """新方法向导收集的最小协议信息。"""

    method_id: str
    display_name: str
    display_name_en: str
    description: str
    description_en: str
    input_mode: InputMode = "none"
    slot_id: str = "columns"
    slot_display_name: str = "分析变量"
    slot_display_name_en: str = "Variables"
    field_kind: FieldKind = "any"
    output_kind: OutputKind = "analysis"
    deterministic: bool = True


@dataclass(frozen=True, slots=True)
class NewAlgorithmDraft:
    """新算法组和首个方法的信息。"""

    algorithm_id: str
    display_name: str
    display_name_en: str
    category: str
    category_display_name: str
    category_display_name_en: str
    description: str
    description_en: str
    method: MethodDraft


@dataclass(frozen=True, slots=True)
class ScaffoldingResult:
    """一次脚手架事务生成或更新的文件。"""

    operation_key: str
    source_path: Path
    test_path: Path
    changed_paths: tuple[Path, ...]
    library_version: str
    module_version: str


def create_algorithm(
    library: StudioLibrary,
    module_id: str,
    draft: NewAlgorithmDraft,
) -> ScaffoldingResult:
    """向现有模块增加算法组、首个方法和可运行测试骨架。"""

    validate_algorithm_draft(draft)

    module = _find_module(library, module_id)
    if any(item.manifest.algorithm_id == draft.algorithm_id for item in module.algorithms):
        raise ScaffoldingError(f"算法编号已存在：{draft.algorithm_id}")

    operation_key = f"{draft.algorithm_id}.{draft.method.method_id}"
    source_stem = draft.algorithm_id
    function_name = f"run_{draft.method.method_id}"
    entry = f"algorithms.{source_stem}:{function_name}"
    algorithm_payload: dict[str, object] = {
        "algorithm_id": draft.algorithm_id,
        "display_name": draft.display_name.strip(),
        "display_name_en": draft.display_name_en.strip(),
        "category": draft.category,
        "category_display_name": draft.category_display_name.strip(),
        "category_display_name_en": draft.category_display_name_en.strip(),
        "description": draft.description.strip(),
        "description_en": draft.description_en.strip(),
        "methods": [_method_payload(draft.method, entry)],
    }
    return _commit_scaffold(
        library=library,
        module_id=module_id,
        operation_key=operation_key,
        source_stem=source_stem,
        function_name=function_name,
        method=draft.method,
        algorithm_payload=algorithm_payload,
        target_algorithm_id=None,
    )


def create_method(
    library: StudioLibrary,
    module_id: str,
    algorithm_id: str,
    draft: MethodDraft,
) -> ScaffoldingResult:
    """向现有算法组增加一个独立源码文件和测试骨架。"""

    validate_method_draft(draft)
    module = _find_module(library, module_id)
    algorithm = next(
        (item for item in module.algorithms if item.manifest.algorithm_id == algorithm_id),
        None,
    )
    if algorithm is None:
        raise ScaffoldingError(f"模块 {module_id} 中不存在算法 {algorithm_id}")
    if any(item.manifest.method_id == draft.method_id for item in algorithm.methods):
        raise ScaffoldingError(f"方法编号已存在：{algorithm_id}.{draft.method_id}")

    operation_key = f"{algorithm_id}.{draft.method_id}"
    source_stem = f"{algorithm_id}_{draft.method_id}"
    function_name = f"run_{draft.method_id}"
    entry = f"algorithms.{source_stem}:{function_name}"
    return _commit_scaffold(
        library=library,
        module_id=module_id,
        operation_key=operation_key,
        source_stem=source_stem,
        function_name=function_name,
        method=draft,
        algorithm_payload=None,
        target_algorithm_id=algorithm_id,
        method_payload=_method_payload(draft, entry),
    )


def _commit_scaffold(
    *,
    library: StudioLibrary,
    module_id: str,
    operation_key: str,
    source_stem: str,
    function_name: str,
    method: MethodDraft,
    algorithm_payload: dict[str, object] | None,
    target_algorithm_id: str | None,
    method_payload: dict[str, object] | None = None,
) -> ScaffoldingResult:
    module = _find_module(library, module_id)
    source_path = module.directory / "algorithms" / f"{source_stem}.py"
    test_path = library.project_root / "tests" / f"test_{operation_key.replace('.', '_')}.py"
    cases_path = library.project_root / "tests" / "studio_operation_cases.json"
    for path in (source_path, test_path):
        if path.exists():
            raise ScaffoldingError(f"目标文件已存在：{path.relative_to(library.project_root)}")

    package_payload = _read_json_object(module.manifest_path)
    algorithms = package_payload.get("algorithms")
    if not isinstance(algorithms, list):
        raise ScaffoldingError(f"{module.manifest_path} 缺少 algorithms 数组")
    if algorithm_payload is not None:
        algorithms.append(algorithm_payload)
    else:
        target = next(
            (
                item
                for item in algorithms
                if isinstance(item, dict) and item.get("algorithm_id") == target_algorithm_id
            ),
            None,
        )
        if not isinstance(target, dict):
            raise ScaffoldingError(f"manifest 中不存在算法 {target_algorithm_id}")
        methods = target.get("methods")
        if not isinstance(methods, list) or method_payload is None:
            raise ScaffoldingError(f"算法 {target_algorithm_id} 缺少 methods 数组")
        methods.append(method_payload)

    new_module_version = _bump_patch(module.manifest.version)
    package_payload["version"] = new_module_version
    try:
        PackageManifest.model_validate(package_payload)
    except ValidationError as exc:
        raise ScaffoldingError(f"生成的模块清单不合法：{_first_validation_message(exc)}") from exc

    library_payload = _read_json_object(library.manifest_path)
    new_library_version = _bump_patch(library.manifest.version)
    library_payload["version"] = new_library_version
    module_references = library_payload.get("modules")
    if not isinstance(module_references, list):
        raise ScaffoldingError("library_manifest.json 缺少 modules 数组")
    reference = next(
        (
            item
            for item in module_references
            if isinstance(item, dict) and item.get("module_id") == module_id
        ),
        None,
    )
    if not isinstance(reference, dict):
        raise ScaffoldingError(f"根清单中不存在模块 {module_id}")
    reference["version"] = new_module_version
    try:
        LibraryManifest.model_validate(library_payload)
    except ValidationError as exc:
        raise ScaffoldingError(f"生成的根清单不合法：{_first_validation_message(exc)}") from exc

    cases_payload = _read_optional_json_object(cases_path)
    if operation_key in cases_payload:
        raise ScaffoldingError(f"运行用例已经存在：{operation_key}")
    cases_payload[operation_key] = {
        "package_id": module_id,
        "slots": _runtime_case_slots(method),
        "parameters": {},
    }

    writes = {
        module.manifest_path: _json_text(package_payload),
        library.manifest_path: _json_text(library_payload),
        source_path: _source_text(operation_key, function_name, method),
        test_path: _test_text(module_id, source_stem, function_name, method),
        cases_path: _json_text(cases_payload),
    }
    _write_transaction(library, writes)
    return ScaffoldingResult(
        operation_key=operation_key,
        source_path=source_path,
        test_path=test_path,
        changed_paths=tuple(writes),
        library_version=new_library_version,
        module_version=new_module_version,
    )


def _method_payload(draft: MethodDraft, entry: str) -> dict[str, object]:
    slots: list[dict[str, object]] = []
    if draft.input_mode != "none":
        allowed_data_types, allowed_analysis_types = _allowed_types(draft.field_kind)
        slot: dict[str, object] = {
            "id": draft.slot_id,
            "display_name": draft.slot_display_name.strip(),
            "display_name_en": draft.slot_display_name_en.strip(),
            "type": "column" if draft.input_mode == "single" else "columns",
            "required": True,
            "min_items": 1,
            "allowed_data_types": allowed_data_types,
            "allowed_analysis_types": allowed_analysis_types,
        }
        if draft.input_mode == "single":
            slot["max_items"] = 1
        slots.append(slot)
    output = {
        "data_type": "table" if draft.output_kind == "data" else "analysis_result",
        "produces_data_version": draft.output_kind == "data",
        "result_blocks": ["metrics", "metadata"],
    }
    return {
        "method_id": draft.method_id,
        "display_name": draft.display_name.strip(),
        "display_name_en": draft.display_name_en.strip(),
        "entry": entry,
        "description": draft.description.strip(),
        "description_en": draft.description_en.strip(),
        "input": {"data_type": "table", "slots": slots, "constraints": []},
        "parameters_schema": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        "output": output,
        "deterministic": draft.deterministic,
    }


def _source_text(operation_key: str, function_name: str, draft: MethodDraft) -> str:
    data_line = "data=data.copy(deep=True),\n        " if draft.output_kind == "data" else ""
    return f'''"""{draft.display_name}算法入口。"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
from pydantic import JsonValue

from algorithm_sdk import AlgorithmResult, Metric


def {function_name}(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """{draft.description.strip()}"""

    del slots, parameters, context
    return AlgorithmResult(
        {data_line}metrics=[Metric(id="row_count", display_name="输入行数", value=len(data.index), unit="行")],
        metadata={{
            "method": "{operation_key}",
            "rows_in": len(data.index),
            "rows_out": len(data.index),
        }},
    )
'''


def _test_text(
    module_id: str,
    source_stem: str,
    function_name: str,
    draft: MethodDraft,
) -> str:
    slots = _standalone_test_slots(draft)
    data_assertion = (
        "assert result.data is not None\n    assert result.data is not data"
        if draft.output_kind == "data"
        else "assert result.data is None"
    )
    return f'''"""{draft.display_name}脚手架测试。"""

from __future__ import annotations

import pandas as pd

from packages.{module_id}.algorithms.{source_stem} import {function_name}


def test_{source_stem}_runs() -> None:
    """新方法应返回符合协议的最小结构化结果。"""

    data = pd.DataFrame({{"value": [1.0, 2.0, 3.0], "group": ["甲", "乙", "甲"]}})
    result = {function_name}(
        data,
        {slots!r},
        {{}},
        {{"random_seed": 42}},
    )

    assert result.status == "success"
    assert result.metrics
    {data_assertion}
'''


def _write_transaction(library: StudioLibrary, writes: dict[Path, str]) -> None:
    previous = {path: path.read_bytes() if path.exists() else None for path in writes}
    try:
        for path, text in writes.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        report = validate_library(library.packages_directory, check_dependencies=False)
        if not report.is_valid:
            messages = "; ".join(issue.message for issue in report.issues)
            raise ScaffoldingError(f"生成后整库校验失败：{messages}")
    except Exception:
        for path, previous_bytes in previous.items():
            if previous_bytes is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(previous_bytes)
        raise


def _runtime_case_slots(draft: MethodDraft) -> dict[str, list[str]]:
    if draft.input_mode == "none":
        return {}
    column = "group" if draft.field_kind == "categorical" else "x1"
    return {draft.slot_id: [column]}


def _standalone_test_slots(draft: MethodDraft) -> dict[str, list[str]]:
    if draft.input_mode == "none":
        return {}
    column = "group" if draft.field_kind == "categorical" else "value"
    return {draft.slot_id: [column]}


def _allowed_types(field_kind: FieldKind) -> tuple[list[str], list[str]]:
    if field_kind == "quantitative":
        return ["integer", "number"], ["quantitative"]
    if field_kind == "categorical":
        return ["integer", "number", "string", "boolean", "datetime", "category"], ["categorical"]
    return ["integer", "number", "string", "boolean", "datetime", "category"], [
        "categorical",
        "quantitative",
    ]


def validate_algorithm_draft(draft: NewAlgorithmDraft) -> None:
    """在写入文件前校验新算法向导输入。"""

    _validate_identifier("algorithm_id", draft.algorithm_id)
    _validate_required("算法中文名称", draft.display_name)
    _validate_required("算法英文名称", draft.display_name_en)
    _validate_identifier("category", draft.category)
    _validate_required("分类中文名称", draft.category_display_name)
    _validate_required("分类英文名称", draft.category_display_name_en)
    _validate_required("算法中文说明", draft.description)
    _validate_required("算法英文说明", draft.description_en)
    validate_method_draft(draft.method)


def validate_method_draft(draft: MethodDraft) -> None:
    """在写入文件前校验新方法向导输入。"""

    _validate_identifier("method_id", draft.method_id)
    _validate_required("方法中文名称", draft.display_name)
    _validate_required("方法英文名称", draft.display_name_en)
    _validate_required("方法中文说明", draft.description)
    _validate_required("方法英文说明", draft.description_en)
    if draft.input_mode not in {"none", "single", "multiple"}:
        raise ScaffoldingError("输入模式不受支持")
    if draft.input_mode != "none":
        _validate_identifier("slot_id", draft.slot_id)
        _validate_required("变量槽位中文名称", draft.slot_display_name)
        _validate_required("变量槽位英文名称", draft.slot_display_name_en)
    if draft.field_kind not in {"any", "quantitative", "categorical"}:
        raise ScaffoldingError("变量类型不受支持")
    if draft.output_kind not in {"analysis", "data"}:
        raise ScaffoldingError("输出类型不受支持")


def _validate_identifier(label: str, value: str) -> None:
    if not IDENTIFIER_PATTERN.fullmatch(value):
        raise ScaffoldingError(f"{label} 必须以小写字母开头，只能包含小写字母、数字和下划线")


def _validate_required(label: str, value: str) -> None:
    if not value.strip():
        raise ScaffoldingError(f"{label}不能为空")


def _find_module(library: StudioLibrary, module_id: str) -> StudioModule:
    module = next(
        (item for item in library.modules if item.manifest.package_id == module_id),
        None,
    )
    if module is None:
        raise ScaffoldingError(f"算法库中不存在模块 {module_id}")
    return module


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ScaffoldingError(f"无法读取 JSON：{path}") from exc
    if not isinstance(payload, dict):
        raise ScaffoldingError(f"JSON 根节点必须是对象：{path}")
    return cast(dict[str, object], payload)


def _read_optional_json_object(path: Path) -> dict[str, object]:
    return _read_json_object(path) if path.exists() else {}


def _json_text(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _bump_patch(version: str) -> str:
    major, minor, patch = (int(item) for item in version.split("."))
    return f"{major}.{minor}.{patch + 1}"


def _first_validation_message(exc: ValidationError) -> str:
    error = exc.errors(include_url=False)[0]
    location = ".".join(str(item) for item in error["loc"])
    return f"{location} {error['msg']}"
