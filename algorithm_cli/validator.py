"""算法包静态结构、依赖和入口校验。"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from jsonschema import Draft202012Validator, exceptions
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from pydantic import JsonValue

from algorithm_sdk.errors import ProtocolError
from algorithm_sdk.manifest import load_library_manifest, load_manifest
from algorithm_sdk.models import LibraryManifest, MethodManifest, PackageManifest
from algorithm_sdk.validation import normalize_parameters_schema

MAX_PACKAGE_FILE_BYTES = 50 * 1024 * 1024
IGNORED_PARTS = {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__"}


@dataclass(slots=True, frozen=True)
class ValidationIssue:
    """一项算法包校验问题。"""

    level: Literal["error", "warning"]
    code: str
    message: str
    location: str | None = None

    def to_payload(self) -> dict[str, JsonValue]:
        """生成命令行 JSON 输出。"""

        return {
            "level": self.level,
            "code": self.code,
            "message": self.message,
            "location": self.location,
        }


@dataclass(slots=True)
class ValidationReport:
    """一个算法包的完整校验结果。"""

    package_id: str | None = None
    version: str | None = None
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """没有 error 级问题时算法包可进入下一步。"""

        return not any(issue.level == "error" for issue in self.issues)

    def add_error(self, code: str, message: str, location: str | None = None) -> None:
        """追加阻止导入的问题。"""

        self.issues.append(ValidationIssue("error", code, message, location))

    def add_warning(self, code: str, message: str, location: str | None = None) -> None:
        """追加不阻止导入但需要管理员确认的问题。"""

        self.issues.append(ValidationIssue("warning", code, message, location))

    def to_payload(self) -> dict[str, JsonValue]:
        """生成可被平台保存的结构化校验结果。"""

        return {
            "valid": self.is_valid,
            "package_id": self.package_id,
            "version": self.version,
            "issues": [issue.to_payload() for issue in self.issues],
        }


@dataclass(slots=True)
class LibraryValidationReport:
    """完整算法库发行包的聚合校验结果。"""

    library_id: str | None = None
    version: str | None = None
    module_count: int = 0
    algorithm_count: int = 0
    method_count: int = 0
    modules: list[JsonValue] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """任一内部模块出现 error 时，完整发行包都不能启用。"""

        return not any(issue.level == "error" for issue in self.issues)

    def add_error(self, code: str, message: str, location: str | None = None) -> None:
        """追加阻止整库登记的问题。"""

        self.issues.append(ValidationIssue("error", code, message, location))

    def to_payload(self) -> dict[str, JsonValue]:
        """生成平台可以保存的整库校验结果。"""

        return {
            "valid": self.is_valid,
            "library_id": self.library_id,
            "version": self.version,
            "module_count": self.module_count,
            "algorithm_count": self.algorithm_count,
            "method_count": self.method_count,
            "modules": self.modules,
            "issues": [issue.to_payload() for issue in self.issues],
        }


def validate_package(
    package_dir: Path,
    *,
    check_dependencies: bool = True,
) -> ValidationReport:
    """校验算法包清单、文件、参数 Schema、依赖和 Python 入口。"""

    report = ValidationReport()
    if not package_dir.is_dir():
        report.add_error("PACKAGE_NOT_FOUND", "算法包目录不存在")
        return report
    _validate_files(package_dir, report)

    try:
        manifest = load_manifest(package_dir)
    except ProtocolError as exc:
        report.add_error("MANIFEST_INVALID", exc.message, "manifest.json")
        return report

    report.package_id = manifest.package_id
    report.version = manifest.version
    _validate_runtime(manifest, report, check_dependencies=check_dependencies)
    _validate_display_metadata(manifest, report)
    for algorithm in manifest.algorithms:
        for method in algorithm.methods:
            operation_key = f"{algorithm.algorithm_id}.{method.method_id}"
            _validate_parameter_schema(method, operation_key, report)
            _validate_entry(package_dir, method, operation_key, report)
    return report


def validate_library(
    library_root: Path,
    *,
    check_dependencies: bool = True,
) -> LibraryValidationReport:
    """校验根清单、全部模块及跨模块 operation key 唯一性。"""

    report = LibraryValidationReport()
    if not library_root.is_dir():
        report.add_error("LIBRARY_NOT_FOUND", "算法库发行目录不存在")
        return report
    try:
        manifest = load_library_manifest(library_root)
    except ProtocolError as exc:
        report.add_error("LIBRARY_MANIFEST_INVALID", exc.message, "library_manifest.json")
        return report

    report.library_id = manifest.library_id
    report.version = manifest.version
    _validate_declared_modules(
        library_root,
        manifest,
        report,
        check_dependencies=check_dependencies,
    )
    return report


def _validate_declared_modules(
    library_root: Path,
    manifest: LibraryManifest,
    report: LibraryValidationReport,
    *,
    check_dependencies: bool,
) -> None:
    """逐个校验模块清单，并拒绝根清单之外的隐式模块。"""

    declared_paths = {module.manifest_path for module in manifest.modules}
    discovered_paths = {
        path.relative_to(library_root).as_posix()
        for path in library_root.rglob("manifest.json")
        if path.is_file()
    }
    for undeclared in sorted(discovered_paths - declared_paths):
        report.add_error(
            "UNDECLARED_MODULE",
            "发现未在根清单登记的算法模块",
            undeclared,
        )

    operation_locations: dict[str, str] = {}
    for reference in manifest.modules:
        location = reference.manifest_path
        if reference.manifest_path.split("/", maxsplit=1)[0] != reference.module_id:
            report.add_error(
                "MODULE_PATH_MISMATCH",
                "模块目录名必须和 module_id 一致",
                location,
            )
            continue
        module_dir = library_root / Path(reference.manifest_path).parent
        package_report = validate_package(
            module_dir,
            check_dependencies=check_dependencies,
        )
        for issue in package_report.issues:
            nested_location = location
            if issue.location:
                nested_location = f"{location}:{issue.location}"
            report.issues.append(
                ValidationIssue(issue.level, issue.code, issue.message, nested_location)
            )
        if package_report.package_id is None or package_report.version is None:
            continue
        try:
            package_manifest = load_manifest(module_dir)
        except ProtocolError:
            continue
        if package_manifest.package_id != reference.module_id:
            report.add_error(
                "MODULE_ID_MISMATCH",
                "根清单 module_id 与模块 package_id 不一致",
                location,
            )
        if package_manifest.version != reference.version:
            report.add_error(
                "MODULE_VERSION_MISMATCH",
                "根清单模块版本与模块 manifest 版本不一致",
                location,
            )

        module_method_count = sum(len(item.methods) for item in package_manifest.algorithms)
        report.modules.append(
            {
                "module_id": package_manifest.package_id,
                "module_name": package_manifest.package_name,
                "version": package_manifest.version,
                "protocol_version": package_manifest.protocol_version,
                "manifest_path": reference.manifest_path,
                "algorithm_count": len(package_manifest.algorithms),
                "method_count": module_method_count,
                "valid": package_report.is_valid,
            }
        )
        report.algorithm_count += len(package_manifest.algorithms)
        report.method_count += module_method_count
        for algorithm in package_manifest.algorithms:
            for method in algorithm.methods:
                operation_key = f"{algorithm.algorithm_id}.{method.method_id}"
                previous_location = operation_locations.get(operation_key)
                if previous_location is not None:
                    report.add_error(
                        "DUPLICATE_OPERATION_KEY",
                        f"方法编号 {operation_key} 已在 {previous_location} 登记",
                        location,
                    )
                else:
                    operation_locations[operation_key] = location
    report.module_count = len(report.modules)


def calculate_package_hash(package_dir: Path) -> str:
    """按相对文件名和内容计算稳定 SHA-256，不包含缓存和构建产物。"""

    if not package_dir.is_dir():
        raise ValueError("算法包目录不存在")
    digest = hashlib.sha256()
    files = sorted(
        (
            path
            for path in package_dir.rglob("*")
            if path.is_file() and not _should_ignore(path.relative_to(package_dir))
        ),
        key=lambda path: path.relative_to(package_dir).as_posix(),
    )
    for path in files:
        relative = path.relative_to(package_dir).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, byteorder="big"))
        digest.update(relative)
        with path.open("rb") as file_handle:
            while chunk := file_handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def calculate_library_hash(library_root: Path) -> str:
    """计算根清单和全部已声明模块共同组成的稳定内容哈希。"""

    return calculate_package_hash(library_root)


def _should_ignore(relative_path: Path) -> bool:
    """排除不会进入算法包发布物的本地文件。"""

    return bool(set(relative_path.parts) & IGNORED_PARTS) or relative_path.suffix == ".pyc"


def _validate_files(package_dir: Path, report: ValidationReport) -> None:
    """检查软链接、单文件体积和必要说明文件。"""

    for required_file in ("manifest.json", "README.md", "requirements.txt"):
        if not (package_dir / required_file).is_file():
            report.add_error(
                "REQUIRED_FILE_MISSING", f"缺少必要文件 {required_file}", required_file
            )
    for path in package_dir.rglob("*"):
        relative = path.relative_to(package_dir)
        if _should_ignore(relative):
            continue
        if path.is_symlink():
            report.add_error("SYMLINK_NOT_ALLOWED", "算法包不允许包含软链接", relative.as_posix())
        elif path.is_file() and path.stat().st_size > MAX_PACKAGE_FILE_BYTES:
            report.add_error(
                "FILE_TOO_LARGE",
                f"单个文件不能超过 {MAX_PACKAGE_FILE_BYTES // 1024 // 1024} MB",
                relative.as_posix(),
            )


def _validate_runtime(
    manifest: PackageManifest,
    report: ValidationReport,
    *,
    check_dependencies: bool,
) -> None:
    """检查当前 Python 和已安装依赖是否满足清单。"""

    try:
        python_specifier = SpecifierSet(manifest.runtime.python)
    except InvalidSpecifier:
        report.add_error("PYTHON_SPEC_INVALID", "runtime.python 版本范围不合法", "runtime.python")
        return
    current_python = ".".join(str(item) for item in sys.version_info[:3])
    if current_python not in python_specifier:
        report.add_error(
            "PYTHON_VERSION_UNSUPPORTED",
            f"当前 Python {current_python} 不满足 {manifest.runtime.python}",
            "runtime.python",
        )
    if not check_dependencies:
        return
    for dependency in manifest.runtime.dependencies:
        try:
            specifier = SpecifierSet(dependency.version)
        except InvalidSpecifier:
            report.add_error(
                "DEPENDENCY_SPEC_INVALID",
                f"依赖 {dependency.name} 的版本范围不合法",
                f"runtime.dependencies.{dependency.name}",
            )
            continue
        try:
            installed_version = importlib.metadata.version(dependency.name)
        except importlib.metadata.PackageNotFoundError:
            report.add_error(
                "DEPENDENCY_MISSING",
                f"缺少依赖 {dependency.name}",
                f"runtime.dependencies.{dependency.name}",
            )
            continue
        if installed_version not in specifier:
            report.add_error(
                "DEPENDENCY_VERSION_MISMATCH",
                f"依赖 {dependency.name} 当前为 {installed_version}，要求 {dependency.version}",
                f"runtime.dependencies.{dependency.name}",
            )


def _validate_parameter_schema(
    method: MethodManifest,
    operation_key: str,
    report: ValidationReport,
) -> None:
    """检查扩展参数描述转换后是否仍是合法 JSON Schema。"""

    try:
        Draft202012Validator.check_schema(normalize_parameters_schema(method.parameters_schema))
    except exceptions.SchemaError as exc:
        report.add_error(
            "PARAMETER_SCHEMA_INVALID",
            f"参数 Schema 不合法：{exc.message}",
            f"{operation_key}.parameters_schema",
        )


def _validate_display_metadata(
    manifest: PackageManifest,
    report: ValidationReport,
) -> None:
    """协议 1.1 起强制提供通用前端渲染所需的中英文展示元数据。"""

    try:
        protocol_minor = int(manifest.protocol_version.split(".", maxsplit=1)[1])
    except (IndexError, ValueError):
        return
    if protocol_minor < 1:
        return

    package_fields = {
        "package_name_en": manifest.package_name_en,
        "description_en": manifest.description_en,
    }
    for field_name, value in package_fields.items():
        if not value:
            report.add_error(
                "DISPLAY_METADATA_MISSING",
                f"协议 1.1 算法包缺少 {field_name}",
                field_name,
            )

    for algorithm in manifest.algorithms:
        algorithm_location = algorithm.algorithm_id
        algorithm_fields = {
            "display_name_en": algorithm.display_name_en,
            "category_display_name": algorithm.category_display_name,
            "category_display_name_en": algorithm.category_display_name_en,
            "description_en": algorithm.description_en,
        }
        for field_name, value in algorithm_fields.items():
            if not value:
                report.add_error(
                    "DISPLAY_METADATA_MISSING",
                    f"算法 {algorithm.algorithm_id} 缺少 {field_name}",
                    f"{algorithm_location}.{field_name}",
                )
        for method in algorithm.methods:
            operation_key = f"{algorithm.algorithm_id}.{method.method_id}"
            if not method.display_name_en:
                report.add_error(
                    "DISPLAY_METADATA_MISSING",
                    f"方法 {operation_key} 缺少 display_name_en",
                    f"{operation_key}.display_name_en",
                )
            if not method.description_en:
                report.add_error(
                    "DISPLAY_METADATA_MISSING",
                    f"方法 {operation_key} 缺少 description_en",
                    f"{operation_key}.description_en",
                )
            for slot in method.input.slots:
                if not slot.display_name_en:
                    report.add_error(
                        "DISPLAY_METADATA_MISSING",
                        f"字段槽位 {operation_key}.{slot.id} 缺少 display_name_en",
                        f"{operation_key}.input.slots.{slot.id}.display_name_en",
                    )
            _validate_parameter_display_metadata(method, operation_key, report)


def _validate_parameter_display_metadata(
    method: MethodManifest,
    operation_key: str,
    report: ValidationReport,
) -> None:
    """校验顶层运行参数及枚举选项是否可由通用界面直接展示。"""

    properties = method.parameters_schema.get("properties")
    if not isinstance(properties, dict):
        return
    for parameter_name, raw_schema in properties.items():
        if not isinstance(parameter_name, str) or not isinstance(raw_schema, dict):
            continue
        location = f"{operation_key}.parameters_schema.properties.{parameter_name}"
        if not raw_schema.get("title"):
            report.add_error(
                "DISPLAY_METADATA_MISSING",
                f"参数 {operation_key}.{parameter_name} 缺少中文 title",
                f"{location}.title",
            )
        if not raw_schema.get("x-title-en"):
            report.add_error(
                "DISPLAY_METADATA_MISSING",
                f"参数 {operation_key}.{parameter_name} 缺少英文 x-title-en",
                f"{location}.x-title-en",
            )
        enum_values = raw_schema.get("enum")
        if not isinstance(enum_values, list):
            continue
        one_of = raw_schema.get("oneOf")
        if not isinstance(one_of, list):
            one_of = []
        labeled_values = [
            option.get("const")
            for option in one_of
            if isinstance(option, dict) and option.get("title") and option.get("x-title-en")
        ]
        missing_values = [
            value
            for value in enum_values
            if not any(value == labeled_value for labeled_value in labeled_values)
        ]
        if missing_values:
            report.add_error(
                "DISPLAY_METADATA_MISSING",
                f"参数 {operation_key}.{parameter_name} 的枚举选项缺少中英文标题",
                f"{location}.oneOf",
            )


def _validate_entry(
    package_dir: Path,
    method: MethodManifest,
    operation_key: str,
    report: ValidationReport,
) -> None:
    """使用文件路径和 AST 检查入口，不在导入阶段执行算法代码。"""

    module_name, function_name = method.entry.split(":", maxsplit=1)
    relative_module = Path(*module_name.split("."))
    file_candidate = package_dir / relative_module.with_suffix(".py")
    package_candidate = package_dir / relative_module / "__init__.py"
    entry_path = file_candidate if file_candidate.is_file() else package_candidate
    if not entry_path.is_file():
        report.add_error(
            "ENTRY_NOT_FOUND",
            f"入口模块不存在：{module_name}",
            f"{operation_key}.entry",
        )
        return
    try:
        entry_path.resolve().relative_to(package_dir.resolve())
    except ValueError:
        report.add_error(
            "ENTRY_OUTSIDE_PACKAGE",
            "算法入口必须位于算法包目录内",
            f"{operation_key}.entry",
        )
        return
    try:
        source = entry_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=entry_path.name)
    except (UnicodeDecodeError, SyntaxError) as exc:
        report.add_error(
            "ENTRY_SOURCE_INVALID",
            f"入口源码不是合法 UTF-8 Python 文件：{exc}",
            entry_path.relative_to(package_dir).as_posix(),
        )
        return
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        ),
        None,
    )
    if function is None:
        report.add_error(
            "ENTRY_FUNCTION_NOT_FOUND",
            f"入口函数不存在：{function_name}",
            entry_path.relative_to(package_dir).as_posix(),
        )
        return
    argument_names = [argument.arg for argument in function.args.args]
    expected = ["data", "slots", "parameters", "context"]
    if argument_names != expected:
        report.add_error(
            "ENTRY_SIGNATURE_INVALID",
            "入口函数参数必须依次为 data、slots、parameters、context",
            entry_path.relative_to(package_dir).as_posix(),
        )
