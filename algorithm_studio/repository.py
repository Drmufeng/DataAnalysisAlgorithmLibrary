"""把算法库清单转换为桌面工作台使用的只读导航模型。"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from algorithm_sdk.manifest import load_library_manifest, load_manifest
from algorithm_sdk.models import (
    AlgorithmManifest,
    LibraryManifest,
    MethodManifest,
    PackageManifest,
)


class StudioRepositoryError(ValueError):
    """算法库无法安全地映射为工作台目录。"""


@dataclass(frozen=True, slots=True)
class StudioMethod:
    """一个可执行方法及其源码入口。"""

    module_id: str
    algorithm_id: str
    manifest: MethodManifest
    source_path: Path
    function_name: str
    function_line: int

    @property
    def operation_key(self) -> str:
        """返回任务协议使用的稳定方法编号。"""

        return f"{self.algorithm_id}.{self.manifest.method_id}"


@dataclass(frozen=True, slots=True)
class StudioAlgorithm:
    """一个算法组及其全部可执行方法。"""

    module_id: str
    manifest: AlgorithmManifest
    methods: tuple[StudioMethod, ...]


@dataclass(frozen=True, slots=True)
class StudioModule:
    """完整发行包中的一个内部算法模块。"""

    directory: Path
    manifest_path: Path
    manifest: PackageManifest
    algorithms: tuple[StudioAlgorithm, ...]

    @property
    def method_count(self) -> int:
        """返回模块中的可执行方法数量。"""

        return sum(len(algorithm.methods) for algorithm in self.algorithms)


@dataclass(frozen=True, slots=True)
class StudioLibrary:
    """工作台当前打开的一套完整算法库源码。"""

    project_root: Path
    packages_directory: Path
    manifest_path: Path
    manifest: LibraryManifest
    modules: tuple[StudioModule, ...]

    @property
    def algorithm_count(self) -> int:
        """返回完整发行包中的算法组数量。"""

        return sum(len(module.algorithms) for module in self.modules)

    @property
    def method_count(self) -> int:
        """返回完整发行包中的可执行方法数量。"""

        return sum(module.method_count for module in self.modules)

    def iter_methods(self) -> tuple[StudioMethod, ...]:
        """按清单顺序返回全部方法。"""

        return tuple(
            method
            for module in self.modules
            for algorithm in module.algorithms
            for method in algorithm.methods
        )


def find_project_root(explicit_root: Path | None = None) -> Path:
    """定位同时包含源码包和完整发行清单的项目根目录。"""

    candidates: tuple[Path, ...]
    if explicit_root is not None:
        normalized = explicit_root.expanduser().resolve()
        candidates = (
            (normalized, normalized.parent) if normalized.name == "packages" else (normalized,)
        )
    else:
        installed_root = Path(__file__).resolve().parents[1]
        current = Path.cwd().resolve()
        candidates = (current, *current.parents, installed_root, *installed_root.parents)

    visited: set[Path] = set()
    for candidate in candidates:
        if candidate in visited:
            continue
        visited.add(candidate)
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "packages" / "library_manifest.json"
        ).is_file():
            return candidate
    target = str(explicit_root) if explicit_root is not None else "当前目录"
    raise StudioRepositoryError(
        f"无法从 {target} 定位包含 pyproject.toml 和 packages/library_manifest.json 的源码项目"
    )


def load_studio_library(project_root: Path | None = None) -> StudioLibrary:
    """读取完整算法库，并解析每个方法对应的源码文件和函数行号。"""

    root = find_project_root(project_root)
    packages_directory = (root / "packages").resolve()
    library_manifest_path = packages_directory / "library_manifest.json"
    library_manifest = load_library_manifest(packages_directory)
    modules: list[StudioModule] = []

    for reference in library_manifest.modules:
        manifest_path = _safe_child(packages_directory, reference.manifest_path)
        module_directory = manifest_path.parent
        package_manifest = load_manifest(module_directory)
        if package_manifest.package_id != reference.module_id:
            raise StudioRepositoryError(
                f"模块 {reference.module_id} 的 manifest 声明为 {package_manifest.package_id}"
            )
        if package_manifest.version != reference.version:
            raise StudioRepositoryError(
                f"模块 {reference.module_id} 版本不一致："
                f"根清单为 {reference.version}，模块清单为 {package_manifest.version}"
            )

        algorithms: list[StudioAlgorithm] = []
        for algorithm_manifest in package_manifest.algorithms:
            methods = tuple(
                _build_method(
                    module_directory,
                    package_manifest.package_id,
                    algorithm_manifest.algorithm_id,
                    method_manifest,
                )
                for method_manifest in algorithm_manifest.methods
            )
            algorithms.append(
                StudioAlgorithm(
                    module_id=package_manifest.package_id,
                    manifest=algorithm_manifest,
                    methods=methods,
                )
            )
        modules.append(
            StudioModule(
                directory=module_directory,
                manifest_path=manifest_path,
                manifest=package_manifest,
                algorithms=tuple(algorithms),
            )
        )

    return StudioLibrary(
        project_root=root,
        packages_directory=packages_directory,
        manifest_path=library_manifest_path,
        manifest=library_manifest,
        modules=tuple(modules),
    )


def _build_method(
    module_directory: Path,
    module_id: str,
    algorithm_id: str,
    manifest: MethodManifest,
) -> StudioMethod:
    module_name, function_name = manifest.entry.split(":", maxsplit=1)
    relative_source = Path(*module_name.split(".")).with_suffix(".py")
    source_path = _safe_child(module_directory, relative_source)
    if not source_path.is_file():
        raise StudioRepositoryError(
            f"方法 {algorithm_id}.{manifest.method_id} 的入口文件不存在：{source_path}"
        )
    return StudioMethod(
        module_id=module_id,
        algorithm_id=algorithm_id,
        manifest=manifest,
        source_path=source_path,
        function_name=function_name,
        function_line=_find_function_line(source_path, function_name),
    )


def _safe_child(root: Path, relative_path: str | Path) -> Path:
    candidate = (root / Path(relative_path)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise StudioRepositoryError(f"清单路径超出算法库目录：{relative_path}") from exc
    return candidate


def _find_function_line(source_path: Path, function_name: str) -> int:
    try:
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(source_path))
    except (OSError, SyntaxError, UnicodeError) as exc:
        raise StudioRepositoryError(f"无法解析源码 {source_path}: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return node.lineno
    raise StudioRepositoryError(f"入口函数 {function_name} 不存在：{source_path}")
