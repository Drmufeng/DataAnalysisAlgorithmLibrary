"""算法库工作台导航模型测试。"""

from __future__ import annotations

from pathlib import Path

from algorithm_studio.repository import find_project_root, load_studio_library

PROJECT_ROOT = Path(__file__).parents[1]


def test_studio_loads_complete_library_tree() -> None:
    """工作台应按现有清单展示全部模块、算法和方法。"""

    library = load_studio_library(PROJECT_ROOT)

    assert library.manifest.library_id == "data_analysis_algorithm_library"
    assert len(library.modules) == 8
    assert library.algorithm_count == 29
    assert library.method_count == 53


def test_studio_resolves_every_method_entry_to_source() -> None:
    """每个方法都应定位到模块内真实存在的入口函数。"""

    library = load_studio_library(PROJECT_ROOT)
    methods = library.iter_methods()

    assert len({method.operation_key for method in methods}) == 53
    assert all(method.source_path.is_file() for method in methods)
    assert all(method.function_line > 0 for method in methods)
    assert all(method.source_path.is_relative_to(library.packages_directory) for method in methods)


def test_studio_accepts_project_or_packages_directory() -> None:
    """启动参数可以指向项目根目录，也可以直接指向 packages。"""

    assert find_project_root(PROJECT_ROOT) == PROJECT_ROOT.resolve()
    assert find_project_root(PROJECT_ROOT / "packages") == PROJECT_ROOT.resolve()
