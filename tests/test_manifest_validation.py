"""算法包清单、入口和内容哈希测试。"""

from __future__ import annotations

import shutil
from pathlib import Path

from algorithm_cli.validator import calculate_package_hash, validate_package
from algorithm_sdk.manifest import load_manifest

PROJECT_ROOT = Path(__file__).parents[1]
PACKAGE_DIR = PROJECT_ROOT / "packages" / "descriptive_analysis"


def test_descriptive_package_manifest_is_valid() -> None:
    """首个算法包应通过结构、依赖和入口检查。"""

    report = validate_package(PACKAGE_DIR)
    assert report.is_valid, [issue.to_payload() for issue in report.issues]
    manifest = load_manifest(PACKAGE_DIR)
    assert manifest.package_id == "descriptive_analysis"
    assert manifest.find_method("data_overview.basic") is not None


def test_package_hash_ignores_python_cache(tmp_path: Path) -> None:
    """本地缓存不能改变平台登记的算法包内容哈希。"""

    copied_package = tmp_path / "descriptive_analysis"
    shutil.copytree(PACKAGE_DIR, copied_package)
    initial_hash = calculate_package_hash(copied_package)
    cache_dir = copied_package / "algorithms" / "__pycache__"
    cache_dir.mkdir(exist_ok=True)
    (cache_dir / "data_overview.pyc").write_bytes(b"cache")
    assert calculate_package_hash(copied_package) == initial_hash


def test_manifest_missing_required_file_is_reported(tmp_path: Path) -> None:
    """缺少 requirements.txt 时给出明确导入错误。"""

    copied_package = tmp_path / "descriptive_analysis"
    shutil.copytree(PACKAGE_DIR, copied_package)
    (copied_package / "requirements.txt").unlink()
    report = validate_package(copied_package, check_dependencies=False)
    assert not report.is_valid
    assert any(issue.code == "REQUIRED_FILE_MISSING" for issue in report.issues)
