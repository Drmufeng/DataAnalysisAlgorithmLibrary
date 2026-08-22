"""算法包清单、入口和内容哈希测试。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from algorithm_cli.validator import (
    calculate_library_hash,
    calculate_package_hash,
    validate_library,
    validate_package,
)
from algorithm_sdk.manifest import load_library_manifest, load_manifest

PROJECT_ROOT = Path(__file__).parents[1]
PACKAGE_DIR = PROJECT_ROOT / "packages" / "descriptive_analysis"
LIBRARY_ROOT = PROJECT_ROOT / "packages"


def test_complete_library_release_is_valid() -> None:
    """根清单应一次登记 8 个模块、29 个算法组和 53 个方法。"""

    report = validate_library(LIBRARY_ROOT)
    assert report.is_valid, [issue.to_payload() for issue in report.issues]
    assert report.library_id == "data_analysis_algorithm_library"
    assert report.version == "0.2.0"
    assert report.module_count == 8
    assert report.algorithm_count == 29
    assert report.method_count == 53
    manifest = load_library_manifest(LIBRARY_ROOT)
    assert {module.module_id for module in manifest.modules} == {
        "clustering_models",
        "data_processing",
        "descriptive_analysis",
        "lightgbm_classification",
        "ml_classification",
        "statistical_association",
        "statistical_models",
        "time_signal_processing",
    }


def test_library_manifest_rejects_module_version_mismatch(tmp_path: Path) -> None:
    """根清单和内部模块版本不一致时，整个发行包不能通过。"""

    copied_library = tmp_path / "packages"
    shutil.copytree(LIBRARY_ROOT, copied_library)
    manifest_path = copied_library / "library_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["modules"][0]["version"] = "9.9.9"
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    report = validate_library(copied_library, check_dependencies=False)

    assert not report.is_valid
    assert any(issue.code == "MODULE_VERSION_MISMATCH" for issue in report.issues)


def test_library_hash_ignores_python_cache(tmp_path: Path) -> None:
    """内部模块缓存文件不能改变完整发行内容哈希。"""

    copied_library = tmp_path / "packages"
    shutil.copytree(LIBRARY_ROOT, copied_library)
    initial_hash = calculate_library_hash(copied_library)
    cache_dir = copied_library / "data_processing" / "__pycache__"
    cache_dir.mkdir(exist_ok=True)
    (cache_dir / "module.pyc").write_bytes(b"cache")
    assert calculate_library_hash(copied_library) == initial_hash


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


def test_protocol_1_1_requires_bilingual_display_metadata(tmp_path: Path) -> None:
    """新版协议缺少界面展示元数据时应在导入校验阶段明确报错。"""

    copied_package = tmp_path / "descriptive_analysis"
    shutil.copytree(PACKAGE_DIR, copied_package)
    manifest_path = copied_package / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["protocol_version"] = "1.1"
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    report = validate_package(copied_package, check_dependencies=False)

    assert not report.is_valid
    assert any(issue.code == "DISPLAY_METADATA_MISSING" for issue in report.issues)
