"""算法库工作台脚手架事务测试。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from algorithm_cli.validator import validate_library
from algorithm_studio.repository import load_studio_library
from algorithm_studio.scaffolding import (
    MethodDraft,
    NewAlgorithmDraft,
    ScaffoldingError,
    create_algorithm,
    create_method,
)

PROJECT_ROOT = Path(__file__).parents[1]


def test_studio_creates_algorithm_and_followup_method(tmp_path: Path) -> None:
    """向导应原子更新清单、版本、源码、测试和运行用例。"""

    project = _copy_studio_project(tmp_path)
    library = load_studio_library(project)
    result = create_algorithm(
        library,
        "descriptive_analysis",
        NewAlgorithmDraft(
            algorithm_id="sample_score",
            display_name="示例得分",
            display_name_en="Sample Score",
            category="descriptive_analysis",
            category_display_name="描述性分析",
            category_display_name_en="Descriptive Analysis",
            description="计算一个可扩展的示例得分。",
            description_en="Calculates an extensible sample score.",
            method=MethodDraft(
                method_id="basic",
                display_name="基础得分",
                display_name_en="Basic Score",
                description="返回输入数据的基础得分。",
                description_en="Returns a basic score for the input data.",
                input_mode="multiple",
                slot_id="x_columns",
                slot_display_name="分析变量",
                slot_display_name_en="Variables",
                field_kind="quantitative",
            ),
        ),
    )

    assert result.operation_key == "sample_score.basic"
    assert result.source_path.is_file()
    assert result.test_path.is_file()
    assert result.library_version == "0.2.1"
    assert result.module_version == "0.1.1"

    updated_library = load_studio_library(project)
    second_result = create_method(
        updated_library,
        "descriptive_analysis",
        "sample_score",
        MethodDraft(
            method_id="with_copy",
            display_name="生成副本",
            display_name_en="Create Copy",
            description="返回输入数据的安全副本。",
            description_en="Returns a safe copy of the input data.",
            output_kind="data",
        ),
    )

    final_library = load_studio_library(project)
    report = validate_library(final_library.packages_directory, check_dependencies=False)
    assert report.is_valid, [issue.to_payload() for issue in report.issues]
    assert final_library.method_count == 55
    assert second_result.library_version == "0.2.2"
    assert second_result.module_version == "0.1.2"
    cases = json.loads(
        (project / "tests" / "studio_operation_cases.json").read_text(encoding="utf-8")
    )
    assert cases["sample_score.basic"]["slots"] == {"x_columns": ["x1"]}
    assert cases["sample_score.with_copy"]["slots"] == {}


def test_studio_rejects_existing_source_without_changing_manifest(tmp_path: Path) -> None:
    """目标文件冲突时不能提前修改版本或 Manifest。"""

    project = _copy_studio_project(tmp_path)
    source = project / "packages" / "descriptive_analysis" / "algorithms" / "collision.py"
    source.write_text("# existing\n", encoding="utf-8")
    original_manifest = (project / "packages" / "library_manifest.json").read_bytes()
    library = load_studio_library(project)

    with pytest.raises(ScaffoldingError, match="目标文件已存在"):
        create_algorithm(
            library,
            "descriptive_analysis",
            NewAlgorithmDraft(
                algorithm_id="collision",
                display_name="冲突算法",
                display_name_en="Collision Algorithm",
                category="descriptive_analysis",
                category_display_name="描述性分析",
                category_display_name_en="Descriptive Analysis",
                description="用于验证文件冲突。",
                description_en="Validates source-file collisions.",
                method=MethodDraft(
                    method_id="basic",
                    display_name="基础方法",
                    display_name_en="Basic Method",
                    description="不会写入文件。",
                    description_en="Does not write files.",
                ),
            ),
        )

    assert (project / "packages" / "library_manifest.json").read_bytes() == original_manifest


def _copy_studio_project(tmp_path: Path) -> Path:
    project = tmp_path / "algorithm-library"
    shutil.copytree(PROJECT_ROOT / "packages", project / "packages")
    (project / "tests").mkdir(parents=True)
    shutil.copy2(
        PROJECT_ROOT / "tests" / "studio_operation_cases.json",
        project / "tests" / "studio_operation_cases.json",
    )
    (project / "pyproject.toml").write_text("[project]\nname='studio-test'\n", encoding="utf-8")
    return project
