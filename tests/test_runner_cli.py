"""本地文件执行器端到端测试。"""

from __future__ import annotations

import json
from pathlib import Path

from runner_contract.cli import main
from tests.helpers import make_overview_request, make_sample_data

PROJECT_ROOT = Path(__file__).parents[1]
PACKAGE_DIR = PROJECT_ROOT / "packages" / "descriptive_analysis"


def test_runner_cli_writes_strict_result(tmp_path: Path) -> None:
    """文件执行器应读取相对协议文件并生成成功结果。"""

    input_path = tmp_path / "input.parquet"
    request_path = tmp_path / "request.json"
    output_dir = tmp_path / "output"
    make_sample_data().to_parquet(input_path, index=False)
    request_path.write_text(
        make_overview_request().model_dump_json(indent=2),
        encoding="utf-8",
    )
    exit_code = main(
        [
            "--package-dir",
            str(PACKAGE_DIR),
            "--input",
            str(input_path),
            "--request",
            str(request_path),
            "--output",
            str(output_dir),
        ]
    )
    assert exit_code == 0
    payload = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "success"
    assert not (output_dir / "data.parquet").exists()


def test_runner_cli_writes_structured_error(tmp_path: Path) -> None:
    """请求字段错误时仍应产生不含堆栈的结构化错误。"""

    input_path = tmp_path / "input.parquet"
    request_path = tmp_path / "request.json"
    output_dir = tmp_path / "output"
    make_sample_data().to_parquet(input_path, index=False)
    request = make_overview_request()
    request_path.write_text(
        request.model_copy(update={"slots": {"columns": []}}).model_dump_json(indent=2),
        encoding="utf-8",
    )
    exit_code = main(
        [
            "--package-dir",
            str(PACKAGE_DIR),
            "--input",
            str(input_path),
            "--request",
            str(request_path),
            "--output",
            str(output_dir),
        ]
    )
    assert exit_code == 1
    payload = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "COLUMN_ERROR"
    assert "traceback" not in json.dumps(payload).lower()
