"""基于文件协议的本地一次性算法执行器。"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from algorithm_sdk.errors import ErrorCode, ProtocolError
from algorithm_sdk.models import AlgorithmRequest, AlgorithmResult
from algorithm_sdk.serialization import result_to_payload
from runner_contract.runner import run_operation

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """创建本地执行器参数。"""

    parser = argparse.ArgumentParser(prog="dal-runner", description="运行一个数据分析算法步骤")
    parser.add_argument("--package-dir", type=Path, required=True, help="算法包目录")
    parser.add_argument("--input", type=Path, required=True, help="输入 Parquet 文件")
    parser.add_argument("--request", type=Path, required=True, help="UTF-8 JSON 请求文件")
    parser.add_argument("--output", type=Path, required=True, help="结果输出目录")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """执行一次文件任务，并始终尝试写入结构化 result.json。"""

    args = build_parser().parse_args(argv)
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        request = _read_request(args.request)
        data = _read_input(args.input)
        result = run_operation(args.package_dir, data, request)
        _write_success(output_dir, result)
        print(f"运行成功：{request.operation_key}，结果已写入 {output_dir.as_posix()}")
        return 0
    except ProtocolError as exc:
        _write_error(output_dir, exc)
        print(f"运行失败：{exc.code.value} {exc.message}")
        return 1
    except Exception:
        LOGGER.exception("本地执行器发生未处理异常")
        error = ProtocolError(
            ErrorCode.INTERNAL_ERROR,
            "执行器发生内部错误，请查看本地日志",
        )
        _write_error(output_dir, error)
        print(f"运行失败：{error.code.value} {error.message}")
        return 2


def _read_request(path: Path) -> AlgorithmRequest:
    """读取并校验 UTF-8 运行请求。"""

    if not path.is_file():
        raise ProtocolError(ErrorCode.DATA_ERROR, "运行请求文件不存在")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return AlgorithmRequest.model_validate(payload)
    except UnicodeDecodeError as exc:
        raise ProtocolError(ErrorCode.DATA_ERROR, "运行请求必须使用 UTF-8 编码") from exc
    except json.JSONDecodeError as exc:
        raise ProtocolError(
            ErrorCode.DATA_ERROR,
            f"运行请求不是合法 JSON：第 {exc.lineno} 行第 {exc.colno} 列",
        ) from exc
    except ValidationError as exc:
        first_error = exc.errors(include_url=False)[0]
        location = ".".join(str(item) for item in first_error["loc"])
        raise ProtocolError(
            ErrorCode.PARAMETER_ERROR,
            f"运行请求结构不正确：{location} {first_error['msg']}",
        ) from exc


def _read_input(path: Path) -> pd.DataFrame:
    """读取执行器内部约定的 Parquet 表格。"""

    if not path.is_file():
        raise ProtocolError(ErrorCode.DATA_ERROR, "输入数据文件不存在")
    if path.suffix.lower() not in {".parquet", ".pq"}:
        raise ProtocolError(ErrorCode.DATA_ERROR, "执行器第一版只接受 Parquet 输入")
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        raise ProtocolError(ErrorCode.DATA_ERROR, "输入 Parquet 文件无法读取") from exc


def _write_success(output_dir: Path, result: AlgorithmResult) -> None:
    """原子写入结果 JSON 和可选处理后数据。"""

    payload = result_to_payload(result)
    _write_json_atomic(output_dir / "result.json", payload)
    output_data = result.data
    if output_data is not None:
        temporary_data = output_dir / ".data.parquet.tmp"
        output_data.to_parquet(temporary_data, index=False)
        temporary_data.replace(output_dir / "data.parquet")


def _write_error(output_dir: Path, error: ProtocolError) -> None:
    """写入不含堆栈和本机路径的错误结果。"""

    payload = {"status": "error", "error": error.to_payload()}
    _write_json_atomic(output_dir / "result.json", payload)


def _write_json_atomic(path: Path, payload: object) -> None:
    """先写临时文件再替换，避免留下半截 JSON。"""

    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)
