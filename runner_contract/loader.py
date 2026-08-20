"""在单次执行进程中加载受控算法入口。"""

from __future__ import annotations

import importlib
import inspect
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from pathlib import Path
from types import ModuleType
from typing import TypeAlias

import pandas as pd
from pydantic import JsonValue

from algorithm_sdk.errors import ErrorCode, ProtocolError
from algorithm_sdk.models import AlgorithmResult

AlgorithmCallable: TypeAlias = Callable[
    [
        pd.DataFrame,
        Mapping[str, list[str]],
        Mapping[str, JsonValue],
        Mapping[str, JsonValue],
    ],
    AlgorithmResult | Mapping[str, object],
]


@contextmanager
def load_algorithm_entry(package_dir: Path, entry: str) -> Iterator[AlgorithmCallable]:
    """加载一个算法入口，并在完成后移除该算法包导入的模块。"""

    module_name, function_name = entry.split(":", maxsplit=1)
    package_root = package_dir.resolve()
    path_text = str(package_root)
    modules_before = set(sys.modules)
    sys.path.insert(0, path_text)
    try:
        importlib.invalidate_caches()
        module = importlib.import_module(module_name)
        _ensure_module_inside_package(module, package_root)
        candidate = getattr(module, function_name, None)
        if not callable(candidate):
            raise ProtocolError(
                ErrorCode.EXECUTION_ERROR,
                "算法入口函数不存在或不可调用",
                details={"entry": entry},
            )
        _validate_signature(candidate)
        yield candidate
    except ProtocolError:
        raise
    except (ImportError, OSError) as exc:
        raise ProtocolError(
            ErrorCode.DEPENDENCY_ERROR,
            "算法入口加载失败，请检查算法包依赖",
            details={"entry": entry},
        ) from exc
    finally:
        if sys.path and sys.path[0] == path_text:
            sys.path.pop(0)
        else:
            with suppress(ValueError):
                sys.path.remove(path_text)
        _remove_package_modules(package_root, modules_before)


def _ensure_module_inside_package(module: ModuleType, package_root: Path) -> None:
    """避免入口编号通过 Python 搜索路径加载到算法包外部模块。"""

    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str):
        raise ProtocolError(ErrorCode.EXECUTION_ERROR, "算法入口模块缺少源码文件")
    try:
        Path(module_file).resolve().relative_to(package_root)
    except ValueError as exc:
        raise ProtocolError(
            ErrorCode.EXECUTION_ERROR,
            "算法入口必须位于算法包目录内",
        ) from exc


def _validate_signature(candidate: Callable[..., object]) -> None:
    """运行前再次检查入口参数，防止导入后内容与清单不一致。"""

    signature = inspect.signature(candidate)
    parameter_names = list(signature.parameters)
    expected = ["data", "slots", "parameters", "context"]
    if parameter_names != expected:
        raise ProtocolError(
            ErrorCode.EXECUTION_ERROR,
            "算法入口参数必须依次为 data、slots、parameters、context",
        )


def _remove_package_modules(package_root: Path, modules_before: set[str]) -> None:
    """清理本次算法包导入，避免后续任务命中同名模块缓存。"""

    for name in set(sys.modules) - modules_before:
        module = sys.modules.get(name)
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            continue
        try:
            Path(module_file).resolve().relative_to(package_root)
        except ValueError:
            continue
        sys.modules.pop(name, None)
