"""算法包清单读取工具。"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from algorithm_sdk.errors import ErrorCode, ProtocolError
from algorithm_sdk.models import LibraryManifest, PackageManifest


def _read_json_manifest(path: Path, *, display_name: str) -> object:
    """读取一个 UTF-8 JSON 清单，并把解析错误转换为统一协议错误。"""

    if not path.is_file():
        raise ProtocolError(
            ErrorCode.DATA_ERROR,
            f"缺少 {display_name}",
            details={"file": display_name},
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise ProtocolError(
            ErrorCode.DATA_ERROR,
            f"{display_name} 必须使用 UTF-8 编码",
            details={"file": display_name},
        ) from exc
    except json.JSONDecodeError as exc:
        raise ProtocolError(
            ErrorCode.DATA_ERROR,
            f"{display_name} 不是合法 JSON：第 {exc.lineno} 行第 {exc.colno} 列",
            details={"line": exc.lineno, "column": exc.colno},
        ) from exc


def load_manifest(package_dir: Path) -> PackageManifest:
    """从算法包根目录读取并校验 UTF-8 JSON 清单。"""

    manifest_path = package_dir / "manifest.json"
    payload = _read_json_manifest(manifest_path, display_name="manifest.json")

    try:
        return PackageManifest.model_validate(payload)
    except ValidationError as exc:
        first_error = exc.errors(include_url=False)[0]
        location = ".".join(str(item) for item in first_error["loc"])
        raise ProtocolError(
            ErrorCode.DATA_ERROR,
            f"manifest.json 结构不正确：{location} {first_error['msg']}",
            details={"location": location},
        ) from exc


def load_library_manifest(library_root: Path) -> LibraryManifest:
    """读取完整算法库发行包的根清单。"""

    manifest_name = "library_manifest.json"
    payload = _read_json_manifest(library_root / manifest_name, display_name=manifest_name)
    try:
        return LibraryManifest.model_validate(payload)
    except ValidationError as exc:
        first_error = exc.errors(include_url=False)[0]
        location = ".".join(str(item) for item in first_error["loc"])
        raise ProtocolError(
            ErrorCode.DATA_ERROR,
            f"{manifest_name} 结构不正确：{location} {first_error['msg']}",
            details={"location": location},
        ) from exc
