"""算法包清单读取工具。"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from algorithm_sdk.errors import ErrorCode, ProtocolError
from algorithm_sdk.models import PackageManifest


def load_manifest(package_dir: Path) -> PackageManifest:
    """从算法包根目录读取并校验 UTF-8 JSON 清单。"""

    manifest_path = package_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ProtocolError(
            ErrorCode.DATA_ERROR,
            "算法包缺少 manifest.json",
            details={"file": "manifest.json"},
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise ProtocolError(
            ErrorCode.DATA_ERROR,
            "manifest.json 必须使用 UTF-8 编码",
            details={"file": "manifest.json"},
        ) from exc
    except json.JSONDecodeError as exc:
        raise ProtocolError(
            ErrorCode.DATA_ERROR,
            f"manifest.json 不是合法 JSON：第 {exc.lineno} 行第 {exc.colno} 列",
            details={"line": exc.lineno, "column": exc.colno},
        ) from exc

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
