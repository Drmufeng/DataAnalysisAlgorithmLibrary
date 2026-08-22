"""算法包开发命令行入口。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from algorithm_cli.validator import (
    calculate_library_hash,
    calculate_package_hash,
    validate_library,
    validate_package,
)
from algorithm_sdk.manifest import load_library_manifest


def build_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。"""

    parser = argparse.ArgumentParser(prog="dal-algorithm", description="数据分析算法包工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="校验算法包")
    validate_parser.add_argument("package_dir", type=Path, help="算法包目录")
    validate_parser.add_argument(
        "--skip-dependencies",
        action="store_true",
        help="只检查结构，不检查当前环境依赖",
    )
    validate_parser.add_argument("--json", action="store_true", help="输出 JSON 校验结果")

    hash_parser = subparsers.add_parser("hash", help="计算算法包内容哈希")
    hash_parser.add_argument("package_dir", type=Path, help="算法包目录")

    validate_library_parser = subparsers.add_parser(
        "validate-library",
        help="校验完整算法库发行包",
    )
    validate_library_parser.add_argument("library_root", type=Path, help="根清单所在目录")
    validate_library_parser.add_argument(
        "--skip-dependencies",
        action="store_true",
        help="只检查结构，不检查当前环境依赖",
    )
    validate_library_parser.add_argument(
        "--json",
        action="store_true",
        help="输出 JSON 校验结果",
    )

    inspect_library_parser = subparsers.add_parser(
        "inspect-library",
        help="读取完整算法库发行信息",
    )
    inspect_library_parser.add_argument("library_root", type=Path, help="根清单所在目录")

    hash_library_parser = subparsers.add_parser(
        "hash-library",
        help="计算完整算法库内容哈希",
    )
    hash_library_parser.add_argument("library_root", type=Path, help="根清单所在目录")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """执行命令并返回适合脚本判断的退出码。"""

    args = build_parser().parse_args(argv)
    if args.command == "validate":
        package_report = validate_package(
            args.package_dir,
            check_dependencies=not args.skip_dependencies,
        )
        if args.json:
            print(
                json.dumps(
                    package_report.to_payload(),
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
            )
        else:
            state = "通过" if package_report.is_valid else "未通过"
            package = package_report.package_id or "未知算法包"
            version = package_report.version or "未知版本"
            print(f"校验{state}：{package} {version}")
            for issue in package_report.issues:
                location = f" [{issue.location}]" if issue.location else ""
                print(f"- {issue.level.upper()} {issue.code}{location}: {issue.message}")
        return 0 if package_report.is_valid else 1
    if args.command == "hash":
        try:
            print(calculate_package_hash(args.package_dir))
        except ValueError as exc:
            print(str(exc))
            return 1
        return 0
    if args.command == "validate-library":
        library_report = validate_library(
            args.library_root,
            check_dependencies=not args.skip_dependencies,
        )
        if args.json:
            print(
                json.dumps(
                    library_report.to_payload(),
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
            )
        else:
            state = "通过" if library_report.is_valid else "未通过"
            library = library_report.library_id or "未知算法库"
            version = library_report.version or "未知版本"
            print(f"整库校验{state}：{library} {version}")
            print(
                f"模块 {library_report.module_count} 个，"
                f"算法 {library_report.algorithm_count} 个，"
                f"方法 {library_report.method_count} 个"
            )
            for issue in library_report.issues:
                location = f" [{issue.location}]" if issue.location else ""
                print(f"- {issue.level.upper()} {issue.code}{location}: {issue.message}")
        return 0 if library_report.is_valid else 1
    if args.command == "inspect-library":
        try:
            manifest = load_library_manifest(args.library_root)
            inspect_report = validate_library(args.library_root, check_dependencies=False)
        except Exception as exc:  # CLI 边界把读取错误转换为可被脚本判断的退出码。
            print(str(exc))
            return 1
        payload = manifest.model_dump(mode="json")
        payload["summary"] = {
            "module_count": inspect_report.module_count,
            "algorithm_count": inspect_report.algorithm_count,
            "method_count": inspect_report.method_count,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if inspect_report.is_valid else 1
    if args.command == "hash-library":
        try:
            print(calculate_library_hash(args.library_root))
        except ValueError as exc:
            print(str(exc))
            return 1
        return 0
    return 2
