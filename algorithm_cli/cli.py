"""算法包开发命令行入口。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from algorithm_cli.validator import calculate_package_hash, validate_package


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """执行命令并返回适合脚本判断的退出码。"""

    args = build_parser().parse_args(argv)
    if args.command == "validate":
        report = validate_package(
            args.package_dir,
            check_dependencies=not args.skip_dependencies,
        )
        if args.json:
            print(json.dumps(report.to_payload(), ensure_ascii=False, indent=2, allow_nan=False))
        else:
            state = "通过" if report.is_valid else "未通过"
            package = report.package_id or "未知算法包"
            version = report.version or "未知版本"
            print(f"校验{state}：{package} {version}")
            for issue in report.issues:
                location = f" [{issue.location}]" if issue.location else ""
                print(f"- {issue.level.upper()} {issue.code}{location}: {issue.message}")
        return 0 if report.is_valid else 1
    if args.command == "hash":
        try:
            print(calculate_package_hash(args.package_dir))
        except ValueError as exc:
            print(str(exc))
            return 1
        return 0
    return 2
