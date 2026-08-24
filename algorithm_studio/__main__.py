"""允许使用 ``python -m algorithm_studio`` 启动工作台。"""

from __future__ import annotations


def main() -> int:
    """延迟导入 Qt，并在未安装可选依赖时给出明确提示。"""

    try:
        from algorithm_studio.app import run
    except ModuleNotFoundError as exc:
        if exc.name in {"PySide6", "qdarktheme"}:
            raise SystemExit(
                '算法库工作台依赖尚未安装，请先执行 pip install -e ".[studio]"'
            ) from exc
        raise
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
