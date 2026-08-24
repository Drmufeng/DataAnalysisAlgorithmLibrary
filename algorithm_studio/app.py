"""算法库工作台应用入口。"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from PySide6.QtCore import QSettings, QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

from algorithm_studio.main_window import StudioMainWindow
from algorithm_studio.repository import find_project_root, load_studio_library
from algorithm_studio.theme import apply_theme


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dal-studio", description="数据分析算法库桌面工作台")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="算法库项目根目录；默认从当前目录自动定位",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    """创建 Qt 应用并打开当前算法库。"""

    args = build_parser().parse_args(argv)
    QApplication.setOrganizationName("DataAnalysisAlgorithmLibrary")
    QApplication.setApplicationName("AlgorithmStudio")
    application = QApplication(["dal-studio"])
    application.setStyle("Fusion")
    settings = QSettings()
    theme = str(settings.value("appearance/theme", "dark"))
    apply_theme(application, theme)

    try:
        project_root = find_project_root(args.project_root)
        library = load_studio_library(project_root)
    except Exception as exc:  # 应用入口必须把清单读取错误展示给桌面用户。
        QMessageBox.critical(None, "无法打开算法库", str(exc))
        return 1

    window = StudioMainWindow(library)
    window.show()
    if args.screenshot is not None:
        screenshot_path = args.screenshot.expanduser().resolve()
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)

        def capture_and_quit() -> None:
            window.grab().save(str(screenshot_path), "PNG")
            application.quit()

        QTimer.singleShot(500, capture_and_quit)
    elif args.smoke_test:
        QTimer.singleShot(250, application.quit)
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1:]))
