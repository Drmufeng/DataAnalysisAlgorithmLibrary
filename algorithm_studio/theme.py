"""工作台主题入口和少量领域专用样式。"""

from __future__ import annotations

import qdarktheme  # type: ignore[import-untyped]
from PySide6.QtWidgets import QApplication

STUDIO_QSS = """
QMainWindow {
    background: palette(window);
}

QToolBar#workspaceToolbar {
    border: 0;
    border-bottom: 1px solid rgba(128, 128, 128, 42);
    padding: 6px 8px;
    spacing: 4px;
}

QToolBar#workspaceToolbar QToolButton {
    padding: 6px 10px;
    border-radius: 5px;
}

QLabel#paneTitle {
    font-size: 15px;
    font-weight: 600;
    padding: 0;
}

QLineEdit#treeSearch {
    min-height: 30px;
    padding-left: 9px;
    border-radius: 5px;
}

QTreeWidget#libraryTree {
    border: 0;
    outline: 0;
    padding: 2px;
}

QTreeWidget#libraryTree::item {
    min-height: 28px;
    padding: 1px 4px;
    border-radius: 4px;
}

QTabWidget#documentTabs::pane {
    border: 0;
    border-top: 1px solid rgba(128, 128, 128, 38);
}

QTabBar::tab {
    min-width: 110px;
    min-height: 30px;
    padding: 4px 12px;
}

QPlainTextEdit#codeEditor {
    border: 0;
    padding: 6px 0;
    selection-background-color: rgba(75, 128, 224, 100);
}

QPlainTextEdit#inspectorJson, QPlainTextEdit#processOutput {
    border: 0;
    padding: 8px;
}

QDockWidget#outputDock {
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}

QStatusBar {
    border-top: 1px solid rgba(128, 128, 128, 38);
}

QLabel#dialogTitle {
    font-size: 20px;
    font-weight: 650;
}

QLabel#dialogSubtitle {
    color: #8F99A5;
    padding-bottom: 4px;
}

QLabel#formSection {
    font-size: 14px;
    font-weight: 600;
    padding-top: 14px;
    padding-bottom: 4px;
}

QLabel#operationPreview {
    padding: 9px 11px;
    border-left: 3px solid #4B80E0;
    background: rgba(75, 128, 224, 22);
}
"""


def apply_theme(application: QApplication, theme: str) -> None:
    """应用 MIT 许可的 qdarktheme 主题和工作台局部样式。"""

    normalized = "light" if theme == "light" else "dark"
    surface = "#202124" if normalized == "dark" else "#F6F7F9"
    text = "#F3F5F7" if normalized == "dark" else "#20242A"
    muted = "#9AA3AE" if normalized == "dark" else "#68717D"
    theme_qss = (
        STUDIO_QSS
        + f"""
QWidget#navigationPane, QWidget#inspectorPane {{
    background: {surface};
}}
QLabel#paneTitle {{
    color: {text};
}}
QLabel#paneSubtitle, QLabel#summaryLabel, QLabel#inspectorMeta {{
    color: {muted};
}}
"""
    )
    qdarktheme.setup_theme(normalized, additional_qss=theme_qss)
    application.setProperty("studioTheme", normalized)
