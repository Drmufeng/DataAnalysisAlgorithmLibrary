"""带行号和基础语法高亮的轻量源码编辑器。"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRect, QRegularExpression, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QPainter,
    QPaintEvent,
    QResizeEvent,
    QSyntaxHighlighter,
    QTextCharFormat,
    QTextDocument,
    QTextFormat,
)
from PySide6.QtWidgets import QPlainTextEdit, QTextEdit, QWidget


class _PatternHighlighter(QSyntaxHighlighter):
    """针对 Python、JSON 和 Markdown 的轻量正则高亮。"""

    def __init__(self, document: QTextDocument, suffix: str) -> None:
        super().__init__(document)
        self._rules = self._build_rules(suffix)

    def highlightBlock(self, text: str) -> None:  # noqa: N802 - Qt 虚方法命名
        """高亮当前文本块。"""

        for pattern, text_format in self._rules:
            iterator = pattern.globalMatch(text)
            while iterator.hasNext():
                match = iterator.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), text_format)

    @staticmethod
    def _build_rules(suffix: str) -> list[tuple[QRegularExpression, QTextCharFormat]]:
        keyword = _format("#4B80E0", bold=True)
        string = _format("#4F9D69")
        comment = _format("#7D8799", italic=True)
        number = _format("#C2713A")
        symbol = _format("#9A6AC4")
        key = _format("#3B8FA3")

        if suffix == ".py":
            keywords = (
                "and|as|assert|async|await|break|class|continue|def|del|elif|else|"
                "except|False|finally|for|from|global|if|import|in|is|lambda|None|"
                "nonlocal|not|or|pass|raise|return|True|try|while|with|yield"
            )
            return [
                (QRegularExpression(rf"\b(?:{keywords})\b"), keyword),
                (QRegularExpression(r"\b(?:class|def)\s+([A-Za-z_][A-Za-z0-9_]*)"), symbol),
                (QRegularExpression(r"@[A-Za-z_][A-Za-z0-9_.]*"), symbol),
                (QRegularExpression(r"\b\d+(?:\.\d+)?\b"), number),
                (
                    QRegularExpression(
                        r"(?:\"[^\"\\]*(?:\\.[^\"\\]*)*\"|'[^'\\]*(?:\\.[^'\\]*)*')"
                    ),
                    string,
                ),
                (QRegularExpression(r"#.*$"), comment),
            ]
        if suffix == ".json":
            return [
                (QRegularExpression(r'"(?:[^"\\]|\\.)*"(?=\s*:)'), key),
                (QRegularExpression(r'"(?:[^"\\]|\\.)*"'), string),
                (QRegularExpression(r"\b(?:true|false|null)\b"), keyword),
                (QRegularExpression(r"-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b"), number),
            ]
        if suffix in {".md", ".markdown"}:
            return [
                (QRegularExpression(r"^#{1,6}\s.*$"), keyword),
                (QRegularExpression(r"`[^`]+`"), string),
                (QRegularExpression(r"\*\*[^*]+\*\*"), symbol),
            ]
        return []


def _format(color: str, *, bold: bool = False, italic: bool = False) -> QTextCharFormat:
    text_format = QTextCharFormat()
    text_format.setForeground(QColor(color))
    if bold:
        text_format.setFontWeight(QFont.Weight.DemiBold)
    text_format.setFontItalic(italic)
    return text_format


class _LineNumberArea(QWidget):
    def __init__(self, editor: CodeEditor) -> None:
        super().__init__(editor)
        self._editor = editor

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt 虚方法命名
        return QSize(self._editor.line_number_area_width(), 0)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt 虚方法命名
        self._editor.paint_line_number_area(event)


class CodeEditor(QPlainTextEdit):
    """以真实文件为数据源的轻量代码编辑器。"""

    dirty_changed = Signal(bool)

    def __init__(self, path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.path = path.resolve()
        self.setObjectName("codeEditor")
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        font = QFont("Cascadia Code")
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(10)
        self.setFont(font)
        self.setTabStopDistance(self.fontMetrics().horizontalAdvance(" ") * 4)

        self._line_number_area = _LineNumberArea(self)
        self._highlighter: _PatternHighlighter | None = None
        self.blockCountChanged.connect(self._update_line_number_area_width)
        self.updateRequest.connect(self._update_line_number_area)
        self.cursorPositionChanged.connect(self._highlight_current_line)
        self.document().modificationChanged.connect(self.dirty_changed)
        self._update_line_number_area_width()
        self.load()

    @property
    def is_dirty(self) -> bool:
        return self.document().isModified()

    def load(self) -> None:
        """从磁盘重新读取文件。"""

        content = self.path.read_text(encoding="utf-8")
        self.setPlainText(content)
        self.document().setModified(False)
        self._highlighter = _PatternHighlighter(self.document(), self.path.suffix.lower())

    def save(self) -> None:
        """以 UTF-8 保存当前内容。"""

        self.path.write_text(self.toPlainText(), encoding="utf-8")
        self.document().setModified(False)

    def jump_to_line(self, line_number: int) -> None:
        """把光标移动到指定的 1 基行号。"""

        block = self.document().findBlockByLineNumber(max(0, line_number - 1))
        if not block.isValid():
            return
        cursor = self.textCursor()
        cursor.setPosition(block.position())
        self.setTextCursor(cursor)
        self.centerCursor()
        self.setFocus(Qt.FocusReason.OtherFocusReason)

    def line_number_area_width(self) -> int:
        digits = max(2, len(str(max(1, self.blockCount()))))
        return 12 + self.fontMetrics().horizontalAdvance("9") * digits

    def paint_line_number_area(self, event: QPaintEvent) -> None:
        painter = QPainter(self._line_number_area)
        painter.fillRect(event.rect(), self.palette().alternateBase())
        painter.setPen(self.palette().mid().color())

        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = round(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + round(self.blockBoundingRect(block).height())
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                painter.drawText(
                    0,
                    top,
                    self._line_number_area.width() - 7,
                    self.fontMetrics().height(),
                    Qt.AlignmentFlag.AlignRight,
                    str(block_number + 1),
                )
            block = block.next()
            top = bottom
            bottom = top + round(self.blockBoundingRect(block).height())
            block_number += 1

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - Qt 虚方法命名
        super().resizeEvent(event)
        content_rect = self.contentsRect()
        self._line_number_area.setGeometry(
            QRect(
                content_rect.left(),
                content_rect.top(),
                self.line_number_area_width(),
                content_rect.height(),
            )
        )

    def _update_line_number_area_width(self, _block_count: int = 0) -> None:
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)

    def _update_line_number_area(self, rect: QRect, dy: int) -> None:
        if dy:
            self._line_number_area.scroll(0, dy)
        else:
            self._line_number_area.update(
                0, rect.y(), self._line_number_area.width(), rect.height()
            )
        if rect.contains(self.viewport().rect()):
            self._update_line_number_area_width()

    def _highlight_current_line(self) -> None:
        selection = QTextEdit.ExtraSelection()
        color = self.palette().highlight().color()
        color.setAlpha(28)
        selection.format.setBackground(color)
        selection.format.setProperty(QTextFormat.Property.FullWidthSelection, True)
        selection.cursor = self.textCursor()
        selection.cursor.clearSelection()
        self.setExtraSelections([selection])
