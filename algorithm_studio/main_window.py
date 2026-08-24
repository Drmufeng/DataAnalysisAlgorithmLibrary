"""算法库工作台主窗口。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from PySide6.QtCore import QFileSystemWatcher, QProcess, QProcessEnvironment, QSettings, Qt
from PySide6.QtGui import QAction, QCloseEvent, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDockWidget,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QSizePolicy,
    QSplitter,
    QStyle,
    QTabWidget,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from algorithm_studio.editor import CodeEditor
from algorithm_studio.repository import (
    StudioAlgorithm,
    StudioLibrary,
    StudioMethod,
    StudioModule,
    load_studio_library,
)
from algorithm_studio.scaffolding import (
    ScaffoldingError,
    ScaffoldingResult,
    create_algorithm,
    create_method,
)
from algorithm_studio.theme import apply_theme
from algorithm_studio.wizard import NewAlgorithmDialog, NewMethodDialog

TreeNode = StudioLibrary | StudioModule | StudioAlgorithm | StudioMethod


class StudioMainWindow(QMainWindow):
    """浏览、编辑、校验和测试当前算法库源码。"""

    def __init__(self, library: StudioLibrary) -> None:
        super().__init__()
        self.library = library
        self._editors: dict[Path, CodeEditor] = {}
        self._process = QProcess(self)
        self._settings = QSettings()
        self._watcher = QFileSystemWatcher(self)
        self._process_actions: list[QAction] = []

        self.setWindowTitle(f"算法库工作台 · {library.manifest.library_name}")
        self.resize(1480, 900)
        self.setMinimumSize(1080, 680)
        self._build_toolbar()
        self._build_workspace()
        self._build_output_dock()
        self._connect_process()
        self._watcher.fileChanged.connect(self._on_external_file_changed)
        self._populate_tree()
        self.statusBar().showMessage(str(self.library.project_root))

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("工作区", self)
        toolbar.setObjectName("workspaceToolbar")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(toolbar)

        new_algorithm_action = self._action(
            "新增算法",
            QStyle.StandardPixmap.SP_FileDialogNewFolder,
            self.create_new_algorithm,
            "Ctrl+N",
        )
        new_method_action = self._action(
            "新增方法",
            QStyle.StandardPixmap.SP_FileIcon,
            self.create_new_method,
            "Ctrl+Shift+N",
        )
        toolbar.addActions([new_algorithm_action, new_method_action])
        toolbar.addSeparator()

        refresh_action = self._action(
            "刷新目录",
            QStyle.StandardPixmap.SP_BrowserReload,
            self.reload_library,
            "F5",
        )
        save_action = self._action(
            "保存",
            QStyle.StandardPixmap.SP_DialogSaveButton,
            self.save_current,
            "Ctrl+S",
        )
        save_all_action = self._action(
            "全部保存",
            QStyle.StandardPixmap.SP_DialogApplyButton,
            self.save_all,
            "Ctrl+Shift+S",
        )
        toolbar.addActions([refresh_action, save_action, save_all_action])
        toolbar.addSeparator()

        validate_action = self._action(
            "整库校验",
            QStyle.StandardPixmap.SP_DialogYesButton,
            self.run_validation,
            "Ctrl+Alt+V",
        )
        test_action = self._action(
            "运行测试",
            QStyle.StandardPixmap.SP_MediaPlay,
            self.run_tests,
            "Ctrl+Alt+T",
        )
        build_action = self._action(
            "构建发行包",
            QStyle.StandardPixmap.SP_DriveHDIcon,
            self.run_build,
            "Ctrl+Alt+B",
        )
        self._process_actions = [validate_action, test_action, build_action]
        toolbar.addActions(self._process_actions)

        spacer = QWidget(toolbar)
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)
        self.theme_action = self._action(
            "切换主题",
            QStyle.StandardPixmap.SP_DesktopIcon,
            self.toggle_theme,
            "Ctrl+Alt+D",
        )
        toolbar.addAction(self.theme_action)

    def _build_workspace(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)

        navigation = QWidget(splitter)
        navigation.setObjectName("navigationPane")
        navigation_layout = QVBoxLayout(navigation)
        navigation_layout.setContentsMargins(14, 14, 10, 10)
        navigation_layout.setSpacing(9)
        navigation_title = QLabel("算法结构", navigation)
        navigation_title.setObjectName("paneTitle")
        self.search_box = QLineEdit(navigation)
        self.search_box.setObjectName("treeSearch")
        self.search_box.setPlaceholderText("搜索模块、算法或方法")
        self.search_box.setClearButtonEnabled(True)
        self.search_box.textChanged.connect(self._filter_tree)
        self.tree = QTreeWidget(navigation)
        self.tree.setObjectName("libraryTree")
        self.tree.setHeaderLabels(["名称", "标识"])
        self.tree.setAlternatingRowColors(False)
        self.tree.setUniformRowHeights(True)
        self.tree.setColumnWidth(0, 190)
        self.tree.currentItemChanged.connect(self._on_tree_selection)
        self.summary_label = QLabel(navigation)
        self.summary_label.setObjectName("summaryLabel")
        navigation_layout.addWidget(navigation_title)
        navigation_layout.addWidget(self.search_box)
        navigation_layout.addWidget(self.tree, 1)
        navigation_layout.addWidget(self.summary_label)

        self.document_tabs = QTabWidget(splitter)
        self.document_tabs.setObjectName("documentTabs")
        self.document_tabs.setTabsClosable(True)
        self.document_tabs.setMovable(True)
        self.document_tabs.setDocumentMode(True)
        self.document_tabs.tabCloseRequested.connect(self._close_tab)
        self.document_tabs.currentChanged.connect(self._on_current_tab_changed)

        inspector = QWidget(splitter)
        inspector.setObjectName("inspectorPane")
        inspector_layout = QVBoxLayout(inspector)
        inspector_layout.setContentsMargins(12, 14, 12, 10)
        inspector_layout.setSpacing(7)
        self.inspector_title = QLabel("未选择算法", inspector)
        self.inspector_title.setObjectName("paneTitle")
        self.inspector_meta = QLabel("从左侧目录选择一个节点", inspector)
        self.inspector_meta.setObjectName("inspectorMeta")
        self.inspector_meta.setWordWrap(True)
        self.inspector_json = QPlainTextEdit(inspector)
        self.inspector_json.setObjectName("inspectorJson")
        self.inspector_json.setReadOnly(True)
        self.inspector_json.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        inspector_layout.addWidget(self.inspector_title)
        inspector_layout.addWidget(self.inspector_meta)
        inspector_layout.addWidget(self.inspector_json, 1)

        splitter.addWidget(navigation)
        splitter.addWidget(self.document_tabs)
        splitter.addWidget(inspector)
        splitter.setSizes([280, 870, 330])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        self.setCentralWidget(splitter)

    def _build_output_dock(self) -> None:
        self.output_dock = QDockWidget("任务输出", self)
        self.output_dock.setObjectName("outputDock")
        self.output_dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        self.process_output = QPlainTextEdit(self.output_dock)
        self.process_output.setObjectName("processOutput")
        self.process_output.setReadOnly(True)
        self.process_output.setMaximumBlockCount(5000)
        self.process_output.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.output_dock.setWidget(self.process_output)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.output_dock)
        self.resizeDocks([self.output_dock], [190], Qt.Orientation.Vertical)
        self.output_dock.hide()

    def _connect_process(self) -> None:
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUTF8", "1")
        environment.insert("PYTHONIOENCODING", "utf-8")
        self._process.setProcessEnvironment(environment)
        self._process.readyReadStandardOutput.connect(self._append_process_output)
        self._process.started.connect(self._on_process_started)
        self._process.finished.connect(self._on_process_finished)

    def _populate_tree(self, selected_operation_key: str | None = None) -> None:
        self.tree.clear()
        requested_item: QTreeWidgetItem | None = None
        root = self._tree_item(
            self.library.manifest.library_name,
            f"v{self.library.manifest.version}",
            self.library,
        )
        self.tree.addTopLevelItem(root)
        for module in self.library.modules:
            module_item = self._tree_item(
                module.manifest.package_name,
                module.manifest.package_id,
                module,
            )
            root.addChild(module_item)
            for algorithm in module.algorithms:
                algorithm_item = self._tree_item(
                    algorithm.manifest.display_name,
                    algorithm.manifest.algorithm_id,
                    algorithm,
                )
                module_item.addChild(algorithm_item)
                for method in algorithm.methods:
                    method_item = self._tree_item(
                        method.manifest.display_name,
                        method.manifest.method_id,
                        method,
                    )
                    method_item.setToolTip(0, method.manifest.description)
                    algorithm_item.addChild(method_item)
                    if method.operation_key == selected_operation_key:
                        requested_item = method_item
        root.setExpanded(True)
        for index in range(root.childCount()):
            child = root.child(index)
            if child is not None:
                child.setExpanded(True)
        initial_item = root
        first_module = root.child(0)
        if first_module is not None:
            first_algorithm = first_module.child(0)
            if first_algorithm is not None:
                first_method = first_algorithm.child(0)
                initial_item = first_method or first_algorithm
        self.tree.setCurrentItem(requested_item or initial_item)
        self.summary_label.setText(
            f"{len(self.library.modules)} 模块 · "
            f"{self.library.algorithm_count} 算法 · {self.library.method_count} 方法"
        )

    def _tree_item(self, name: str, identifier: str, node: TreeNode) -> QTreeWidgetItem:
        item = QTreeWidgetItem([name, identifier])
        item.setData(0, Qt.ItemDataRole.UserRole, node)
        return item

    def _on_tree_selection(
        self,
        current: QTreeWidgetItem | None,
        _previous: QTreeWidgetItem | None,
    ) -> None:
        if current is None:
            return
        node = current.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(node, StudioLibrary):
            self._show_inspector(
                node.manifest.library_name,
                f"完整发行版 v{node.manifest.version}",
                node.manifest.model_dump(mode="json"),
            )
            self.open_document(node.manifest_path)
        elif isinstance(node, StudioModule):
            self._show_inspector(
                node.manifest.package_name,
                f"内部模块 · {node.manifest.package_id} · v{node.manifest.version}",
                node.manifest.model_dump(mode="json", exclude={"algorithms"}),
            )
            self.open_document(node.manifest_path)
        elif isinstance(node, StudioAlgorithm):
            self._show_inspector(
                node.manifest.display_name,
                f"算法组 · {node.manifest.algorithm_id} · {len(node.methods)} 个方法",
                node.manifest.model_dump(mode="json"),
            )
            if node.methods:
                self.open_document(node.methods[0].source_path, node.methods[0].function_line)
        elif isinstance(node, StudioMethod):
            self._show_inspector(
                node.manifest.display_name,
                f"方法 · {node.operation_key} · {node.manifest.entry}",
                node.manifest.model_dump(mode="json"),
            )
            self.open_document(node.source_path, node.function_line)

    def _show_inspector(self, title: str, metadata: str, payload: dict[str, Any]) -> None:
        self.inspector_title.setText(title)
        self.inspector_meta.setText(metadata)
        self.inspector_json.setPlainText(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        )

    def open_document(self, path: Path, line_number: int = 1) -> None:
        normalized = path.resolve()
        editor = self._editors.get(normalized)
        if editor is None:
            try:
                editor = CodeEditor(normalized, self.document_tabs)
            except (OSError, UnicodeError) as exc:
                QMessageBox.critical(self, "无法打开文件", str(exc))
                return
            editor.dirty_changed.connect(
                lambda dirty, file_path=normalized: self._update_tab_title(file_path, dirty)
            )
            self._editors[normalized] = editor
            relative_name = self._display_path(normalized)
            index = self.document_tabs.addTab(editor, relative_name)
            self.document_tabs.setTabToolTip(index, str(normalized))
            self._watcher.addPath(str(normalized))
        self.document_tabs.setCurrentWidget(editor)
        editor.jump_to_line(line_number)

    def save_current(self) -> None:
        editor = self._current_editor()
        if editor is not None:
            self._save_editor(editor)

    def save_all(self) -> None:
        for editor in self._editors.values():
            if editor.is_dirty:
                self._save_editor(editor)

    def _save_editor(self, editor: CodeEditor) -> bool:
        try:
            editor.save()
        except OSError as exc:
            QMessageBox.critical(self, "保存失败", f"{editor.path}\n\n{exc}")
            return False
        self.statusBar().showMessage(f"已保存 {self._display_path(editor.path)}", 3500)
        return True

    def reload_library(self) -> None:
        if not self._confirm_discard_dirty("刷新目录会重新读取清单和已打开文件。"):
            return
        selected_operation_key = self._selected_operation_key()
        try:
            library = load_studio_library(self.library.project_root)
            for editor in self._editors.values():
                editor.load()
        except Exception as exc:  # 窗口边界展示完整、可操作的读取错误。
            QMessageBox.critical(self, "刷新失败", str(exc))
            return
        self.library = library
        self._populate_tree(selected_operation_key)
        self.statusBar().showMessage("算法目录已刷新", 3500)

    def create_new_algorithm(self) -> None:
        """打开新算法向导，并以原子事务写入全部骨架文件。"""

        selected_module_id = self._selected_module_id()
        dialog = NewAlgorithmDialog(self.library.modules, selected_module_id, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        if not self._confirm_discard_dirty("创建算法后需要刷新工作区，未保存修改将被丢弃。"):
            return
        try:
            result = create_algorithm(
                self.library,
                dialog.selected_module_id,
                dialog.draft,
            )
        except ScaffoldingError as exc:
            QMessageBox.critical(self, "创建算法失败", str(exc))
            return
        self._finish_scaffolding(result)

    def create_new_method(self) -> None:
        """为当前算法组创建一个方法、源码和测试骨架。"""

        context = self._selected_algorithm_context()
        if context is None:
            QMessageBox.information(self, "请选择算法", "请先在左侧选择一个算法或具体方法。")
            return
        module, algorithm = context
        dialog = NewMethodDialog(module, algorithm, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        if not self._confirm_discard_dirty("创建方法后需要刷新工作区，未保存修改将被丢弃。"):
            return
        try:
            result = create_method(
                self.library,
                module.manifest.package_id,
                algorithm.manifest.algorithm_id,
                dialog.draft,
            )
        except ScaffoldingError as exc:
            QMessageBox.critical(self, "创建方法失败", str(exc))
            return
        self._finish_scaffolding(result)

    def _finish_scaffolding(self, result: ScaffoldingResult) -> None:
        try:
            library = load_studio_library(self.library.project_root)
            for editor in self._editors.values():
                if editor.path.is_file():
                    editor.load()
        except Exception as exc:
            QMessageBox.critical(self, "刷新工作区失败", str(exc))
            return
        self.library = library
        self._populate_tree(result.operation_key)
        changed = "\n".join(
            f"• {path.relative_to(self.library.project_root).as_posix()}"
            for path in result.changed_paths
        )
        QMessageBox.information(
            self,
            "创建完成",
            f"已创建 {result.operation_key}\n"
            f"算法库版本：v{result.library_version}\n"
            f"模块版本：v{result.module_version}\n\n"
            f"变更文件：\n{changed}",
        )
        self.statusBar().showMessage(f"已创建 {result.operation_key}", 6000)

    def run_validation(self) -> None:
        self._run_command(
            "整库校验",
            ["-m", "algorithm_cli", "validate-library", "packages"],
        )

    def run_tests(self) -> None:
        self._run_command("测试", ["-m", "pytest"])

    def run_build(self) -> None:
        self._run_command("构建", ["-m", "build"])

    def _run_command(self, title: str, arguments: list[str]) -> None:
        if self._process.state() != QProcess.ProcessState.NotRunning:
            self.statusBar().showMessage("已有任务正在运行", 3000)
            return
        self.save_all()
        command = " ".join([sys.executable, *arguments])
        self.process_output.clear()
        self.process_output.appendPlainText(f"[{title}]\n$ {command}\n")
        self.output_dock.show()
        self._process.setWorkingDirectory(str(self.library.project_root))
        self._process.setProgram(sys.executable)
        self._process.setArguments(arguments)
        self._process.setProperty("studioTaskTitle", title)
        self._process.start()

    def _append_process_output(self) -> None:
        output = bytes(self._process.readAllStandardOutput().data()).decode(
            "utf-8", errors="replace"
        )
        self.process_output.moveCursor(QTextCursor.MoveOperation.End)
        self.process_output.insertPlainText(output)
        self.process_output.ensureCursorVisible()

    def _on_process_started(self) -> None:
        title = str(self._process.property("studioTaskTitle"))
        for action in self._process_actions:
            action.setEnabled(False)
        self.statusBar().showMessage(f"正在运行：{title}")

    def _on_process_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        title = str(self._process.property("studioTaskTitle"))
        state = "完成" if exit_code == 0 else f"失败（退出码 {exit_code}）"
        self.process_output.appendPlainText(f"\n[{title}{state}]")
        for action in self._process_actions:
            action.setEnabled(True)
        self.statusBar().showMessage(f"{title}{state}", 6000)

    def toggle_theme(self) -> None:
        application = QApplication.instance()
        if not isinstance(application, QApplication):
            return
        current = str(application.property("studioTheme") or "dark")
        target = "light" if current == "dark" else "dark"
        apply_theme(application, target)
        self._settings.setValue("appearance/theme", target)
        self.statusBar().showMessage(f"已切换为{'浅色' if target == 'light' else '深色'}主题", 3000)

    def _filter_tree(self, query: str) -> None:
        normalized = query.strip().casefold()

        def visit(item: QTreeWidgetItem) -> bool:
            child_matches = False
            for index in range(item.childCount()):
                child = item.child(index)
                if child is not None and visit(child):
                    child_matches = True
            own_text = f"{item.text(0)} {item.text(1)}".casefold()
            matched = not normalized or normalized in own_text or child_matches
            item.setHidden(not matched)
            if normalized and child_matches:
                item.setExpanded(True)
            return matched

        for index in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(index)
            if item is not None:
                visit(item)

    def _close_tab(self, index: int) -> None:
        editor = self.document_tabs.widget(index)
        if not isinstance(editor, CodeEditor):
            self.document_tabs.removeTab(index)
            return
        if editor.is_dirty:
            answer = QMessageBox.question(
                self,
                "文件尚未保存",
                f"是否保存 {self._display_path(editor.path)}？",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
            )
            if answer == QMessageBox.StandardButton.Cancel:
                return
            if answer == QMessageBox.StandardButton.Save and not self._save_editor(editor):
                return
        self._watcher.removePath(str(editor.path))
        self._editors.pop(editor.path, None)
        self.document_tabs.removeTab(index)
        editor.deleteLater()

    def _on_current_tab_changed(self, _index: int) -> None:
        editor = self._current_editor()
        if editor is not None:
            self.statusBar().showMessage(self._display_path(editor.path))

    def _on_external_file_changed(self, raw_path: str) -> None:
        """在 IDE 修改已打开文件后刷新内容，同时保护工作台中的未保存修改。"""

        path = Path(raw_path).resolve()
        editor = self._editors.get(path)
        if editor is None:
            return
        if editor.is_dirty:
            self.statusBar().showMessage(
                f"磁盘上的 {self._display_path(path)} 已变化；当前标签有未保存修改",
                7000,
            )
        else:
            try:
                editor.load()
                self.statusBar().showMessage(
                    f"已重新读取 {self._display_path(path)}",
                    3500,
                )
            except (OSError, UnicodeError) as exc:
                self.statusBar().showMessage(f"重新读取文件失败：{exc}", 7000)
        if path.is_file() and raw_path not in self._watcher.files():
            self._watcher.addPath(raw_path)

    def _update_tab_title(self, path: Path, dirty: bool) -> None:
        editor = self._editors.get(path)
        if editor is None:
            return
        index = self.document_tabs.indexOf(editor)
        if index >= 0:
            prefix = "● " if dirty else ""
            self.document_tabs.setTabText(index, prefix + self._display_path(path))

    def _current_editor(self) -> CodeEditor | None:
        widget = self.document_tabs.currentWidget()
        return widget if isinstance(widget, CodeEditor) else None

    def _selected_node(self) -> TreeNode | None:
        item = self.tree.currentItem()
        if item is None:
            return None
        node = item.data(0, Qt.ItemDataRole.UserRole)
        return (
            node
            if isinstance(node, (StudioLibrary, StudioModule, StudioAlgorithm, StudioMethod))
            else None
        )

    def _selected_module_id(self) -> str | None:
        node = self._selected_node()
        if isinstance(node, StudioModule):
            return node.manifest.package_id
        if isinstance(node, (StudioAlgorithm, StudioMethod)):
            return node.module_id
        return None

    def _selected_operation_key(self) -> str | None:
        node = self._selected_node()
        return node.operation_key if isinstance(node, StudioMethod) else None

    def _selected_algorithm_context(self) -> tuple[StudioModule, StudioAlgorithm] | None:
        node = self._selected_node()
        if isinstance(node, StudioAlgorithm):
            module_id = node.module_id
            algorithm_id = node.manifest.algorithm_id
        elif isinstance(node, StudioMethod):
            module_id = node.module_id
            algorithm_id = node.algorithm_id
        else:
            return None
        module = next(
            (item for item in self.library.modules if item.manifest.package_id == module_id),
            None,
        )
        if module is None:
            return None
        algorithm = next(
            (item for item in module.algorithms if item.manifest.algorithm_id == algorithm_id),
            None,
        )
        return (module, algorithm) if algorithm is not None else None

    def _display_path(self, path: Path) -> str:
        try:
            return path.relative_to(self.library.project_root).as_posix()
        except ValueError:
            return path.name

    def _confirm_discard_dirty(self, explanation: str) -> bool:
        if not any(editor.is_dirty for editor in self._editors.values()):
            return True
        answer = QMessageBox.warning(
            self,
            "存在未保存修改",
            explanation,
            QMessageBox.StandardButton.SaveAll
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Cancel:
            return False
        if answer == QMessageBox.StandardButton.SaveAll:
            self.save_all()
        return True

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt 虚方法命名
        if self._process.state() != QProcess.ProcessState.NotRunning:
            answer = QMessageBox.question(
                self,
                "任务正在运行",
                "后台校验或测试尚未结束，确定退出吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._process.kill()
        if not self._confirm_discard_dirty("退出会丢弃尚未保存的源码修改。"):
            event.ignore()
            return
        event.accept()

    def _action(
        self,
        text: str,
        standard_icon: QStyle.StandardPixmap,
        callback: Any,
        shortcut: str,
    ) -> QAction:
        action = QAction(self.style().standardIcon(standard_icon), text, self)
        action.setShortcut(shortcut)
        action.triggered.connect(callback)
        return action
