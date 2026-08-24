"""新建算法和方法的线性表单向导。"""

from __future__ import annotations

from typing import cast

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from algorithm_studio.repository import StudioAlgorithm, StudioModule
from algorithm_studio.scaffolding import (
    FieldKind,
    InputMode,
    MethodDraft,
    NewAlgorithmDraft,
    OutputKind,
    ScaffoldingError,
    validate_algorithm_draft,
    validate_method_draft,
)


class MethodForm(QWidget):
    """算法和方法向导共用的方法协议表单。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(18)
        layout.setVerticalSpacing(9)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.method_id = _line_edit("例如 basic、auto 或 robust")
        self.method_id.setText("basic")
        self.display_name = _line_edit("用户看到的中文方法名称")
        self.display_name_en = _line_edit("English method name")
        self.description = _text_edit("说明这个方法完成什么计算")
        self.description_en = _text_edit("Describe what this method calculates")
        layout.addRow("方法编号", self.method_id)
        layout.addRow("中文名称", self.display_name)
        layout.addRow("英文名称", self.display_name_en)
        layout.addRow("中文说明", self.description)
        layout.addRow("英文说明", self.description_en)

        layout.addRow(_section_label("输入与输出"))
        self.input_mode = QComboBox(self)
        self.input_mode.addItem("不选择变量", "none")
        self.input_mode.addItem("选择单个变量", "single")
        self.input_mode.addItem("选择一个或多个变量", "multiple")
        self.input_mode.setCurrentIndex(2)
        self.slot_id = _line_edit("例如 x_columns")
        self.slot_id.setText("x_columns")
        self.slot_display_name = _line_edit("例如分析变量")
        self.slot_display_name.setText("分析变量")
        self.slot_display_name_en = _line_edit("For example: Variables")
        self.slot_display_name_en.setText("Variables")
        self.field_kind = QComboBox(self)
        self.field_kind.addItem("定类或定量均可", "any")
        self.field_kind.addItem("仅定量变量", "quantitative")
        self.field_kind.addItem("仅定类变量", "categorical")
        self.output_kind = QComboBox(self)
        self.output_kind.addItem("结构化分析结果", "analysis")
        self.output_kind.addItem("生成新的数据版本", "data")
        self.deterministic = QCheckBox("相同输入和参数应得到相同结果", self)
        self.deterministic.setChecked(True)

        layout.addRow("变量选择", self.input_mode)
        layout.addRow("槽位编号", self.slot_id)
        layout.addRow("槽位中文名", self.slot_display_name)
        layout.addRow("槽位英文名", self.slot_display_name_en)
        layout.addRow("变量要求", self.field_kind)
        layout.addRow("输出类型", self.output_kind)
        layout.addRow("可复现", self.deterministic)
        self._slot_widgets = (
            self.slot_id,
            self.slot_display_name,
            self.slot_display_name_en,
            self.field_kind,
        )
        self.input_mode.currentIndexChanged.connect(self._sync_slot_state)
        self._sync_slot_state()

    def draft(self) -> MethodDraft:
        """读取表单中的方法协议草稿。"""

        return MethodDraft(
            method_id=self.method_id.text().strip(),
            display_name=self.display_name.text().strip(),
            display_name_en=self.display_name_en.text().strip(),
            description=self.description.toPlainText().strip(),
            description_en=self.description_en.toPlainText().strip(),
            input_mode=cast(InputMode, str(self.input_mode.currentData())),
            slot_id=self.slot_id.text().strip(),
            slot_display_name=self.slot_display_name.text().strip(),
            slot_display_name_en=self.slot_display_name_en.text().strip(),
            field_kind=cast(FieldKind, str(self.field_kind.currentData())),
            output_kind=cast(OutputKind, str(self.output_kind.currentData())),
            deterministic=self.deterministic.isChecked(),
        )

    def _sync_slot_state(self) -> None:
        enabled = self.input_mode.currentData() != "none"
        form = cast(QFormLayout, self.layout())
        for widget in self._slot_widgets:
            widget.setEnabled(enabled)
            label = form.labelForField(widget)
            if label is not None:
                label.setEnabled(enabled)


class NewAlgorithmDialog(QDialog):
    """向现有内部模块增加算法组和首个方法。"""

    def __init__(
        self,
        modules: tuple[StudioModule, ...],
        selected_module_id: str | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("新增算法")
        self.resize(720, 780)
        self.setMinimumSize(620, 640)
        self._modules = {module.manifest.package_id: module for module in modules}
        self._draft: NewAlgorithmDraft | None = None

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(18, 16, 18, 16)
        root_layout.setSpacing(12)
        title = QLabel("新增算法", self)
        title.setObjectName("dialogTitle")
        subtitle = QLabel(
            "选择所属模块并定义算法及首个方法。创建后会同步生成入口源码、测试和版本变更。",
            self,
        )
        subtitle.setObjectName("dialogSubtitle")
        subtitle.setWordWrap(True)
        root_layout.addWidget(title)
        root_layout.addWidget(subtitle)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        content = QWidget(scroll)
        form = QFormLayout(content)
        form.setContentsMargins(4, 8, 12, 8)
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(9)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        form.addRow(_section_label("算法信息"))
        self.module = QComboBox(content)
        for module in modules:
            self.module.addItem(
                f"{module.manifest.package_name} · {module.manifest.package_id}",
                module.manifest.package_id,
            )
        self.algorithm_id = _line_edit("例如 robust_summary")
        self.display_name = _line_edit("用户端显示的中文算法名称")
        self.display_name_en = _line_edit("English algorithm name")
        self.category = _line_edit("稳定的英文分类编号")
        self.category_display_name = _line_edit("分类中文名称")
        self.category_display_name_en = _line_edit("English category name")
        self.description = _text_edit("说明这个算法组解决什么问题")
        self.description_en = _text_edit("Describe the purpose of this algorithm")
        form.addRow("所属模块", self.module)
        form.addRow("算法编号", self.algorithm_id)
        form.addRow("中文名称", self.display_name)
        form.addRow("英文名称", self.display_name_en)
        form.addRow("分类编号", self.category)
        form.addRow("分类中文名", self.category_display_name)
        form.addRow("分类英文名", self.category_display_name_en)
        form.addRow("中文说明", self.description)
        form.addRow("英文说明", self.description_en)

        form.addRow(_section_label("首个方法"))
        self.method_form = MethodForm(content)
        form.addRow(self.method_form)
        scroll.setWidget(content)
        root_layout.addWidget(scroll, 1)

        self.preview = QLabel(self)
        self.preview.setObjectName("operationPreview")
        root_layout.addWidget(self.preview)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("创建算法")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root_layout.addWidget(buttons)

        if selected_module_id is not None:
            index = self.module.findData(selected_module_id)
            if index >= 0:
                self.module.setCurrentIndex(index)
        self.module.currentIndexChanged.connect(self._sync_category_defaults)
        self.algorithm_id.textChanged.connect(self._update_preview)
        self.method_form.method_id.textChanged.connect(self._update_preview)
        self._sync_category_defaults()
        self._update_preview()

    @property
    def selected_module_id(self) -> str:
        return str(self.module.currentData())

    @property
    def draft(self) -> NewAlgorithmDraft:
        if self._draft is None:
            raise RuntimeError("向导尚未确认")
        return self._draft

    def accept(self) -> None:
        draft = NewAlgorithmDraft(
            algorithm_id=self.algorithm_id.text().strip(),
            display_name=self.display_name.text().strip(),
            display_name_en=self.display_name_en.text().strip(),
            category=self.category.text().strip(),
            category_display_name=self.category_display_name.text().strip(),
            category_display_name_en=self.category_display_name_en.text().strip(),
            description=self.description.toPlainText().strip(),
            description_en=self.description_en.toPlainText().strip(),
            method=self.method_form.draft(),
        )
        try:
            validate_algorithm_draft(draft)
        except ScaffoldingError as exc:
            QMessageBox.warning(self, "信息不完整", str(exc))
            return
        self._draft = draft
        super().accept()

    def _sync_category_defaults(self) -> None:
        module = self._modules.get(self.selected_module_id)
        if module is None:
            return
        if module.algorithms:
            algorithm = module.algorithms[0].manifest
            self.category.setText(algorithm.category)
            self.category_display_name.setText(
                algorithm.category_display_name or module.manifest.package_name
            )
            self.category_display_name_en.setText(
                algorithm.category_display_name_en
                or module.manifest.package_name_en
                or module.manifest.package_id
            )
        else:
            self.category.setText(module.manifest.package_id)
            self.category_display_name.setText(module.manifest.package_name)
            self.category_display_name_en.setText(
                module.manifest.package_name_en or module.manifest.package_id
            )

    def _update_preview(self) -> None:
        algorithm_id = self.algorithm_id.text().strip() or "algorithm_id"
        method_id = self.method_form.method_id.text().strip() or "method_id"
        self.preview.setText(f"将创建：{algorithm_id}.{method_id}")


class NewMethodDialog(QDialog):
    """向选中的算法组增加方法。"""

    def __init__(
        self,
        module: StudioModule,
        algorithm: StudioAlgorithm,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("新增方法")
        self.resize(680, 690)
        self.setMinimumSize(600, 580)
        self._draft: MethodDraft | None = None
        self.module_id = module.manifest.package_id
        self.algorithm_id = algorithm.manifest.algorithm_id

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)
        title = QLabel("新增方法", self)
        title.setObjectName("dialogTitle")
        context = QLabel(
            f"{module.manifest.package_name} / {algorithm.manifest.display_name} "
            f"· {algorithm.manifest.algorithm_id}",
            self,
        )
        context.setObjectName("dialogSubtitle")
        layout.addWidget(title)
        layout.addWidget(context)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.method_form = MethodForm(scroll)
        method_id = "basic"
        existing_ids = {method.manifest.method_id for method in algorithm.methods}
        counter = 2
        while method_id in existing_ids:
            method_id = f"method_{counter}"
            counter += 1
        self.method_form.method_id.setText(method_id)
        scroll.setWidget(self.method_form)
        layout.addWidget(scroll, 1)

        self.preview = QLabel(self)
        self.preview.setObjectName("operationPreview")
        layout.addWidget(self.preview)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("创建方法")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.method_form.method_id.textChanged.connect(self._update_preview)
        self._update_preview()

    @property
    def draft(self) -> MethodDraft:
        if self._draft is None:
            raise RuntimeError("向导尚未确认")
        return self._draft

    def accept(self) -> None:
        draft = self.method_form.draft()
        try:
            validate_method_draft(draft)
        except ScaffoldingError as exc:
            QMessageBox.warning(self, "信息不完整", str(exc))
            return
        self._draft = draft
        super().accept()

    def _update_preview(self) -> None:
        method_id = self.method_form.method_id.text().strip() or "method_id"
        self.preview.setText(f"将创建：{self.algorithm_id}.{method_id}")


def _line_edit(placeholder: str) -> QLineEdit:
    widget = QLineEdit()
    widget.setPlaceholderText(placeholder)
    widget.setClearButtonEnabled(True)
    return widget


def _text_edit(placeholder: str) -> QPlainTextEdit:
    widget = QPlainTextEdit()
    widget.setPlaceholderText(placeholder)
    widget.setFixedHeight(62)
    return widget


def _section_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("formSection")
    label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    return label
