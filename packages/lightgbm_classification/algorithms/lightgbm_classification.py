"""支持二分类和多分类的 LightGBM 分类。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pandas as pd
from pydantic import JsonValue
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize

from algorithm_sdk import AlgorithmResult, ChartSpec, Metric, ResultTable, WarningItem
from algorithm_sdk.algorithm_utils import (
    encode_features,
    get_float_parameter,
    get_int_parameter,
    get_single_slot,
    get_slot,
    get_str_parameter,
    listwise_model_data,
    random_seed_from_context,
    table_from_frame,
)
from algorithm_sdk.errors import ErrorCode, ProtocolError


def run(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """训练 LightGBM 分类模型并返回评估、概率和特征重要性。"""

    classifier_type = _load_classifier_type()
    y_column = get_single_slot(slots, "y_column")
    x_columns = get_slot(slots, "x_columns")
    frame, dropped = listwise_model_data(data, y_column, x_columns)
    features, encoding = encode_features(frame, x_columns, context, drop_first=False)
    label_encoder = LabelEncoder()
    target = label_encoder.fit_transform(frame[y_column].astype(str))
    classes = [str(value) for value in label_encoder.classes_.tolist()]
    class_count = len(classes)
    if class_count < 2:
        raise ProtocolError(ErrorCode.DATA_ERROR, "LightGBM 分类因变量至少需要两个类别")

    objective = get_str_parameter(parameters, "objective", "auto")
    expected_objective = "binary" if class_count == 2 else "multiclass"
    if objective != "auto" and objective != expected_objective:
        raise ProtocolError(
            ErrorCode.PARAMETER_ERROR,
            f"当前因变量有 {class_count} 个类别，objective 应设置为 {expected_objective}",
        )
    objective = expected_objective

    train_indices, test_indices = _stratified_indices(target, parameters, context)
    x_values = features.to_numpy(dtype=float)
    model_parameters: dict[str, object] = {
        "objective": objective,
        "n_estimators": get_int_parameter(parameters, "n_estimators", 200),
        "learning_rate": get_float_parameter(parameters, "learning_rate", 0.05),
        "num_leaves": get_int_parameter(parameters, "num_leaves", 31),
        "max_depth": get_int_parameter(parameters, "max_depth", -1),
        "min_child_samples": get_int_parameter(parameters, "min_child_samples", 20),
        "subsample": get_float_parameter(parameters, "subsample", 1.0),
        "colsample_bytree": get_float_parameter(parameters, "colsample_bytree", 1.0),
        "reg_alpha": get_float_parameter(parameters, "reg_alpha", 0.0),
        "reg_lambda": get_float_parameter(parameters, "reg_lambda", 0.0),
        "random_state": random_seed_from_context(context),
        "n_jobs": get_int_parameter(parameters, "n_jobs", 1),
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
    }
    if objective == "multiclass":
        model_parameters["num_class"] = class_count
    model = classifier_type(**model_parameters)
    model.fit(x_values[train_indices], target[train_indices])

    actual = target[test_indices]
    predicted = np.asarray(model.predict(x_values[test_indices]), dtype=int)
    probabilities = np.asarray(model.predict_proba(x_values[test_indices]), dtype=float)
    if probabilities.shape != (len(test_indices), class_count):
        raise ProtocolError(ErrorCode.EXECUTION_ERROR, "LightGBM 返回的预测概率维度不正确")

    metrics = _classification_metrics(actual, predicted, probabilities, class_count)
    tables = _result_tables(
        model,
        features.columns.tolist(),
        classes,
        frame.index[test_indices],
        actual,
        predicted,
        probabilities,
    )
    charts = _result_charts(model, features.columns.tolist(), actual, probabilities, class_count)
    warnings: list[WarningItem] = []
    if dropped:
        warnings.append(
            WarningItem(
                code="MISSING_ROWS_DROPPED",
                message=f"建模前删除了 {dropped} 条包含缺失值的记录",
            )
        )
    return AlgorithmResult(
        metrics=metrics,
        tables=tables,
        charts=charts,
        warnings=warnings,
        metadata={
            "method": "lightgbm_classification.standard",
            "objective": objective,
            "classes": classes,
            "sample_count": len(frame.index),
            "training_sample_count": len(train_indices),
            "test_sample_count": len(test_indices),
            "dropped_rows": dropped,
            "feature_names": features.columns.tolist(),
            "categorical_encoding": encoding,
            "random_seed": random_seed_from_context(context),
            "model_parameters": model_parameters,
        },
    )


def _load_classifier_type() -> type[Any]:
    """延迟加载可选依赖，使缺少 LightGBM 时返回稳定错误码。"""

    try:
        from lightgbm import LGBMClassifier
    except (ImportError, OSError) as exc:
        raise ProtocolError(
            ErrorCode.DEPENDENCY_ERROR,
            "当前算法运行环境未安装或无法加载 LightGBM",
            details={"dependency": "lightgbm"},
        ) from exc
    return cast(type[Any], LGBMClassifier)


def _stratified_indices(
    target: np.ndarray,
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> tuple[np.ndarray, np.ndarray]:
    """生成可复现的分层训练集与测试集索引。"""

    class_counts = np.bincount(target)
    if bool((class_counts < 2).any()):
        raise ProtocolError(ErrorCode.DATA_ERROR, "每个目标类别至少需要两个样本")
    train_ratio = get_float_parameter(parameters, "train_ratio", 0.8)
    sample_count = len(target)
    class_count = len(class_counts)
    train_count = int(np.floor(sample_count * train_ratio))
    test_count = sample_count - train_count
    if train_count < class_count or test_count < class_count:
        raise ProtocolError(
            ErrorCode.DATA_ERROR,
            "训练集和测试集都必须至少包含每个目标类别一个样本，请调整训练占比",
        )
    indices = np.arange(sample_count)
    train_indices, test_indices = train_test_split(
        indices,
        train_size=train_ratio,
        stratify=target,
        shuffle=True,
        random_state=random_seed_from_context(context),
    )
    return np.asarray(train_indices, dtype=int), np.asarray(test_indices, dtype=int)


def _classification_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
    class_count: int,
) -> list[Metric]:
    """构造二分类和多分类共同及各自的核心指标。"""

    metrics = [
        Metric(id="accuracy", display_name="准确率", value=float(accuracy_score(actual, predicted)))
    ]
    for average, display_name in (
        ("macro", "宏平均"),
        ("micro", "微平均"),
        ("weighted", "加权平均"),
    ):
        metrics.extend(
            [
                Metric(
                    id=f"precision_{average}",
                    display_name=f"{display_name}精确率",
                    value=float(
                        precision_score(actual, predicted, average=average, zero_division=0)
                    ),
                ),
                Metric(
                    id=f"recall_{average}",
                    display_name=f"{display_name}召回率",
                    value=float(recall_score(actual, predicted, average=average, zero_division=0)),
                ),
                Metric(
                    id=f"f1_{average}",
                    display_name=f"{display_name} F1",
                    value=float(f1_score(actual, predicted, average=average, zero_division=0)),
                ),
            ]
        )
    if class_count == 2:
        positive_probability = probabilities[:, 1]
        metrics.extend(
            [
                Metric(
                    id="roc_auc",
                    display_name="ROC-AUC",
                    value=float(roc_auc_score(actual, positive_probability)),
                ),
                Metric(
                    id="pr_auc",
                    display_name="PR-AUC",
                    value=float(average_precision_score(actual, positive_probability)),
                ),
            ]
        )
    else:
        one_hot = label_binarize(actual, classes=np.arange(class_count))
        metrics.extend(
            [
                Metric(
                    id="roc_auc_ovr_macro",
                    display_name="宏平均 OvR ROC-AUC",
                    value=float(
                        roc_auc_score(actual, probabilities, multi_class="ovr", average="macro")
                    ),
                ),
                Metric(
                    id="roc_auc_ovr_weighted",
                    display_name="加权 OvR ROC-AUC",
                    value=float(
                        roc_auc_score(
                            actual,
                            probabilities,
                            multi_class="ovr",
                            average="weighted",
                        )
                    ),
                ),
                Metric(
                    id="roc_auc_ovr_micro",
                    display_name="微平均 OvR ROC-AUC",
                    value=float(roc_auc_score(one_hot.ravel(), probabilities.ravel())),
                ),
            ]
        )
    return metrics


def _result_tables(
    model: Any,
    feature_names: list[str],
    classes: list[str],
    test_index: pd.Index,
    actual: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
) -> list[ResultTable]:
    """生成混淆矩阵、分类报告、概率和特征重要性结果表。"""

    class_numbers = np.arange(len(classes))
    confusion = confusion_matrix(actual, predicted, labels=class_numbers)
    confusion_frame = pd.DataFrame(confusion, columns=classes)
    confusion_frame.insert(0, "actual_class", classes)
    report = classification_report(
        actual,
        predicted,
        labels=class_numbers,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    report_frame = pd.DataFrame(report).T.reset_index(names="class")
    prediction_frame = pd.DataFrame(
        {
            "row_index": [str(value) for value in test_index],
            "actual": [classes[value] for value in actual],
            "predicted": [classes[value] for value in predicted],
        }
    )
    for class_index, class_name in enumerate(classes):
        prediction_frame[f"probability_{class_name}"] = probabilities[:, class_index]
    split_importance = np.asarray(model.feature_importances_, dtype=float)
    gain_importance = np.asarray(
        model.booster_.feature_importance(importance_type="gain"),
        dtype=float,
    )
    importance_frame = pd.DataFrame(
        {
            "feature": feature_names,
            "split_importance": split_importance,
            "gain_importance": gain_importance,
        }
    ).sort_values("gain_importance", ascending=False)
    return [
        table_from_frame("confusion_matrix", "混淆矩阵", confusion_frame),
        table_from_frame("classification_report", "分类指标", report_frame),
        table_from_frame("predictions", "测试集预测概率", prediction_frame),
        table_from_frame("feature_importance", "特征重要性", importance_frame),
    ]


def _result_charts(
    model: Any,
    feature_names: list[str],
    actual: np.ndarray,
    probabilities: np.ndarray,
    class_count: int,
) -> list[ChartSpec]:
    """生成特征重要性以及二分类 ROC、PR 曲线。"""

    charts = [
        ChartSpec(
            id="feature_importance",
            display_name="特征重要性",
            chart_type="bar",
            data={
                "feature": feature_names,
                "importance": np.asarray(model.feature_importances_, dtype=float),
            },
        )
    ]
    if class_count == 2:
        positive_probability = probabilities[:, 1]
        false_positive, true_positive, _ = roc_curve(actual, positive_probability)
        precision, recall, _ = precision_recall_curve(actual, positive_probability)
        charts.extend(
            [
                ChartSpec(
                    id="roc_curve",
                    display_name="ROC 曲线",
                    chart_type="line",
                    data={"fpr": false_positive, "tpr": true_positive},
                ),
                ChartSpec(
                    id="pr_curve",
                    display_name="PR 曲线",
                    chart_type="line",
                    data={"recall": recall, "precision": precision},
                ),
            ]
        )
    return charts
