"""逻辑回归和 Gaussian、Categorical、Mixed 朴素贝叶斯。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from packaging.version import Version
from pydantic import JsonValue
from scipy.special import logsumexp
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

from algorithm_sdk import AlgorithmResult, ChartSpec, Metric, ResultTable
from algorithm_sdk.algorithm_utils import (
    encode_features,
    field_analysis_type,
    get_bool_parameter,
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
from algorithm_sdk.serialization import to_json_compatible


@dataclass(slots=True)
class ClassificationData:
    """分类模型共享的特征、标签和编码快照。"""

    frame: pd.DataFrame
    features: pd.DataFrame
    target: np.ndarray
    classes: list[str]
    encoding: dict[str, JsonValue]
    dropped_rows: int


def run_logistic(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """训练用于预测评估的二项逻辑回归。"""

    prepared = _prepare_encoded_data(data, slots, context)
    if len(prepared.classes) != 2:
        raise ProtocolError(ErrorCode.DATA_ERROR, "二项逻辑回归的因变量必须恰好有两个类别")
    x_train, x_test, y_train, y_test = _split(prepared, parameters, context)
    standardize = get_bool_parameter(parameters, "standardize_numeric", True)
    scaler = StandardScaler() if standardize else None
    if scaler:
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
    regularization = get_str_parameter(parameters, "regularization", "l2")
    c_value = get_float_parameter(parameters, "c", 1.0)
    max_iter = get_int_parameter(parameters, "max_iter", 1000)
    seed = random_seed_from_context(context)
    if Version(sklearn_version) >= Version("1.8"):
        # scikit-learn 1.8 起以 l1_ratio 取代 penalty；SAGA 也更符合梯度优化语义。
        l1_ratio = 1.0 if regularization == "l1" else 0.0
        model = LogisticRegression(
            l1_ratio=l1_ratio,
            C=np.inf if regularization == "none" else c_value,
            max_iter=max_iter,
            solver="saga",
            random_state=seed,
        ).fit(x_train, y_train)
        solver = "saga"
    else:
        if regularization == "l1":
            penalty, solver = "l1", "liblinear"
        elif regularization == "none":
            penalty, solver = None, "lbfgs"
        else:
            penalty, solver = "l2", "lbfgs"
        model = LogisticRegression(
            penalty=penalty,
            C=c_value,
            max_iter=max_iter,
            solver=solver,
            random_state=seed,
        ).fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)
    predictions = model.predict(x_test)
    coefficient_frame = pd.DataFrame(
        {
            "feature": prepared.features.columns,
            "coefficient": model.coef_[0],
            "odds_ratio": np.exp(model.coef_[0]),
        }
    )
    cv_rows = _cross_validation_rows(
        model,
        prepared.features.to_numpy(dtype=float),
        prepared.target,
        parameters,
        context,
    )
    return _classification_result(
        "logistic_regression.binary_gradient_descent",
        prepared,
        y_test,
        predictions,
        probabilities,
        [
            table_from_frame("coefficients", "逻辑回归系数", coefficient_frame),
            table_from_frame("cross_validation", "交叉验证", pd.DataFrame(cv_rows)),
        ],
        metadata={
            "intercept": float(model.intercept_[0]),
            "standardize_numeric": standardize,
            "regularization": regularization,
            "solver": solver,
        },
    )


def run_gaussian_nb(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """仅对定量自变量使用 GaussianNB。"""

    x_columns = get_slot(slots, "x_columns")
    if any(field_analysis_type(context, column) != "quantitative" for column in x_columns):
        raise ProtocolError(ErrorCode.COLUMN_ERROR, "GaussianNB 只接受定量自变量")
    prepared = _prepare_encoded_data(data, slots, context)
    x_train, x_test, y_train, y_test = _split(prepared, parameters, context)
    model = GaussianNB(var_smoothing=get_float_parameter(parameters, "var_smoothing", 1e-9)).fit(
        x_train, y_train
    )
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)
    condition_rows = []
    for class_index, class_name in enumerate(prepared.classes):
        for feature_index, feature in enumerate(prepared.features.columns):
            condition_rows.append(
                {
                    "class": class_name,
                    "feature": feature,
                    "mean": float(model.theta_[class_index, feature_index]),
                    "variance": float(model.var_[class_index, feature_index]),
                }
            )
    return _classification_result(
        "naive_bayes_classification.gaussian",
        prepared,
        y_test,
        predictions,
        probabilities,
        [table_from_frame("condition_summary", "高斯条件分布", pd.DataFrame(condition_rows))],
        metadata={"class_prior": model.class_prior_},
    )


def run_categorical_nb(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """仅对定类自变量使用带拉普拉斯平滑的 CategoricalNB。"""

    y_column = get_single_slot(slots, "y_column")
    x_columns = get_slot(slots, "x_columns")
    if any(field_analysis_type(context, column) != "categorical" for column in x_columns):
        raise ProtocolError(ErrorCode.COLUMN_ERROR, "CategoricalNB 只接受定类自变量")
    frame, dropped = listwise_model_data(data, y_column, x_columns)
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoded = encoder.fit_transform(frame[x_columns].astype(str)).astype(int)
    if bool((encoded < 0).any()):
        raise ProtocolError(ErrorCode.DATA_ERROR, "定类字段编码出现未知类别")
    label_encoder = LabelEncoder()
    target = label_encoder.fit_transform(frame[y_column].astype(str))
    prepared = ClassificationData(
        frame=frame,
        features=pd.DataFrame(encoded, columns=x_columns, index=frame.index),
        target=target,
        classes=label_encoder.classes_.tolist(),
        encoding={
            column: {"categories": encoder.categories_[index].tolist()}
            for index, column in enumerate(x_columns)
        },
        dropped_rows=dropped,
    )
    x_train, x_test, y_train, y_test = _split(prepared, parameters, context)
    model = CategoricalNB(alpha=get_float_parameter(parameters, "alpha", 1.0)).fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)
    return _classification_result(
        "naive_bayes_classification.categorical",
        prepared,
        y_test,
        predictions,
        probabilities,
        [],
        metadata={"class_log_prior": model.class_log_prior_},
    )


def run_mixed_nb(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """分别估计数值高斯和类别条件概率，再组合对数概率。"""

    y_column = get_single_slot(slots, "y_column")
    x_columns = get_slot(slots, "x_columns")
    numeric_columns = [
        column for column in x_columns if field_analysis_type(context, column) == "quantitative"
    ]
    categorical_columns = [column for column in x_columns if column not in numeric_columns]
    if not numeric_columns or not categorical_columns:
        raise ProtocolError(ErrorCode.COLUMN_ERROR, "mixed 方法必须同时包含定量和定类自变量")
    frame, dropped = listwise_model_data(data, y_column, x_columns)
    label_encoder = LabelEncoder()
    target = label_encoder.fit_transform(frame[y_column].astype(str))
    indices = np.arange(len(frame.index))
    train_indices, test_indices = train_test_split(
        indices,
        train_size=get_float_parameter(parameters, "train_ratio", 0.8),
        stratify=target,
        random_state=random_seed_from_context(context),
    )
    classes = label_encoder.classes_.tolist()
    alpha = get_float_parameter(parameters, "alpha", 1.0)
    smoothing = get_float_parameter(parameters, "var_smoothing", 1e-9)
    log_probabilities = np.zeros((len(test_indices), len(classes)), dtype=float)
    condition_rows: list[dict[str, JsonValue]] = []
    for class_index, class_name in enumerate(classes):
        class_train = train_indices[target[train_indices] == class_index]
        if len(class_train) < 2:
            raise ProtocolError(ErrorCode.DATA_ERROR, f"类别 {class_name} 的训练样本不足 2 个")
        prior = (len(class_train) + alpha) / (len(train_indices) + alpha * len(classes))
        log_probabilities[:, class_index] = np.log(prior)
        for column in numeric_columns:
            train_values = pd.to_numeric(frame.iloc[class_train][column], errors="coerce").to_numpy(
                dtype=float
            )
            test_values = pd.to_numeric(frame.iloc[test_indices][column], errors="coerce").to_numpy(
                dtype=float
            )
            mean = float(train_values.mean())
            variance = float(train_values.var(ddof=1) + smoothing)
            log_probabilities[:, class_index] += -0.5 * (
                np.log(2 * np.pi * variance) + ((test_values - mean) ** 2) / variance
            )
            condition_rows.append(
                {
                    "class": class_name,
                    "feature": column,
                    "distribution": "gaussian",
                    "mean": mean,
                    "variance": variance,
                }
            )
        for column in categorical_columns:
            train_values = frame.iloc[class_train][column].astype(str)
            test_values = frame.iloc[test_indices][column].astype(str)
            categories = sorted(frame[column].astype(str).unique().tolist())
            counts = train_values.value_counts()
            denominator = len(train_values.index) + alpha * len(categories)
            probability_map = {
                category: (float(counts.get(category, 0)) + alpha) / denominator
                for category in categories
            }
            log_probabilities[:, class_index] += np.array(
                [np.log(probability_map[value]) for value in test_values]
            )
            condition_rows.append(
                {
                    "class": class_name,
                    "feature": column,
                    "distribution": "categorical",
                    "probabilities": to_json_compatible(probability_map),
                }
            )
    probabilities = np.exp(log_probabilities - logsumexp(log_probabilities, axis=1, keepdims=True))
    predictions = np.argmax(probabilities, axis=1)
    prepared = ClassificationData(
        frame=frame,
        features=frame[x_columns],
        target=target,
        classes=classes,
        encoding={
            column: {"categories": sorted(frame[column].astype(str).unique())}
            for column in categorical_columns
        },
        dropped_rows=dropped,
    )
    return _classification_result(
        "naive_bayes_classification.mixed",
        prepared,
        target[test_indices],
        predictions,
        probabilities,
        [table_from_frame("condition_summary", "混合条件分布", pd.DataFrame(condition_rows))],
        metadata={"numeric_columns": numeric_columns, "categorical_columns": categorical_columns},
    )


def _prepare_encoded_data(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    context: Mapping[str, JsonValue],
) -> ClassificationData:
    """整行删除缺失、编码 X 和固定 Y 类别顺序。"""

    y_column = get_single_slot(slots, "y_column")
    x_columns = get_slot(slots, "x_columns")
    frame, dropped = listwise_model_data(data, y_column, x_columns)
    features, encoding = encode_features(frame, x_columns, context, drop_first=False)
    label_encoder = LabelEncoder()
    target = label_encoder.fit_transform(frame[y_column].astype(str))
    if len(label_encoder.classes_) < 2:
        raise ProtocolError(ErrorCode.DATA_ERROR, "分类因变量至少需要两个类别")
    return ClassificationData(
        frame=frame,
        features=features,
        target=target,
        classes=label_encoder.classes_.tolist(),
        encoding=encoding,
        dropped_rows=dropped,
    )


def _split(
    prepared: ClassificationData,
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """执行带分层约束的训练测试划分。"""

    class_counts = np.bincount(prepared.target)
    if bool((class_counts < 2).any()):
        raise ProtocolError(ErrorCode.DATA_ERROR, "每个目标类别至少需要两个样本")
    values = prepared.features.to_numpy(dtype=float)
    train_ratio = get_float_parameter(parameters, "train_ratio", 0.8)
    train_count = int(np.floor(len(prepared.target) * train_ratio))
    test_count = len(prepared.target) - train_count
    class_count = len(prepared.classes)
    if train_count < class_count or test_count < class_count:
        raise ProtocolError(
            ErrorCode.DATA_ERROR,
            "训练集和测试集都必须至少包含每个目标类别一个样本，请调整训练占比",
        )
    if get_bool_parameter(parameters, "shuffle", True):
        x_train, x_test, y_train, y_test = train_test_split(
            values,
            prepared.target,
            train_size=train_ratio,
            stratify=prepared.target,
            shuffle=True,
            random_state=random_seed_from_context(context),
        )
        return (
            np.asarray(x_train),
            np.asarray(x_test),
            np.asarray(y_train),
            np.asarray(y_test),
        )

    # 关闭洗牌时保留原始顺序；若某个类别只落在一侧，拒绝给出失真的评估结果。
    x_train = values[:train_count]
    x_test = values[train_count:]
    y_train = prepared.target[:train_count]
    y_test = prepared.target[train_count:]
    expected_classes = set(range(class_count))
    if set(np.unique(y_train)) != expected_classes or set(np.unique(y_test)) != expected_classes:
        raise ProtocolError(
            ErrorCode.DATA_ERROR,
            "关闭数据洗牌后，训练集或测试集缺少目标类别；请调整数据顺序、训练占比或开启洗牌",
        )
    return x_train, x_test, y_train, y_test


def _cross_validation_rows(
    model: LogisticRegression,
    features: np.ndarray,
    target: np.ndarray,
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> list[dict[str, JsonValue]]:
    """按用户选择的折数返回准确率，不启用时返回空表。"""

    cv_folds = get_int_parameter(parameters, "cv_folds", 0)
    if cv_folds == 0:
        return []
    if int(np.bincount(target).min()) < cv_folds:
        raise ProtocolError(ErrorCode.DATA_ERROR, "每个类别的样本数必须不少于交叉验证折数")
    splitter = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_seed_from_context(context),
    )
    scores = cross_val_score(model, features, target, cv=splitter, scoring="accuracy")
    return [{"fold": index + 1, "accuracy": float(score)} for index, score in enumerate(scores)]


def _classification_result(
    method: str,
    prepared: ClassificationData,
    actual: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
    extra_tables: list[ResultTable],
    *,
    metadata: Mapping[str, object],
) -> AlgorithmResult:
    """构造二分类或多分类共享的指标、混淆矩阵和曲线。"""

    class_count = len(prepared.classes)
    average = "binary" if class_count == 2 else "weighted"
    confusion = confusion_matrix(actual, predicted, labels=np.arange(class_count))
    confusion_frame = pd.DataFrame(confusion, columns=prepared.classes)
    confusion_frame.insert(0, "actual_class", prepared.classes)
    report = classification_report(
        actual,
        predicted,
        labels=np.arange(class_count),
        target_names=prepared.classes,
        output_dict=True,
        zero_division=0,
    )
    report_frame = pd.DataFrame(report).T.reset_index(names="class")
    metrics = [
        Metric(
            id="accuracy", display_name="准确率", value=float(accuracy_score(actual, predicted))
        ),
        Metric(
            id="precision",
            display_name="精确率",
            value=float(precision_score(actual, predicted, average=average, zero_division=0)),
        ),
        Metric(
            id="recall",
            display_name="召回率",
            value=float(recall_score(actual, predicted, average=average, zero_division=0)),
        ),
        Metric(
            id="f1",
            display_name="F1",
            value=float(f1_score(actual, predicted, average=average, zero_division=0)),
        ),
    ]
    charts: list[ChartSpec] = []
    if class_count == 2:
        positive_probability = probabilities[:, 1]
        auc = float(roc_auc_score(actual, positive_probability))
        fpr, tpr, _ = roc_curve(actual, positive_probability)
        precision_values, recall_values, _ = precision_recall_curve(actual, positive_probability)
        metrics.append(Metric(id="roc_auc", display_name="ROC-AUC", value=auc))
        charts.extend(
            [
                ChartSpec(
                    id="roc_curve",
                    display_name="ROC 曲线",
                    chart_type="line",
                    data={"fpr": fpr, "tpr": tpr},
                ),
                ChartSpec(
                    id="pr_curve",
                    display_name="PR 曲线",
                    chart_type="line",
                    data={"recall": recall_values, "precision": precision_values},
                ),
            ]
        )
    else:
        auc = float(roc_auc_score(actual, probabilities, multi_class="ovr", average="weighted"))
        metrics.append(
            Metric(id="roc_auc_ovr_weighted", display_name="加权 OvR ROC-AUC", value=auc)
        )
    tables = [
        table_from_frame("confusion_matrix", "混淆矩阵", confusion_frame),
        table_from_frame("classification_report", "分类指标", report_frame),
        *extra_tables,
    ]
    return AlgorithmResult(
        metrics=metrics,
        tables=tables,
        charts=charts,
        metadata={
            "method": method,
            "classes": prepared.classes,
            "sample_count": len(prepared.frame.index),
            "test_sample_count": len(actual),
            "dropped_rows": prepared.dropped_rows,
            "encoding": prepared.encoding,
            **{key: to_json_compatible(value) for key, value in metadata.items()},
        },
    )
