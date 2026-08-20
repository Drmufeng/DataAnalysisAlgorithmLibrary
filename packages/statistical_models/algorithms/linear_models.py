"""普通最小二乘、岭回归和分层回归。"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from pydantic import JsonValue
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from statsmodels.api import OLS, add_constant
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera

from algorithm_sdk import AlgorithmResult, ChartSpec, Metric
from algorithm_sdk.algorithm_utils import (
    check_sample_parameter_count,
    encode_features,
    get_bool_parameter,
    get_float_parameter,
    get_int_parameter,
    get_list_parameter,
    get_single_slot,
    get_slot,
    listwise_model_data,
    random_seed_from_context,
    table_from_frame,
)
from algorithm_sdk.errors import ErrorCode, ProtocolError


def run_ols(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """使用 statsmodels OLS 返回完整统计推断。"""

    y_column = get_single_slot(slots, "y_column")
    x_columns = get_slot(slots, "x_columns")
    frame, dropped = listwise_model_data(data, y_column, x_columns)
    features, encoding = encode_features(frame, x_columns, context)
    y = pd.to_numeric(frame[y_column], errors="coerce").astype(float)
    if not bool(np.isfinite(y.to_numpy()).all()) or y.nunique() <= 1:
        raise ProtocolError(ErrorCode.DATA_ERROR, "因变量必须包含至少两个有限数值")
    standardize = get_bool_parameter(parameters, "standardize_x", False)
    if standardize:
        features = pd.DataFrame(
            StandardScaler().fit_transform(features),
            index=features.index,
            columns=features.columns,
        )
    include_intercept = get_bool_parameter(parameters, "include_intercept", True)
    design = add_constant(features, has_constant="add") if include_intercept else features
    check_sample_parameter_count(len(frame.index), design.shape[1])
    if np.linalg.matrix_rank(design.to_numpy()) < design.shape[1]:
        raise ProtocolError(ErrorCode.DATA_ERROR, "自变量设计矩阵完全共线或奇异")
    model = OLS(y, design).fit()
    confidence = get_float_parameter(parameters, "confidence_level", 0.95)
    intervals = model.conf_int(alpha=1 - confidence)
    coefficient_table = pd.DataFrame(
        {
            "feature": design.columns,
            "coefficient": model.params.to_numpy(),
            "standard_error": model.bse.to_numpy(),
            "t_statistic": model.tvalues.to_numpy(),
            "p_value": model.pvalues.to_numpy(),
            "confidence_lower": intervals.iloc[:, 0].to_numpy(),
            "confidence_upper": intervals.iloc[:, 1].to_numpy(),
        }
    )
    predictions = model.predict(design)
    residuals = y - predictions
    jarque_stat, jarque_p, skewness, kurtosis = jarque_bera(residuals)
    breusch_stat, breusch_p, _, _ = het_breuschpagan(residuals, design)
    prediction_table = pd.DataFrame(
        {
            "row_index": [str(index) for index in frame.index],
            "actual": y.to_numpy(),
            "predicted": predictions.to_numpy(),
            "residual": residuals.to_numpy(),
        }
    )
    return AlgorithmResult(
        metrics=[
            Metric(id="r_squared", display_name="R²", value=float(model.rsquared)),
            Metric(
                id="adjusted_r_squared", display_name="调整 R²", value=float(model.rsquared_adj)
            ),
            Metric(id="f_statistic", display_name="F 统计量", value=float(model.fvalue)),
            Metric(id="f_p_value", display_name="F 检验 P 值", value=float(model.f_pvalue)),
        ],
        tables=[
            table_from_frame("coefficients", "回归系数", coefficient_table),
            table_from_frame("predictions", "预测值与残差", prediction_table.head(500)),
        ],
        charts=[
            ChartSpec(
                id="residual_plot",
                display_name="预测值与残差",
                chart_type="scatter",
                data={"predicted": predictions.to_numpy(), "residual": residuals.to_numpy()},
            )
        ],
        metadata={
            "method": "linear_regression.ols",
            "sample_count": len(frame.index),
            "dropped_rows": dropped,
            "feature_names": design.columns.tolist(),
            "categorical_encoding": encoding,
            "standardize_x": standardize,
            "diagnostics": {
                "durbin_watson": float(durbin_watson(residuals)),
                "jarque_bera": float(jarque_stat),
                "jarque_bera_p": float(jarque_p),
                "residual_skewness": float(skewness),
                "residual_kurtosis": float(kurtosis),
                "breusch_pagan": float(breusch_stat),
                "breusch_pagan_p": float(breusch_p),
            },
        },
    )


def run_ridge_fixed(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """使用固定 alpha 拟合岭回归。"""

    return _run_ridge(data, slots, parameters, context, False)


def run_ridge_cv(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """通过 K 折交叉验证选择岭回归 alpha。"""

    return _run_ridge(data, slots, parameters, context, True)


def _run_ridge(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
    cross_validated: bool,
) -> AlgorithmResult:
    """共享特征编码、标准化、交叉验证和结果整理。"""

    y_column = get_single_slot(slots, "y_column")
    x_columns = get_slot(slots, "x_columns")
    frame, dropped = listwise_model_data(data, y_column, x_columns)
    features, encoding = encode_features(frame, x_columns, context)
    y = pd.to_numeric(frame[y_column], errors="coerce").to_numpy(dtype=float)
    standardize = get_bool_parameter(parameters, "standardize_x", True)
    scaler = StandardScaler() if standardize else None
    x = scaler.fit_transform(features) if scaler else features.to_numpy(dtype=float)
    include_intercept = get_bool_parameter(parameters, "include_intercept", True)
    curve_rows: list[dict[str, JsonValue]] = []
    mean_mse_by_alpha: dict[float, float] = {}
    if cross_validated:
        alpha_values = get_list_parameter(parameters, "alpha_grid", [0.01, 0.1, 1.0, 10.0, 100.0])
        alphas = [
            float(value) for value in alpha_values if isinstance(value, (int, float)) and value >= 0
        ]
        cv_folds = get_int_parameter(parameters, "cv_folds", 5)
        if cv_folds > len(frame.index):
            raise ProtocolError(ErrorCode.PARAMETER_ERROR, "cv_folds 不能超过有效样本数")
        splitter = KFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=random_seed_from_context(context),
        )
        for alpha in alphas:
            candidate = Ridge(alpha=alpha, fit_intercept=include_intercept)
            scores = -cross_val_score(
                candidate, x, y, cv=splitter, scoring="neg_mean_squared_error"
            )
            curve_rows.append(
                {
                    "alpha": alpha,
                    "mean_mse": float(scores.mean()),
                    "std_mse": float(scores.std(ddof=1)),
                }
            )
            mean_mse_by_alpha[alpha] = float(scores.mean())
        if not curve_rows:
            raise ProtocolError(ErrorCode.PARAMETER_ERROR, "alpha_grid 没有有效非负数值")
        alpha = min(mean_mse_by_alpha, key=mean_mse_by_alpha.__getitem__)
    else:
        alpha = get_float_parameter(parameters, "alpha", 1.0)
    model = Ridge(alpha=alpha, fit_intercept=include_intercept).fit(x, y)
    predictions = model.predict(x)
    coefficient_table = pd.DataFrame({"feature": features.columns, "coefficient": model.coef_})
    return AlgorithmResult(
        metrics=[
            Metric(id="alpha", display_name="最终 Alpha", value=alpha),
            Metric(
                id="training_mse",
                display_name="训练均方误差",
                value=float(mean_squared_error(y, predictions)),
            ),
            Metric(
                id="training_r_squared",
                display_name="训练 R²",
                value=float(r2_score(y, predictions)),
            ),
        ],
        tables=[
            table_from_frame("ridge_coefficients", "岭回归系数", coefficient_table),
            table_from_frame("cv_curve", "交叉验证曲线", pd.DataFrame(curve_rows)),
        ],
        metadata={
            "method": "ridge_regression.cross_validated"
            if cross_validated
            else "ridge_regression.fixed_alpha",
            "intercept": float(model.intercept_),
            "sample_count": len(frame.index),
            "dropped_rows": dropped,
            "feature_names": features.columns.tolist(),
            "categorical_encoding": encoding,
            "standardize_x": standardize,
        },
    )


def run_hierarchical(
    data: pd.DataFrame,
    slots: Mapping[str, list[str]],
    parameters: Mapping[str, JsonValue],
    context: Mapping[str, JsonValue],
) -> AlgorithmResult:
    """按 block_1 至 block_5 顺序累计拟合 OLS 并比较 R² 变化。"""

    y_column = get_single_slot(slots, "y_column")
    blocks = [get_slot(slots, f"block_{index}") for index in range(1, 6)]
    blocks = [block for block in blocks if block]
    if len(blocks) < 2:
        raise ProtocolError(ErrorCode.COLUMN_ERROR, "分层回归至少需要两个非空层次")
    flattened = [column for block in blocks for column in block]
    if len(flattened) != len(set(flattened)):
        raise ProtocolError(ErrorCode.COLUMN_ERROR, "不同回归层次不能重复使用同一字段")
    frame, dropped = listwise_model_data(data, y_column, flattened)
    y = pd.to_numeric(frame[y_column], errors="coerce").astype(float)
    include_intercept = get_bool_parameter(parameters, "include_intercept", True)
    model_rows: list[dict[str, JsonValue]] = []
    coefficient_rows: list[dict[str, JsonValue]] = []
    previous_model = None
    cumulative: list[str] = []
    final_residuals = None
    encoding_all: dict[str, JsonValue] = {}
    for block_index, block in enumerate(blocks, start=1):
        cumulative.extend(block)
        features, encoding = encode_features(frame, cumulative, context)
        encoding_all.update(encoding)
        design = add_constant(features, has_constant="add") if include_intercept else features
        check_sample_parameter_count(len(frame.index), design.shape[1])
        if np.linalg.matrix_rank(design.to_numpy()) < design.shape[1]:
            raise ProtocolError(ErrorCode.DATA_ERROR, f"第 {block_index} 层设计矩阵共线或奇异")
        model = OLS(y, design).fit()
        if previous_model is None:
            delta_r2 = float(model.rsquared)
            f_change = None
            f_change_p = None
        else:
            delta_r2 = float(model.rsquared - previous_model.rsquared)
            added_parameters = int(model.df_model - previous_model.df_model)
            denominator_df = int(model.df_resid)
            f_change = (delta_r2 / added_parameters) / ((1 - model.rsquared) / denominator_df)
            f_change_p = float(stats.f.sf(f_change, added_parameters, denominator_df))
        model_rows.append(
            {
                "block": block_index,
                "added_fields": [str(column) for column in block],
                "r_squared": float(model.rsquared),
                "adjusted_r_squared": float(model.rsquared_adj),
                "delta_r_squared": delta_r2,
                "f_change": f_change,
                "f_change_p": f_change_p,
            }
        )
        for feature, coefficient, p_value in zip(
            design.columns, model.params, model.pvalues, strict=True
        ):
            coefficient_rows.append(
                {
                    "block": block_index,
                    "feature": feature,
                    "coefficient": float(coefficient),
                    "p_value": float(p_value),
                }
            )
        previous_model = model
        final_residuals = model.resid
    assert previous_model is not None and final_residuals is not None
    return AlgorithmResult(
        metrics=[
            Metric(
                id="final_r_squared", display_name="最终 R²", value=float(previous_model.rsquared)
            ),
            Metric(id="block_count", display_name="模型层次数", value=len(blocks), unit="层"),
        ],
        tables=[
            table_from_frame("hierarchical_models", "分层模型比较", pd.DataFrame(model_rows)),
            table_from_frame(
                "hierarchical_coefficients", "各层系数", pd.DataFrame(coefficient_rows)
            ),
        ],
        metadata={
            "method": "hierarchical_regression.blockwise_ols",
            "sample_count": len(frame.index),
            "dropped_rows": dropped,
            "categorical_encoding": encoding_all,
            "durbin_watson": float(durbin_watson(final_residuals)),
        },
    )
