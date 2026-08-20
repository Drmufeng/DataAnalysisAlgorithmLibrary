"""核心统计算法与 SciPy、statsmodels、scikit-learn 或手工结果对照。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import JsonValue
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from statsmodels.api import OLS, add_constant

from algorithm_sdk.models import AlgorithmResult, ResultTable
from runner_contract.runner import run_operation
from tests.operation_cases import (
    OPERATION_CASES,
    OperationCase,
    make_algorithm_data,
    make_request,
)

PROJECT_ROOT = Path(__file__).parents[1]
PACKAGES_ROOT = PROJECT_ROOT / "packages"


def _run(operation_key: str, case: OperationCase | None = None) -> AlgorithmResult:
    """执行一个固定或临时数值对照用例。"""

    data = make_algorithm_data()
    actual_case = OPERATION_CASES[operation_key] if case is None else case
    return run_operation(
        PACKAGES_ROOT / actual_case.package_id,
        data,
        make_request(operation_key, actual_case, data),
    )


def _table(result: AlgorithmResult, table_id: str) -> ResultTable:
    """按稳定编号取得结果表。"""

    return next(table for table in result.tables if table.id == table_id)


def _metric(result: AlgorithmResult, metric_id: str) -> float:
    """按稳定编号取得数值指标。"""

    value = next(metric.value for metric in result.metrics if metric.id == metric_id)
    assert isinstance(value, (int, float)) and not isinstance(value, bool)
    return float(value)


def _number(value: JsonValue) -> float:
    """把结果表中已经断言为数值的 JSON 单元格转为浮点数。"""

    assert isinstance(value, (int, float)) and not isinstance(value, bool)
    return float(value)


def test_descriptive_statistics_match_pandas_and_scipy() -> None:
    """均值、样本标准差、偏度、方差和标准误应与主流库一致。"""

    data = make_algorithm_data()
    case = OperationCase(
        "descriptive_analysis",
        {"x_columns": ["x1"]},
        {
            "statistics": [
                "mean",
                "std",
                "variance",
                "skewness",
                "range",
                "iqr",
                "standard_error",
            ],
            "round_digits": 10,
        },
    )
    result = run_operation(
        PACKAGES_ROOT / case.package_id,
        data,
        make_request("descriptive_statistics.basic", case, data),
    )
    row = _table(result, "descriptive_statistics").rows[0]
    values = data["x1"]
    assert row["mean"] == pytest.approx(float(values.mean()), abs=1e-9)
    assert row["std"] == pytest.approx(float(values.std(ddof=1)), abs=1e-9)
    assert row["variance"] == pytest.approx(float(values.var(ddof=1)), abs=1e-9)
    assert row["skewness"] == pytest.approx(float(stats.skew(values, bias=False)), abs=1e-9)
    assert row["range"] == pytest.approx(float(values.max() - values.min()), abs=1e-9)
    assert row["iqr"] == pytest.approx(
        float(values.quantile(0.75) - values.quantile(0.25)), abs=1e-9
    )
    assert row["standard_error"] == pytest.approx(
        float(values.std(ddof=1) / np.sqrt(len(values))), abs=1e-9
    )


def test_pearson_matches_scipy() -> None:
    """Pearson 系数和 P 值应与 scipy.stats.pearsonr 一致。"""

    data = make_algorithm_data()
    case = OperationCase(
        "statistical_association",
        {"x_columns": ["x1", "x2"]},
        {},
    )
    result = run_operation(
        PACKAGES_ROOT / case.package_id,
        data,
        make_request("correlation_analysis.pearson", case, data),
    )
    row = next(
        item
        for item in _table(result, "pearson_correlation").rows
        if item["variable_1"] == "x1" and item["variable_2"] == "x2"
    )
    reference = stats.pearsonr(data["x1"], data["x2"])
    assert row["coefficient"] == pytest.approx(float(reference.statistic), abs=1e-12)
    assert row["p_value"] == pytest.approx(float(reference.pvalue), abs=1e-12)


def test_ols_coefficients_match_statsmodels() -> None:
    """OLS 截距、系数和 R² 应与 statsmodels 直接拟合一致。"""

    data = make_algorithm_data()
    case = OperationCase(
        "statistical_models",
        {"y_column": ["y_cont"], "x_columns": ["x1", "x2"]},
        {"include_intercept": True, "standardize_x": False},
    )
    result = run_operation(
        PACKAGES_ROOT / case.package_id,
        data,
        make_request("linear_regression.ols", case, data),
    )
    reference = OLS(data["y_cont"], add_constant(data[["x1", "x2"]])).fit()
    rows = {str(row["feature"]): row for row in _table(result, "coefficients").rows}
    for feature in ("const", "x1", "x2"):
        assert rows[feature]["coefficient"] == pytest.approx(
            float(reference.params[feature]), abs=1e-10
        )
        assert rows[feature]["p_value"] == pytest.approx(
            float(reference.pvalues[feature]), abs=1e-10
        )
    assert _metric(result, "r_squared") == pytest.approx(float(reference.rsquared), abs=1e-12)


def test_pca_explained_variance_matches_sklearn() -> None:
    """标准化 PCA 的特征值和解释方差率应与 scikit-learn 一致。"""

    data = make_algorithm_data()
    case = OperationCase(
        "statistical_models",
        {"x_columns": ["x1", "x2", "x3", "x4"], "row_index": []},
        {"n_components": 2, "standardize": True, "rotation": "none"},
    )
    result = run_operation(
        PACKAGES_ROOT / case.package_id,
        data,
        make_request("principal_component_analysis.fixed_components", case, data),
    )
    transformed = StandardScaler().fit_transform(data[["x1", "x2", "x3", "x4"]])
    reference = PCA(n_components=2).fit(transformed)
    rows = _table(result, "explained_variance").rows
    assert [_number(row["eigenvalue"]) for row in rows] == pytest.approx(
        reference.explained_variance_.tolist(), abs=1e-12
    )
    assert [_number(row["explained_variance_ratio"]) for row in rows] == pytest.approx(
        reference.explained_variance_ratio_.tolist(), abs=1e-12
    )


def test_cohen_kappa_matches_sklearn() -> None:
    """普通 Cohen Kappa 应与 sklearn.metrics.cohen_kappa_score 一致。"""

    data = make_algorithm_data()
    case = OperationCase(
        "statistical_association",
        {"rater_columns": ["rater_1", "rater_2"], "frequency_weight": []},
        {},
    )
    result = run_operation(
        PACKAGES_ROOT / case.package_id,
        data,
        make_request("kappa_agreement.cohen", case, data),
    )
    reference = cohen_kappa_score(data["rater_1"], data["rater_2"])
    assert _metric(result, "kappa") == pytest.approx(float(reference), abs=1e-12)


def test_poisson_p_value_matches_chi_square_survival_function() -> None:
    """泊松检验报告的 P 值应由最终统计量和自由度计算。"""

    result = _run("poisson_goodness_of_fit.chi_square")
    chi_square = _metric(result, "chi_square")
    degrees = result.metadata["degrees_of_freedom"]
    assert isinstance(degrees, int)
    assert _metric(result, "p_value") == pytest.approx(
        float(stats.chi2.sf(chi_square, degrees)), abs=1e-12
    )


def test_periodic_downsampling_matches_manual_slice() -> None:
    """直接降采样的行选择应等于稳定排序后的 Python 切片。"""

    data = make_algorithm_data()
    result = _run("data_downsampling.periodic_sample")
    assert result.data is not None
    expected = data.sort_values("time", kind="stable").iloc[1::3]
    pd.testing.assert_frame_equal(result.data, expected)


def test_sliding_window_first_row_matches_manual_construction() -> None:
    """滑窗首行的历史特征和预测目标应符合窗口、步阶和预测期定义。"""

    data = make_algorithm_data()
    result = _run("time_series_window.sliding")
    assert result.data is not None
    first = result.data.iloc[0]
    for index in range(6):
        assert first[f"signal_lag_{index + 1}"] == pytest.approx(data.loc[index, "signal"])
    assert first["signal_target"] == pytest.approx(data.loc[6, "signal"])


def test_manual_encoding_matches_declared_mapping() -> None:
    """手工编码只按协议 mapping 修改目标字段。"""

    data = make_algorithm_data()
    result = _run("data_encoding.manual")
    assert result.data is not None
    output_column = str(result.metadata["output_column"])
    expected = data["group"].map({"甲": 1, "乙": 2, "丙": 3}).astype(object)
    pd.testing.assert_series_equal(
        result.data[output_column].reset_index(drop=True),
        expected.reset_index(drop=True).rename(output_column),
    )
