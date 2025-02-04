from numpy import result_type
import pytest
import pandas as pd
from pandas.testing import assert_series_equal
from data_loader import get_benchmark_gain_daily, rename_yahoo_data_columns
from test_utils import load_pickle, assert_identical, series_struct_info


@pytest.mark.order(8)  # test_get_benchmark_gain_daily()
def test_get_benchmark_gain_daily():
    """
    Test getting of usefully indexed benchmark historical NAV using
    Yahoo Finance.
    """
    # Load Pickled test input data
    mock_benchmark_data = load_pickle("tests/data/benchmark_data.pkl")
    mock_benchmark_data = rename_yahoo_data_columns(mock_benchmark_data)

    expected_result = load_pickle("tests/data/benchmark_returns.pkl")
    print(f"expected_result: {expected_result}")
    print(f"type(expected_result): {type(expected_result)}")

    result = get_benchmark_gain_daily(mock_benchmark_data)
    print(f"result.index: {result.index}")
    assert result.index.name == "date", "Index name mismatch: expected 'date'"
    print(f"result: {result}")
    print(f"type(result): {type(result)}")

    assert len(result) > 0, "No data returned from function-under-test!"
    assert isinstance(result.index[0], pd.Timestamp), "Index is not datetime64[ns]!"

    print("Info about expected_result:")
    series_struct_info(expected_result)
    print("Info about result:")
    series_struct_info(result)
    assert_series_equal(result, expected_result)
