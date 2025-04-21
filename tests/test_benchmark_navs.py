from numpy import result_type
import pytest
import pandas as pd
import json
from pathlib import Path
from pandas.testing import assert_series_equal
from data_loader import get_benchmark_gain_daily
from timeseries import TimeseriesFrame
from test_utils import load_pickle, assert_identical, series_struct_info


"""
✅Check	                Meaning
------------------------------------------------------------------------------------
Chronologically sorted	        Index is increasing in time.
No duplicate dates	        Each date appears only once.
No missing or invalid NAVs	All nav values are finite floats (not NaN or inf).
DatetimeIndex	                Index is pd.DatetimeIndex, not string/object.
Index has no NaT	        All dates successfully parsed.
Index is named date	        Matches downstream code assumptions.
"""
@pytest.mark.order(8)  # test_local_benchmark_data_is_clean()
def test_local_benchmark_data_is_clean():
    """Check uploaded benchmark NAV data for cleanliness."""
    nav_path = Path("tests/data/icici_nifty_50_index_fund.txt")
    with open(nav_path) as f:
        nav_json = json.load(f)
    df = pd.DataFrame(nav_json["data"])
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["nav"] = df["nav"].astype(float)
    df = df.set_index("date").sort_index()
    assert isinstance(df.index, pd.DatetimeIndex)
    assert not df.index.hasnans
    assert df.index.is_monotonic_increasing
    assert df.index.is_unique
    assert df["nav"].dtype == float
    assert df["nav"].notna().all()
    assert df["nav"].apply(pd.api.types.is_number).all()


@pytest.mark.order(9)  # test_get_benchmark_gain_daily()
def test_get_benchmark_gain_daily():
    """
    Test the loading of usefully indexed benchmark historical NAVs.
    """
    # Load Pickled test input data
    raw_df = load_pickle("tests/data/benchmark_data.pkl")
    series = raw_df["Close"]
    series.name = "value"
    mock_benchmark_data = TimeseriesFrame(series)
        
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
