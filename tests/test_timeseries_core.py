import pandas as pd
import pytest
from timeseries import TimeseriesFrame

# === Core Safety Tests ===

@pytest.mark.order(24)
def test_max_drawdown_empty():
    """Verify that .max_drawdown() raises on empty series."""
    ts = TimeseriesFrame(pd.Series([], name="value"))
    with pytest.raises(ValueError, match="Max drawdown requires at least one data point"):
        ts.max_drawdown()


@pytest.mark.order(25)
def test_compare_to_empty():
    """Verify that .compare_to() raises when one or both series are empty."""
    ts1 = TimeseriesFrame(pd.Series([], name="value"))
    ts2 = TimeseriesFrame(pd.Series([], name="value"))
    with pytest.raises(ValueError, match="Cannot compare empty series"):
        ts1.compare_to(ts2)


@pytest.mark.order(26)
def test_compare_to_missing_value_column():
    """Verify that .compare_to() raises if 'value' column is missing."""
    ts1 = TimeseriesFrame(pd.Series([1, 2, 3]))
    ts2 = TimeseriesFrame(pd.Series([4, 5, 6]))
    # Drop 'value' column
    ts1.columns = ["not_value"]
    ts2.columns = ["not_value"]
    with pytest.raises(KeyError, match="Missing 'value' column"):
        ts1.compare_to(ts2)
