import pytest
import pandas as pd
import toml
from test_utils import load_json, load_pickle, assert_identical
from data_loader import get_aligned_portfolio_civs


@pytest.mark.order(7)  # test_get_aligned_portfolio_civs
def test_get_aligned_portfolio_civs():
    """Test the aligning of Current Investment Values (CIVs) data."""

    # Load mock data from Pickle file for an argument
    mock_argument_data = toml.load("tests/data/port-x.toml")
    expected_data = load_pickle("tests/data/aligned_portfolio_civs.pkl")
    print(f"type(expected_data): {type(expected_data)}")
    print(f"expected_data: {expected_data}")

    # Call the actual function-under-test with the mocked argument
    result = get_aligned_portfolio_civs(mock_argument_data)
    print(f"type(result): {type(result)}")
    print(f"result: {result}")

    # Compare only the first 1000 rows/items
    assert_identical(result.iloc[:1000], expected_data.iloc[:1000])
