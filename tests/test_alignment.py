import pytest
from test_utils import load_pickle
from pandas.testing import assert_frame_equal
from data_loader import align_portfolio_civs

@pytest.mark.order(6)  # test_align_portfolio_civs(mocker)
def test_align_portfolio_civs(mocker):
    """Test alignment of CIV data using fixed parts of Pickled inputs."""
    import pandas as pd

    # Load Pickled test input data
    portfolio_civs = load_pickle("tests/data/portfolio_civs.pkl")

    # Load Pickled expected result data
    expected_alignment = load_pickle("tests/data/aligned_civs.pkl")
    col_type = type(expected_alignment.columns)
    assert isinstance(expected_alignment.columns, pd.MultiIndex), (
        f"Expected MultiIndex columns in golden data, but got {col_type}"
    )

    # Extract deterministic parts (for example, first 1000 rows)
    fixed_portfolio_civs = {k: v.iloc[:1000] for k, v in portfolio_civs.items()}

    # Here we're not generating fixed_expected_alignment from the function you're testing.
    # We're still using the golden data from the pre-saved .pkl file.
    aligned = align_portfolio_civs(fixed_portfolio_civs)
    fixed_expected_alignment = expected_alignment.loc[aligned.index]

    # Compare fixed portions ignoring column order and data types if needed
    assert_frame_equal(aligned, fixed_expected_alignment, check_dtype=True)
