import pytest
from test_utils import load_pickle
from pandas.testing import assert_frame_equal
from data_loader import align_portfolio_civs

@pytest.mark.order(6)  # test_align_portfolio_civs(mocker)
def test_align_portfolio_civs(mocker):
    """Test alignment of CIV data using fixed parts of Pickled inputs."""

    # Load Pickled test input data
    portfolio_civs = load_pickle("tests/data/portfolio_civs.pkl")
    print(portfolio_civs)

    # Load Pickled expected result data
    expected_alignment = load_pickle("tests/data/aligned_civs.pkl")

    # Extract deterministic parts (for example, first 1000 rows)
    fixed_portfolio_civs = {k: v.iloc[:1000] for k, v in portfolio_civs.items()}
    fixed_expected_alignment = expected_alignment.iloc[:1000]

    aligned = align_portfolio_civs(fixed_portfolio_civs)

    # Compare fixed portions ignoring column order and data types if needed
    assert_frame_equal(aligned, fixed_expected_alignment, check_dtype=True)
