import pandas as pd
import numpy as np
import pickle
from pandas.testing import assert_frame_equal


def assert_identical(actual, expected, tolerance=1e-6):
    """Ensure two objects are identical within a tolerance."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        for key in actual.keys():
            assert key in expected, f"Missing key: {key} in expected"
            assert_identical(actual[key], expected[key], tolerance)
    elif isinstance(actual, (int, float, np.number)):
        assert abs(actual - expected) < tolerance, f"Mismatch: {actual} != {expected}"
    elif isinstance(actual, pd.Series) or isinstance(actual, pd.DataFrame):
        assert actual.equals(expected), f"Data mismatch in {actual.name if hasattr(actual, 'name') else 'DataFrame'}"
    else:
        assert actual == expected, f"Objects differ: {actual} != {expected}"

def load_pickle(filepath):
    """Load data from a Pickle file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)

def compare_with_golden(actual, expected, tolerance=1e-6):
    """Compare an actual result with the golden dataset within a tolerance."""
    
    if isinstance(actual, dict) and isinstance(expected, dict):
        for k in actual:
            assert k in expected, f"Key {k} missing in golden data"
            assert abs(actual[k] - expected[k]) < tolerance, f"Mismatch for {k}: {actual[k]} vs {expected[k]}"
    
    elif isinstance(actual, (float, int)) and isinstance(expected, (float, int)):
        assert abs(actual - expected) < tolerance, f"Mismatch: {actual} vs {expected}"
    
    elif isinstance(actual, pd.DataFrame) and isinstance(expected, pd.DataFrame):
        assert_frame_equal(actual, expected, atol=tolerance)
    
    elif isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray):
        assert_array_equal(actual, expected)
    
    elif hasattr(actual, "equals") and hasattr(expected, "equals"):  # Pandas Series or DataFrame
        assert actual.equals(expected), "Pandas object mismatch"
    
    else:
        unittest.TestCase().assertEqual(actual, expected)

def series_struct_info(series):
    print("Data type:", series.dtype)
    print("Index:", series.index)
    print("Size:", series.size)
    print("Shape:", series.shape)
