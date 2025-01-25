# tests/test_utils.py
import pandas as pd
import json

mock_nav_data = pd.DataFrame(
    {"nav": [100.0, 101.0, 102.0]},
    index=pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
)
mock_nav_data.index.name = "date"

def load_json(filepath):
    """
    Load JSON data from a file.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        dict: Parsed JSON data.
    """
    with open(filepath, "r") as file:
        return json.load(file)

'''
def load_h5_to_dict(filepath):
    """
    Load data from an HDF5 file into a dictionary.

    Args:
        filepath (str): Path to the .h5 file.

    Returns:
        dict: Dictionary with DataFrames stored under their respective keys.
    """
    data = {}
    with pd.HDFStore(filepath, mode='r') as store:
        for key in store.keys():
            data[key.strip('/')] = store[key]
    return data
'''

def assert_identical(actual, expected):
    import pandas as pd
    import numpy as np
    import unittest
    from pandas.testing import assert_frame_equal
    from numpy.testing import assert_array_equal

    if isinstance(actual, pd.DataFrame) and isinstance(expected, pd.DataFrame):
        assert_frame_equal(actual, expected)
    elif isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray):
        assert_array_equal(actual, expected)
    else:
        unittest.TestCase().assertEqual(actual, expected)
