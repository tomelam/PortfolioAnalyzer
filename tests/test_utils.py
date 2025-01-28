# tests/test_utils.py
import pandas as pd
import json
import pickle

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


def load_pickle(filepath):
    """
    Load data from a Pickle file.

    Args:
        filepath (str): Path to the Pickle file.
    Returns:
        a datum of any type or mixed and nested types
    """
    with open(filepath, "rb") as file:
        return pickle.load(file)


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
