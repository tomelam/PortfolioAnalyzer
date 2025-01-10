# tests/test_utils.py
import pandas as pd

mock_nav_data = pd.DataFrame(
    {"nav": [100.0, 101.0, 102.0]},
    index=pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
)
mock_nav_data.index.name = "date"
