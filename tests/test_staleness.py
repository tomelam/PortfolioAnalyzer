import pandas as pd
from datetime import datetime, timedelta
from utils import warn_if_stale
from unittest.mock import patch
import pytest

def test_fresh_data_does_not_warn():
    df = pd.DataFrame({
        "date": [datetime.now()],
        "value": [100]
    })
    # Should not raise or print anything fatal
    warn_if_stale(df, label="Fresh", quiet=False)

def test_stale_data_with_yes_input_does_not_abort():
    df = pd.DataFrame({
        "date": [datetime.now() - timedelta(days=3)],
        "value": [100]
    })
    with patch("builtins.input", return_value="y"):
        warn_if_stale(df, label="Stale", quiet=False)

def test_stale_data_with_no_input_aborts():
    df = pd.DataFrame({
        "date": [datetime.now() - timedelta(days=3)],
        "value": [100]
    })
    with patch("builtins.input", return_value="n"):
        with pytest.raises(SystemExit):
            warn_if_stale(df, label="Stale", quiet=False)

def test_quiet_mode_skips_prompt():
    df = pd.DataFrame({
        "date": [datetime.now() - timedelta(days=3)],
        "value": [100]
    })
    # Should not raise or prompt
    warn_if_stale(df, label="Stale", quiet=True)
