from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd
import pytest

from utils import warn_if_stale


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
    """When the user declines to proceed on stale data, ``warn_if_stale``
    must raise ``ValueError`` so the CLI aborts.

    Time pinned to a Tuesday so the function's Sunday / Monday-before-9am
    skip doesn't bypass the staleness gate on flaky days.
    """
    from freezegun import freeze_time

    with freeze_time("2024-06-04 14:00:00"):  # Tuesday afternoon
        df = pd.DataFrame({
            "date": [datetime.now() - timedelta(days=3)],
            "value": [100]
        })
        with patch("builtins.input", return_value="n"):
            with pytest.raises(ValueError, match="stale data|too stale"):
                warn_if_stale(df, label="Stale", quiet=False)

def test_quiet_mode_skips_prompt():
    df = pd.DataFrame({
        "date": [datetime.now() - timedelta(days=3)],
        "value": [100]
    })
    # Should not raise or prompt
    warn_if_stale(df, label="Stale", quiet=True)
