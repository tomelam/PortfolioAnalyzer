import pandas as pd
import numpy as np
from civ_to_returns import civ_to_returns

class TimeseriesCIV:
    """
    Represents a timeseries of NAVs or cumulative investment values (CIV).
    Provides conversion to returns.
    """

    def __init__(self, series: pd.Series):
        if not isinstance(series, pd.Series):
            raise TypeError(f"TimeseriesCIV expects a pandas Series, got {type(series)}")
        if series.name != "value":
            raise ValueError(f"Expected series name 'value', got {series.name}")
        if not np.issubdtype(series.dtype, np.number):
            raise ValueError("TimeseriesCIV requires numeric data.")
        self.series = series.sort_index()

    def to_returns(self, frequency: str = "monthly") -> pd.Series:
        """
        Convert the NAV/CIV series to returns.
        frequency: 'daily' or 'monthly'
        """
        return civ_to_returns(self.series, frequency=frequency)

    def max_drawdowns(self, threshold=0.05):
        """
        Find drawdowns larger than a threshold (default 5%).
        Returns a list of dicts with:
        - start, end, trough (as dates)
        - drawdown (float)
        - drawdown_days (int)
        - recovery_days (int or None if not recovered)
        """
        s = self.series.dropna()
        peak = s.iloc[0]
        peak_date = s.index[0]
        trough = s.iloc[0]
        trough_date = s.index[0]
        recovery_date = None
        in_drawdown = False
        drawdowns = []

        for date, value in s.items():
            if value > peak:
                if in_drawdown:
                    # Check if drawdown exceeded threshold
                    dd = (trough - peak) / peak
                    if dd < -threshold:
                        drawdowns.append({
                            "start": peak_date,
                            "end": recovery_date or date,
                            "trough": trough_date,
                            "drawdown": abs(dd),
                            "drawdown_days": (trough_date - peak_date).days,
                            "recovery_days": ((recovery_date or date) - trough_date).days
                        })
                peak = value
                peak_date = date
                trough = value
                trough_date = date
                recovery_date = None
                in_drawdown = False
            elif value < trough:
                trough = value
                trough_date = date
                in_drawdown = True
            elif in_drawdown and value >= peak:
                recovery_date = date
                in_drawdown = False

        # Handle final unrecovered drawdown
        if in_drawdown:
            dd = (trough - peak) / peak
            if dd < -threshold:
                drawdowns.append({
                    "start": peak_date,
                    "end": s.index[-1],
                    "trough": trough_date,
                    "drawdown": abs(dd),
                    "drawdown_days": (trough_date - peak_date).days,
                    "recovery_days": None
                })

        return drawdowns
