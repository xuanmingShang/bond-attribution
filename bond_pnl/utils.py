"""
utils.py – Shared utility functions used by dashboard, main, and tests.

These are extracted here so they can be imported without triggering
Streamlit side-effects.
"""
from __future__ import annotations

import pandas as pd


def get_dates(df: pd.DataFrame):
    """Safely extract date axis from a DataFrame (column or index)."""
    if "Date" in df.columns:
        return df["Date"]
    return df.index


def snap_to_business_day(
    ydf: pd.DataFrame,
    date_str: str,
    direction: str = "forward",
) -> str:
    """Snap a date string to the nearest available business day in *ydf*.

    Parameters
    ----------
    ydf : DataFrame whose index contains business-day timestamps.
    date_str : ISO date string, e.g. "2024-01-06".
    direction : "forward" — pick next available date (inclusive).
                "backward" — pick previous available date (inclusive).

    Returns
    -------
    ISO date string of the snapped date.
    """
    ts = pd.Timestamp(date_str)
    if ts in ydf.index:
        return date_str
    if direction == "forward":
        later = ydf.index[ydf.index >= ts]
        if len(later) > 0:
            return str(later[0].date())
        return str(ydf.index[-1].date())
    else:  # backward
        earlier = ydf.index[ydf.index <= ts]
        if len(earlier) > 0:
            return str(earlier[-1].date())
        return str(ydf.index[0].date())


def input_fingerprint(
    start: str, end: str, maturity: int, coupon: float,
    face: float, freq: int, pca: bool, ladder: bool,
    ladder_capital: float = 0, ladder_rebal: int = 0,
) -> str:
    """Return a hashable fingerprint of the current sidebar inputs."""
    return (f"{start}|{end}|{maturity}|{coupon}|{face}|{freq}"
            f"|{pca}|{ladder}|{ladder_capital}|{ladder_rebal}")
