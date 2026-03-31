"""
yield_curve.py
==============
Load U.S. Treasury constant-maturity yield data from FRED,
store it locally as CSV cache, and provide interpolation
across tenors for any business date.
"""

import os
import pathlib
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from fredapi import Fred

# ── FRED series IDs for constant-maturity Treasury rates ──────────────
TENOR_MAP = {
    "1M":  ("DGS1MO",  1 / 12),
    "3M":  ("DGS3MO",  3 / 12),
    "6M":  ("DGS6MO",  6 / 12),
    "1Y":  ("DGS1",    1.0),
    "2Y":  ("DGS2",    2.0),
    "3Y":  ("DGS3",    3.0),
    "5Y":  ("DGS5",    5.0),
    "7Y":  ("DGS7",    7.0),
    "10Y": ("DGS10",  10.0),
    "20Y": ("DGS20",  20.0),
    "30Y": ("DGS30",  30.0),
}

TENOR_YEARS = np.array([v[1] for v in TENOR_MAP.values()])
TENOR_LABELS = list(TENOR_MAP.keys())

CACHE_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"


def _read_api_key(key_path: str | None = None) -> str:
    """Read the FRED API key from file or FRED_API_KEY env var."""
    env_key = os.environ.get("FRED_API_KEY")
    if env_key:
        return env_key.strip()
    if key_path is None:
        key_path = pathlib.Path(__file__).resolve().parent.parent / "fred-api-key.txt"
    p = pathlib.Path(key_path)
    if not p.exists():
        raise FileNotFoundError(
            f"FRED API key not found. Either set FRED_API_KEY env var "
            f"or create {p}")
    return p.read_text().strip()


def fetch_yields(start: str, end: str, api_key: str | None = None,
                 cache: bool = True) -> pd.DataFrame:
    """
    Download daily Treasury yields from FRED for all tenors.

    Parameters
    ----------
    start, end : str  (YYYY-MM-DD)
    api_key : str or None  – if None, read from fred-api-key.txt
    cache : bool – if True, save/load from local CSV

    Returns
    -------
    pd.DataFrame  index=date, columns=tenor labels, values=yield (%)
    """
    cache_file = CACHE_DIR / f"yields_{start}_{end}.csv"
    if cache and cache_file.exists():
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df

    if api_key is None:
        api_key = _read_api_key()
    fred = Fred(api_key=api_key)

    frames = {}
    for label, (series_id, _) in TENOR_MAP.items():
        s = fred.get_series(series_id, observation_start=start,
                            observation_end=end)
        frames[label] = s

    df = pd.DataFrame(frames)
    df.index.name = "date"

    # Drop rows where ALL tenors are NaN (holidays etc.)
    df = df.dropna(how="all")
    # Forward-fill minor gaps only (no bfill to avoid look-ahead bias)
    df = df.ffill()
    # Drop any remaining rows with NaN (only at start of series)
    df = df.dropna()

    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_file)

    return df


class YieldCurve:
    """
    Wrapper around a yield-curve snapshot for a single date.
    Supports interpolation to arbitrary tenors via cubic spline.
    """

    def __init__(self, date: pd.Timestamp, yields_pct: np.ndarray):
        """
        Parameters
        ----------
        date : pd.Timestamp
        yields_pct : array-like of shape (n_tenors,) in percent
        """
        self.date = date
        self.tenors = TENOR_YEARS.copy()
        self.yields_pct = np.asarray(yields_pct, dtype=float)
        self.yields = self.yields_pct / 100.0  # decimal
        self._spline = CubicSpline(self.tenors, self.yields,
                                   bc_type="natural")

    def rate(self, tenor_years: float) -> float:
        """Interpolated continuously-compounded yield (decimal)."""
        return float(self._spline(tenor_years))

    def rate_pct(self, tenor_years: float) -> float:
        """Interpolated yield in percent."""
        return self.rate(tenor_years) * 100.0

    def rates(self, tenors: np.ndarray) -> np.ndarray:
        """Vectorised interpolation (decimal)."""
        return self._spline(tenors)

    def discount(self, t: float) -> float:
        """Discount factor  P(0,t) = exp(-r(t)*t)."""
        return np.exp(-self.rate(t) * t)


class YieldCurveHistory:
    """
    Container for a time series of yield curves.
    Provides indexing by date and convenience accessors.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Parameters
        ----------
        df : pd.DataFrame  – output of ``fetch_yields``
        """
        self.df = df.copy()
        self.dates = df.index.to_list()
        self._cache: dict[pd.Timestamp, YieldCurve] = {}

    def __getitem__(self, date) -> YieldCurve:
        date = pd.Timestamp(date)
        if date not in self._cache:
            if date not in self.df.index:
                # Find nearest available date
                idx = self.df.index.get_indexer([date], method="ffill")[0]
                if idx < 0:
                    raise KeyError(f"No yield data on or before {date}")
                date = self.df.index[idx]
            row = self.df.loc[date].values
            self._cache[date] = YieldCurve(date, row)
        return self._cache[date]

    def business_dates(self, start=None, end=None):
        """Return list of available dates in [start, end]."""
        dates = self.df.index
        if start is not None:
            dates = dates[dates >= pd.Timestamp(start)]
        if end is not None:
            dates = dates[dates <= pd.Timestamp(end)]
        return dates.to_list()

    def changes(self) -> pd.DataFrame:
        """First differences of yields (in percent) across all tenors."""
        return self.df.diff().dropna()
