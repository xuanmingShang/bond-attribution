"""
test_integration.py – Tests for entry-point utilities, date snapping,
dashboard helpers, and annualization logic.

Imports real functions from bond_pnl.utils (shared with dashboard.py)
to test the actual implementation, not mirrored logic.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd
import numpy as np

from bond_pnl.yield_curve import YieldCurve, YieldCurveHistory, TENOR_YEARS, _read_api_key
from bond_pnl.utils import get_dates, snap_to_business_day, input_fingerprint
from main import should_customize_bond


# ── API key resolution ────────────────────────────────────────────────
class TestAPIKeyResolution:
    def test_env_var_takes_priority(self, monkeypatch):
        """FRED_API_KEY env var should be returned even if file exists."""
        monkeypatch.setenv("FRED_API_KEY", "test_key_123")
        assert _read_api_key() == "test_key_123"

    def test_missing_both_raises(self, monkeypatch, tmp_path):
        """No env var + no file should raise FileNotFoundError."""
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        fake_path = tmp_path / "nonexistent.txt"
        with pytest.raises(FileNotFoundError):
            _read_api_key(str(fake_path))


# ── Fixtures ──────────────────────────────────────────────────────────
@pytest.fixture
def sample_ydf():
    """Create a small synthetic yield DataFrame for testing."""
    dates = pd.bdate_range("2024-01-02", periods=20)  # business days only
    n_tenors = len(TENOR_YEARS)
    rng = np.random.RandomState(42)
    data = 4.0 + rng.randn(len(dates), n_tenors) * 0.1
    from bond_pnl.yield_curve import TENOR_LABELS
    return pd.DataFrame(data, index=dates, columns=TENOR_LABELS)


@pytest.fixture
def ch(sample_ydf):
    return YieldCurveHistory(sample_ydf)


# ── YieldCurveHistory date snapping ───────────────────────────────────
class TestYieldCurveHistorySnapping:
    def test_exact_date_lookup(self, ch, sample_ydf):
        d = sample_ydf.index[5]
        curve = ch[d]
        assert isinstance(curve, YieldCurve)

    def test_weekend_date_forward_fills(self, ch):
        """Looking up a weekend date should ffill to most recent business day."""
        curve = ch["2024-01-06"]
        assert isinstance(curve, YieldCurve)
        assert curve.date == pd.Timestamp("2024-01-05")

    def test_business_dates_returns_subset(self, ch):
        dates = ch.business_dates("2024-01-10", "2024-01-15")
        for d in dates:
            assert d.weekday() < 5


# ── Real get_dates from bond_pnl.utils ────────────────────────────────
class TestGetDates:
    def test_returns_column_when_present(self):
        df = pd.DataFrame({"Date": ["2024-01-02", "2024-01-03"], "Value": [1, 2]})
        result = get_dates(df)
        assert list(result) == ["2024-01-02", "2024-01-03"]

    def test_falls_back_to_index(self):
        df = pd.DataFrame({"Value": [1, 2]}, index=["2024-01-02", "2024-01-03"])
        result = get_dates(df)
        assert list(result) == ["2024-01-02", "2024-01-03"]


# ── Real snap_to_business_day from bond_pnl.utils ─────────────────────
class TestSnapToBusinessDay:
    def test_exact_hit_returned_unchanged(self, sample_ydf):
        exact = str(sample_ydf.index[3].date())
        assert snap_to_business_day(sample_ydf, exact, "forward") == exact
        assert snap_to_business_day(sample_ydf, exact, "backward") == exact

    def test_snap_forward_from_weekend(self, sample_ydf):
        result = snap_to_business_day(sample_ydf, "2024-01-06", "forward")
        assert result == "2024-01-08"  # Monday

    def test_snap_backward_from_weekend(self, sample_ydf):
        result = snap_to_business_day(sample_ydf, "2024-01-06", "backward")
        assert result == "2024-01-05"  # Friday

    def test_snap_forward_past_end_falls_back(self, sample_ydf):
        """If no later date exists, return last available."""
        far_future = "2099-12-31"
        result = snap_to_business_day(sample_ydf, far_future, "forward")
        assert result == str(sample_ydf.index[-1].date())

    def test_snap_backward_before_start_falls_forward(self, sample_ydf):
        """If no earlier date exists, return first available."""
        far_past = "1900-01-01"
        result = snap_to_business_day(sample_ydf, far_past, "backward")
        assert result == str(sample_ydf.index[0].date())


# ── Input fingerprint ─────────────────────────────────────────────────
class TestInputFingerprint:
    def test_same_inputs_same_fingerprint(self):
        fp1 = input_fingerprint("2024-01-01", "2024-12-31", 5, 0.0, 100.0, 2, True, True)
        fp2 = input_fingerprint("2024-01-01", "2024-12-31", 5, 0.0, 100.0, 2, True, True)
        assert fp1 == fp2

    def test_different_inputs_different_fingerprint(self):
        fp1 = input_fingerprint("2024-01-01", "2024-12-31", 5, 0.0, 100.0, 2, True, True)
        fp2 = input_fingerprint("2024-01-01", "2024-12-31", 10, 0.0, 100.0, 2, True, True)
        assert fp1 != fp2

    def test_ladder_params_affect_fingerprint(self):
        fp1 = input_fingerprint("2024-01-01", "2024-12-31", 5, 0.0, 100.0, 2, True, True,
                                ladder_capital=1_000_000, ladder_rebal=12)
        fp2 = input_fingerprint("2024-01-01", "2024-12-31", 5, 0.0, 100.0, 2, True, True,
                                ladder_capital=2_000_000, ladder_rebal=12)
        assert fp1 != fp2

    def test_ladder_rebal_affects_fingerprint(self):
        fp1 = input_fingerprint("2024-01-01", "2024-12-31", 5, 0.0, 100.0, 2, True, True,
                                ladder_capital=1_000_000, ladder_rebal=6)
        fp2 = input_fingerprint("2024-01-01", "2024-12-31", 5, 0.0, 100.0, 2, True, True,
                                ladder_capital=1_000_000, ladder_rebal=12)
        assert fp1 != fp2


# ── Annualization math ────────────────────────────────────────────────
class TestAnnualization:
    def test_elapsed_year_fraction(self):
        start = pd.Timestamp("2024-01-02")
        end = pd.Timestamp("2024-12-31")
        elapsed = (end - start).days / 365.25
        assert 0.9 < elapsed < 1.1

    def test_annualized_return(self):
        total_ret = 0.01
        elapsed_years = 0.5
        ann = (1 + total_ret) ** (1 / elapsed_years) - 1
        assert 0.019 < ann < 0.021

    def test_annualized_pnl_scaling(self):
        pnl = 5.0
        elapsed = 2.0
        assert pnl / elapsed == 2.5


# ── Main.py CLI bond customization logic ──────────────────────────────
class TestMainCLI:
    def test_maturity_years_without_coupon(self):
        assert should_customize_bond(None, 10) is True

    def test_default_maturity_no_customize(self):
        assert should_customize_bond(None, 5) is False

    def test_coupon_only_triggers_customize(self):
        assert should_customize_bond(3.5, 5) is True

    def test_both_custom(self):
        assert should_customize_bond(4.0, 10) is True


# ── Max Drawdown calculation ──────────────────────────────────────────
class TestMaxDrawdown:
    def test_peak_to_trough(self):
        cum_ret = pd.Series([0.0, 0.02, 0.05, 0.03, 0.01, 0.04])
        running_max = cum_ret.cummax()
        drawdown = cum_ret - running_max
        assert drawdown.min() == pytest.approx(-0.04)

    def test_no_drawdown(self):
        cum_ret = pd.Series([0.0, 0.01, 0.02, 0.03])
        running_max = cum_ret.cummax()
        drawdown = cum_ret - running_max
        assert drawdown.min() == 0.0
