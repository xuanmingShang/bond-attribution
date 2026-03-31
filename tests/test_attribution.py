"""
Tests for attribution.py – Campisi-style PnL attribution.
"""
import pytest, numpy as np, pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bond_pnl.yield_curve import YieldCurve, TENOR_YEARS
from bond_pnl.bond import BondSpec
from bond_pnl.attribution import compute_daily_attribution


def _flat_curve(rate_pct, date="2024-01-02"):
    return YieldCurve(pd.Timestamp(date), np.full(len(TENOR_YEARS), rate_pct))


class TestAttributionIdentity:
    """Market + Time must equal Actual (the core identity)."""

    def test_identity_flat_curve(self):
        b = BondSpec(maturity="2029-01-01", coupon=0.04, freq=2,
                     issue_date="2024-01-01")
        c0 = _flat_curve(4.0, "2024-01-02")
        c1 = _flat_curve(4.05, "2024-01-03")
        a = compute_daily_attribution(b, c0, c1,
                pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"), 0.05)
        assert abs(a.actual_pnl - (a.market_impact + a.time_impact)) < 1e-10

    def test_identity_rate_change(self):
        b = BondSpec(maturity="2029-01-01", coupon=0.04, freq=2,
                     issue_date="2024-01-01")
        c0 = _flat_curve(4.0, "2024-06-01")
        c1 = _flat_curve(5.0, "2024-06-02")  # 100bp move
        a = compute_daily_attribution(b, c0, c1,
                pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-02"), 0.05)
        assert abs(a.actual_pnl - (a.market_impact + a.time_impact)) < 1e-10


class TestAttributionDecomposition:
    """Sub-component decomposition should be internally consistent."""

    def test_market_decomposition(self):
        """Duration + Convexity + Reshape + Rate Residual = Market Impact."""
        b = BondSpec(maturity="2029-01-01", coupon=0.04, freq=2,
                     issue_date="2024-01-01")
        c0 = _flat_curve(4.0, "2024-06-01")
        c1 = _flat_curve(4.50, "2024-06-02")
        a = compute_daily_attribution(b, c0, c1,
                pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-02"), 0.05)
        mkt_sum = a.duration_effect + a.convexity_effect + a.curve_reshape + a.rate_residual
        assert abs(mkt_sum - a.market_impact) < 1e-8

    def test_time_decomposition(self):
        """Carry + Time Residual = Time Impact."""
        b = BondSpec(maturity="2029-01-01", coupon=0.04, freq=2,
                     issue_date="2024-01-01")
        c0 = _flat_curve(4.0, "2024-06-01")
        c1 = _flat_curve(4.0, "2024-06-02")
        a = compute_daily_attribution(b, c0, c1,
                pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-02"), 0.05)
        assert abs((a.carry + a.time_residual) - a.time_impact) < 1e-8


class TestMaturityPeriod:
    """Bond maturing in the attribution interval must decompose cleanly."""

    def test_maturity_actual_pnl_includes_principal(self):
        """Actual PnL in the terminal period must include face redemption."""
        b = BondSpec(maturity="2024-06-02", coupon=0.04, freq=2,
                     issue_date="2024-01-01")
        c0 = _flat_curve(4.0, "2024-06-01")
        c1 = _flat_curve(4.0, "2024-06-02")
        a = compute_daily_attribution(b, c0, c1,
                pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-02"), 0.05)
        # Actual should be small (≈ carry for 1 day), not -100
        assert abs(a.actual_pnl) < 5.0, f"Actual = {a.actual_pnl}"

    def test_maturity_identity_holds(self):
        b = BondSpec(maturity="2024-06-02", coupon=0.04, freq=2,
                     issue_date="2024-01-01")
        c0 = _flat_curve(4.0, "2024-06-01")
        c1 = _flat_curve(4.0, "2024-06-02")
        a = compute_daily_attribution(b, c0, c1,
                pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-02"), 0.05)
        assert abs(a.actual_pnl - (a.market_impact + a.time_impact)) < 1e-10

    def test_maturity_time_residual_bounded(self):
        """Time residual may be non-trivial at maturity (pull-to-par effect)
        but the overall Market+Time=Actual identity still holds."""
        b = BondSpec(maturity="2024-06-02", coupon=0.04, freq=2,
                     issue_date="2024-01-01")
        c0 = _flat_curve(4.0, "2024-06-01")
        c1 = _flat_curve(4.0, "2024-06-02")
        a = compute_daily_attribution(b, c0, c1,
                pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-02"), 0.05)
        # At maturity, time residual can be larger because carry decomposition
        # involves a large pull-to-par rolldown, but keep it under $5 for a
        # $100 face bond.
        assert abs(a.time_residual) < 5.0, f"Time Residual = {a.time_residual}"
