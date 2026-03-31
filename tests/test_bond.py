"""
Tests for bond.py – pricing, risk measures, edge cases.
"""
import pytest, numpy as np, pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bond_pnl.yield_curve import YieldCurve, TENOR_YEARS
from bond_pnl.bond import BondSpec


def _flat_curve(rate_pct, date="2024-01-01"):
    """Create a flat yield curve at *rate_pct* %."""
    return YieldCurve(pd.Timestamp(date),
                      np.full(len(TENOR_YEARS), rate_pct))


# ── Pricing Invariants ──────────────────────────────────────────────

class TestDirtyPrice:
    def test_par_bond_near_100(self):
        """A bond priced at its own yield should be near par."""
        curve = _flat_curve(4.0)
        b = BondSpec(maturity="2029-01-01", coupon=0.04, freq=2,
                     issue_date="2024-01-01")
        px = b.dirty_price(pd.Timestamp("2024-01-01"), curve)
        assert abs(px - 100.0) < 1.0, f"Par bond price {px} too far from 100"

    def test_higher_coupon_higher_price(self):
        """Higher coupon → higher price, all else equal."""
        curve = _flat_curve(4.0)
        b4 = BondSpec(maturity="2029-01-01", coupon=0.04, issue_date="2024-01-01")
        b6 = BondSpec(maturity="2029-01-01", coupon=0.06, issue_date="2024-01-01")
        assert b6.dirty_price(pd.Timestamp("2024-01-01"), curve) > \
               b4.dirty_price(pd.Timestamp("2024-01-01"), curve)

    def test_higher_rate_lower_price(self):
        """Higher yield → lower price for fixed coupon."""
        b = BondSpec(maturity="2029-01-01", coupon=0.04, issue_date="2024-01-01")
        p4 = b.dirty_price(pd.Timestamp("2024-01-01"), _flat_curve(4.0))
        p6 = b.dirty_price(pd.Timestamp("2024-01-01"), _flat_curve(6.0))
        assert p6 < p4

    def test_matured_bond_returns_zero(self):
        """After maturity, dirty price should be 0."""
        b = BondSpec(maturity="2024-01-01", coupon=0.04, issue_date="2019-01-01")
        px = b.dirty_price(pd.Timestamp("2024-01-02"), _flat_curve(4.0))
        assert px == 0.0

    def test_positive_price(self):
        """Price must always be non-negative."""
        b = BondSpec(maturity="2029-01-01", coupon=0.04, issue_date="2024-01-01")
        for rate in [0.0, 2.0, 5.0, 10.0, 20.0]:
            px = b.dirty_price(pd.Timestamp("2024-01-01"), _flat_curve(rate))
            assert px >= 0.0, f"Negative price at rate {rate}%"


# ── Accrued Interest ────────────────────────────────────────────────

class TestAccruedInterest:
    def test_zero_at_coupon_date(self):
        """Accrued interest is zero on a coupon payment date (= maturity)."""
        # Maturity on a coupon date → AI should be ~0 right before
        b = BondSpec(maturity="2025-01-01", coupon=0.04, freq=2,
                     issue_date="2024-01-01")
        ai = b.accrued_interest(pd.Timestamp("2024-01-01"))
        assert abs(ai) < 0.01

    def test_after_maturity_returns_zero(self):
        b = BondSpec(maturity="2024-06-01", coupon=0.04, issue_date="2024-01-01")
        assert b.accrued_interest(pd.Timestamp("2024-06-02")) == 0.0


# ── Clean Price ─────────────────────────────────────────────────────

class TestCleanPrice:
    def test_after_maturity_returns_zero(self):
        b = BondSpec(maturity="2024-06-01", coupon=0.04, issue_date="2024-01-01")
        assert b.clean_price(pd.Timestamp("2024-06-02"), _flat_curve(4.0)) == 0.0

    def test_clean_less_than_dirty_mid_period(self):
        b = BondSpec(maturity="2029-01-01", coupon=0.04, freq=2,
                     issue_date="2024-01-01")
        settle = pd.Timestamp("2024-04-01")
        c = _flat_curve(4.0)
        assert b.clean_price(settle, c) < b.dirty_price(settle, c)


# ── DV01 ────────────────────────────────────────────────────────────

class TestDV01:
    def test_positive_dv01(self):
        """DV01 should be positive for a normal bond."""
        b = BondSpec(maturity="2029-01-01", coupon=0.04, issue_date="2024-01-01")
        dv = b.dv01(pd.Timestamp("2024-01-01"), _flat_curve(4.0))
        assert dv > 0.0

    def test_longer_maturity_higher_dv01(self):
        """Longer maturity → higher DV01."""
        c = _flat_curve(4.0)
        b5 = BondSpec(maturity="2029-01-01", coupon=0.04, issue_date="2024-01-01")
        b10 = BondSpec(maturity="2034-01-01", coupon=0.04, issue_date="2024-01-01")
        assert b10.dv01(pd.Timestamp("2024-01-01"), c) > \
               b5.dv01(pd.Timestamp("2024-01-01"), c)

    def test_dv01_magnitude(self):
        """5Y 4% bond DV01 should be ~0.04–0.05 per 100 face."""
        b = BondSpec(maturity="2029-01-01", coupon=0.04, issue_date="2024-01-01")
        dv = b.dv01(pd.Timestamp("2024-01-01"), _flat_curve(4.0))
        assert 0.03 < dv < 0.06, f"DV01 = {dv}"


# ── Convexity ───────────────────────────────────────────────────────

class TestConvexity:
    def test_positive_convexity(self):
        """Dollar convexity should be positive for a plain vanilla bond."""
        b = BondSpec(maturity="2029-01-01", coupon=0.04, issue_date="2024-01-01")
        cv = b.convexity_dollar(pd.Timestamp("2024-01-01"), _flat_curve(4.0))
        assert cv > 0.0


# ── Cashflow Schedule ───────────────────────────────────────────────

class TestCashflows:
    def test_cashflow_count(self):
        """5Y semi-annual should have 10 cashflows."""
        b = BondSpec(maturity="2029-01-01", coupon=0.04, freq=2,
                     issue_date="2024-01-01")
        cfs = b.cashflow_schedule(pd.Timestamp("2024-01-01"))
        assert len(cfs) == 10

    def test_final_cf_includes_principal(self):
        """Last cashflow includes coupon + face."""
        b = BondSpec(maturity="2029-01-01", coupon=0.04, freq=2,
                     issue_date="2024-01-01")
        cfs = b.cashflow_schedule(pd.Timestamp("2024-01-01"))
        last_cf = cfs[-1][1]
        expected = 100.0 * 0.04 / 2 + 100.0  # coupon + face
        assert abs(last_cf - expected) < 0.01
