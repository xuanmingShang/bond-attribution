"""
Tests for ladder.py – Bond ladder backtest and par-bond solver.
"""
import pytest, numpy as np, pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bond_pnl.yield_curve import YieldCurve, YieldCurveHistory, TENOR_YEARS
from bond_pnl.ladder import LadderBacktest


def _flat_curve(rate_pct, date="2024-01-02"):
    return YieldCurve(pd.Timestamp(date), np.full(len(TENOR_YEARS), rate_pct))


def _make_lb(rate_pct, date="2024-01-02"):
    """Build a minimal LadderBacktest with a one-date YieldCurveHistory."""
    lb = LadderBacktest.__new__(LadderBacktest)
    lb.rungs = [2, 5, 10]
    lb.freq = 2
    lb.face = 100.0
    dt = pd.Timestamp(date)
    yc = _flat_curve(rate_pct, date)
    lb.curve_history = type('CH', (), {'__getitem__': lambda s, d: yc})()
    return lb, dt


class TestParSolver:
    """The bisection par solver should produce bonds priced at 100."""

    def test_par_at_normal_rate(self):
        lb, dt = _make_lb(4.0)
        b = lb._make_bond(5, dt)
        px = b.dirty_price(dt, _flat_curve(4.0))
        assert abs(px - 100.0) < 0.02, f"Price = {px}"

    def test_par_at_zero_rate(self):
        lb, dt = _make_lb(0.0)
        b = lb._make_bond(5, dt)
        px = b.dirty_price(dt, _flat_curve(0.0))
        assert abs(px - 100.0) < 0.02, f"Price = {px}"

    def test_par_at_high_rate(self):
        lb, dt = _make_lb(12.0)
        b = lb._make_bond(5, dt)
        px = b.dirty_price(dt, _flat_curve(12.0))
        assert abs(px - 100.0) < 0.02, f"Price = {px}"

    def test_par_at_negative_rate(self):
        lb, dt = _make_lb(-0.5)
        b = lb._make_bond(5, dt)
        px = b.dirty_price(dt, _flat_curve(-0.5))
        assert abs(px - 100.0) < 0.02, f"Price = {px}"


class TestLadderIncome:
    """Income must equal ΔAI + coupon received (not linear accrual)."""

    def test_income_non_negative_no_maturity(self):
        """Between coupon dates, income ≈ ΔAI ≥ 0 for positive-coupon bonds."""
        lb, dt = _make_lb(4.0, "2024-03-01")
        b = lb._make_bond(5, dt)
        ai0 = b.accrued_interest(pd.Timestamp("2024-03-01"))
        ai1 = b.accrued_interest(pd.Timestamp("2024-03-02"))
        delta_ai = ai1 - ai0
        assert delta_ai >= 0
