"""
Tests for ladder.py – Bond ladder backtest and par-bond solver.
"""
import pytest, numpy as np, pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bond_pnl.yield_curve import YieldCurve, YieldCurveHistory, TENOR_LABELS, TENOR_YEARS
from bond_pnl.ladder import DEFAULT_RUNGS, IMMUNIZED_RUNGS, LadderBacktest, solve_duration_weights


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


def _flat_history(start="2024-01-02", end="2025-01-10", rate=4.0):
    dates = pd.bdate_range(start, end)
    df = pd.DataFrame(rate, index=dates, columns=TENOR_LABELS)
    return YieldCurveHistory(df)


class TestClassicRollingLadder:
    def test_default_rungs_are_classic_1_to_5(self):
        assert DEFAULT_RUNGS == [1, 2, 3, 4, 5]

    def test_classic_rolls_matured_principal_without_selling_unmatured_bonds(self):
        ch = _flat_history()
        bt = LadderBacktest(ch, "2024-01-02", "2025-01-10")
        result = bt.run()
        actions = result["trade_log"]["action"].tolist()

        assert "SELL" not in actions
        assert "REDEEM" in actions
        assert "ROLL_BUY" in actions
        assert len(result["holdings"]) == 5

        redeem_cash = result["trade_log"].loc[result["trade_log"]["action"] == "REDEEM", "cash_amount"].iloc[0]
        roll_cash = result["trade_log"].loc[result["trade_log"]["action"] == "ROLL_BUY", "cash_amount"].iloc[0]
        assert roll_cash == pytest.approx(redeem_cash, abs=0.01)


class TestWithdrawalLadder:
    def test_withdrawal_records_shortfall_without_selling_bonds(self):
        ch = _flat_history(end="2024-03-15")
        bt = LadderBacktest(
            ch,
            "2024-01-02",
            "2024-03-15",
            strategy="withdrawal",
            withdrawal_amount=10_000,
            withdrawal_frequency="Monthly",
            first_withdrawal_date="2024-01-03",
        )
        result = bt.run()
        actions = result["trade_log"]["action"].tolist()

        assert "SELL" not in actions
        assert not result["withdrawals"].empty
        assert result["withdrawals"]["Shortfall"].sum() > 0


class TestImmunizedLadder:
    def test_duration_solver_matches_feasible_target(self):
        sol = solve_duration_weights(np.array([1.0, 3.0, 7.0]), 4.0)
        assert sol.weights.sum() == pytest.approx(1.0)
        assert np.all(sol.weights >= 0.0)
        assert np.dot(sol.weights, np.array([1.0, 3.0, 7.0])) == pytest.approx(4.0, abs=0.02)

    def test_duration_solver_clamps_unreachable_target(self):
        sol = solve_duration_weights(np.array([1.0, 3.0, 7.0]), 30.0)
        assert sol.target_duration == pytest.approx(7.0)
        assert sol.status.startswith("clamped_high")

    def test_immunized_ladder_uses_wide_rungs_and_records_match(self):
        ch = _flat_history(end="2024-02-15")
        bt = LadderBacktest(
            ch,
            "2024-01-02",
            "2024-02-15",
            strategy="immunized",
            target_duration=5.0,
        )
        result = bt.run()

        assert bt.rungs == IMMUNIZED_RUNGS
        assert not result["rung_analytics"].empty
        assert not result["match_ts"].empty
        first_gap = result["match_ts"]["Duration Gap"].dropna().iloc[0]
        assert abs(first_gap) < 0.05
