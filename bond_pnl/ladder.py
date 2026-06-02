"""
ladder.py - Bond-ladder backtest.

Three teaching strategies are supported:

* classic: hold-to-maturity rolling ladder. Matured principal is rolled into
  a new longest-rung synthetic Treasury. Coupons remain in cash.
* withdrawal: coupon and principal cash flows are used to cover a simple
  periodic spending schedule. Holdings are not sold to cover shortfalls.
* immunized: a duration-targeted construction using long-only weights across
  a wider set of synthetic Treasury rungs.

Attribution uses start-of-period holdings and clean prices for rolldown and
rate-movement buckets, so accrual is captured only in Income.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .bond import BondSpec
from .yield_curve import YieldCurveHistory

DEFAULT_RUNGS = [1, 2, 3, 4, 5]
IMMUNIZED_RUNGS = [1, 2, 3, 5, 7, 10, 20, 30]
VALID_STRATEGIES = {"classic", "withdrawal", "immunized"}


@dataclass
class LadderHolding:
    bond: BondSpec
    quantity: float
    rung_target: int
    purchase_date: str
    purchase_price: float
    lot_id: str = ""
    purchase_cash: float = 0.0
    target_weight: float = 0.0


@dataclass
class RebalanceRecord:
    date: str
    action: str
    rung: int | str
    bond_maturity: str
    coupon: float
    par_amount: float
    price: float
    cash_amount: float = 0.0
    cash_after: float = 0.0
    lot_id: str = ""
    source_lot_id: str = ""
    note: str = ""


@dataclass
class WeightSolution:
    weights: np.ndarray
    target_duration: float
    requested_duration: float
    status: str


def _interpolated_weights(durations: np.ndarray, target: float) -> np.ndarray:
    order = np.argsort(durations)
    ds = durations[order]
    weights = np.zeros(len(durations), dtype=float)
    if target <= ds[0]:
        weights[order[0]] = 1.0
        return weights
    if target >= ds[-1]:
        weights[order[-1]] = 1.0
        return weights
    for j in range(len(ds) - 1):
        lo, hi = ds[j], ds[j + 1]
        if lo <= target <= hi:
            if abs(hi - lo) < 1e-12:
                weights[order[j]] = 1.0
            else:
                hi_w = (target - lo) / (hi - lo)
                weights[order[j]] = 1.0 - hi_w
                weights[order[j + 1]] = hi_w
            return weights
    weights[order[np.argmin(np.abs(ds - target))]] = 1.0
    return weights


def solve_duration_weights(
    durations: list[float] | np.ndarray,
    target_duration: float,
    duration_tolerance: float = 0.02,
) -> WeightSolution:
    """Solve long-only rung weights matching a target duration."""
    ds = np.asarray(durations, dtype=float)
    valid = np.isfinite(ds)
    if not valid.all():
        raise ValueError("Duration inputs must be finite")
    if len(ds) == 0:
        raise ValueError("At least one candidate rung is required")

    requested = float(target_duration)
    d_min, d_max = float(ds.min()), float(ds.max())
    target = float(np.clip(requested, d_min, d_max))
    status = "matched"
    if requested < d_min:
        status = "clamped_low"
    elif requested > d_max:
        status = "clamped_high"

    if len(ds) == 1:
        return WeightSolution(np.ones(1), target, requested, status)

    n = len(ds)
    equal = np.full(n, 1.0 / n)
    x0 = _interpolated_weights(ds, target)

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "eq", "fun": lambda w: float(np.dot(w, ds) - target)},
    ]
    bounds = [(0.0, 1.0)] * n

    res = minimize(
        lambda w: float(np.sum((w - equal) ** 2)),
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 500, "disp": False},
    )
    if res.success:
        weights = np.clip(res.x, 0.0, 1.0)
        weights = weights / weights.sum()
        if abs(np.dot(weights, ds) - target) <= duration_tolerance:
            return WeightSolution(weights, target, requested, status)

    weights = _interpolated_weights(ds, target)
    fallback_status = "fallback" if status == "matched" else f"{status}_fallback"
    return WeightSolution(weights, target, requested, fallback_status)


@dataclass
class LadderBacktest:
    curve_history: YieldCurveHistory
    start: str
    end: str
    rungs: list[int] = field(default_factory=lambda: DEFAULT_RUNGS.copy())
    capital: float = 1_000_000.0
    rebal_months: int = 12
    financing_rate: float = 0.05
    strategy: str = "classic"
    withdrawal_amount: float = 0.0
    withdrawal_frequency: str = "Annual"
    first_withdrawal_date: str | None = None
    target_mode: str = "target_duration"
    target_duration: float | None = None
    liability_date: str | None = None
    liability_amount: float | None = None
    duration_tolerance: float = 0.02

    def __post_init__(self):
        self.strategy = self.strategy.lower()
        if self.strategy not in VALID_STRATEGIES:
            raise ValueError(f"Unknown ladder strategy: {self.strategy}")
        if self.strategy == "immunized" and self.rungs == DEFAULT_RUNGS:
            self.rungs = IMMUNIZED_RUNGS.copy()

        self.holdings: list[LadderHolding] = []
        self.cash: float = float(self.capital)
        self.rebalance_log: list[RebalanceRecord] = []
        self.portfolio_ts: list[dict] = []
        self.attr_records: list[dict] = []
        self.cashflow_records: list[dict] = []
        self.withdrawal_records: list[dict] = []
        self.match_records: list[dict] = []
        self.rung_analytics_records: list[dict] = []
        self.target_log_records: list[dict] = []
        self._lot_seq = 0
        self._withdrawal_due_by_date: dict[pd.Timestamp, float] = {}
        self._cumulative_withdrawn = 0.0
        self._last_date: pd.Timestamp | None = None

    # -- helpers ------------------------------------------------------
    def _new_lot_id(self) -> str:
        self._lot_seq += 1
        return f"L{self._lot_seq:04d}"

    def _append_log(
        self,
        dt: pd.Timestamp,
        action: str,
        rung: int | str = "",
        bond_maturity: str = "",
        coupon: float = 0.0,
        par_amount: float = 0.0,
        price: float = 0.0,
        cash_amount: float = 0.0,
        lot_id: str = "",
        source_lot_id: str = "",
        note: str = "",
    ) -> None:
        rung_value = f"{rung}Y" if isinstance(rung, (int, np.integer)) else str(rung)
        self.rebalance_log.append(
            RebalanceRecord(
                dt.strftime("%Y-%m-%d"),
                action,
                rung_value,
                bond_maturity,
                round(coupon, 8),
                round(par_amount, 2),
                round(price, 4),
                round(cash_amount, 2),
                round(self.cash, 2),
                lot_id,
                source_lot_id,
                note,
            )
        )

    def _make_bond(self, rung: int, settle: pd.Timestamp) -> BondSpec:
        """Create a synthetic par bond for *rung* at *settle*."""
        c = self.curve_history[settle]
        mat = settle + pd.DateOffset(years=int(rung))
        guess = c.rate(float(rung))

        def _price_at_cpn(cpn):
            return BondSpec(
                maturity=mat.strftime("%Y-%m-%d"),
                face=100.0,
                coupon=cpn,
                freq=2,
                day_count="ACT/ACT",
                issue_date=settle.strftime("%Y-%m-%d"),
            ).dirty_price(settle, c)

        lo = min(guess - 0.05, -0.05)
        hi = max(abs(guess) * 3, 0.10) + 0.01
        while _price_at_cpn(hi) < 100.0:
            hi *= 2
        while _price_at_cpn(lo) >= 100.0:
            lo -= 0.05
        for _ in range(80):
            mid_c = (lo + hi) / 2.0
            if _price_at_cpn(mid_c) < 100.0:
                lo = mid_c
            else:
                hi = mid_c
        par_cpn = round((lo + hi) / 2.0, 6)
        result = BondSpec(
            maturity=mat.strftime("%Y-%m-%d"),
            face=100.0,
            coupon=par_cpn,
            freq=2,
            day_count="ACT/ACT",
            issue_date=settle.strftime("%Y-%m-%d"),
        )
        px = result.dirty_price(settle, c)
        if abs(px - 100.0) > 0.01:
            raise RuntimeError(
                f"Par bond solver failed for {rung}Y at {settle}: "
                f"price={px:.4f}, coupon={par_cpn:.6f}"
            )
        return result

    def _candidate_risks(self, settle: pd.Timestamp) -> list[dict]:
        curve = self.curve_history[settle]
        rows = []
        for rung in self.rungs:
            bond = self._make_bond(rung, settle)
            price = bond.dirty_price(settle, curve)
            duration = bond.modified_duration(settle, curve)
            dv01 = bond.dv01(settle, curve)
            convexity = bond.convexity_dollar(settle, curve)
            if not np.isfinite([price, duration, dv01, convexity]).all():
                continue
            rows.append(
                {
                    "rung": rung,
                    "bond": bond,
                    "price": price,
                    "duration": duration,
                    "dv01": dv01,
                    "convexity": convexity,
                }
            )
        if not rows:
            raise ValueError("No valid ladder rungs were available")
        return rows

    def _liability_target(self, settle: pd.Timestamp, investable_value: float) -> dict:
        if not self.liability_date or not self.liability_amount:
            raise ValueError("Liability mode requires liability_date and liability_amount")
        horizon = (pd.Timestamp(self.liability_date) - settle).days / 365.25
        if horizon <= 0:
            raise ValueError("Liability date must be after the ladder start date")
        if horizon > 30.0:
            raise ValueError("Liability horizon above 30Y is outside this teaching model")
        curve = self.curve_history[settle]
        liability_pv = float(self.liability_amount) * np.exp(-curve.rate(horizon) * horizon)
        liability_dv01 = liability_pv * horizon / 10_000.0
        target = liability_dv01 * 10_000.0 / max(investable_value, 1e-12)
        return {
            "target_duration": target,
            "liability_pv": liability_pv,
            "liability_dv01": liability_dv01,
            "funding_ratio": investable_value / liability_pv if liability_pv else np.nan,
        }

    def _resolve_target(self, settle: pd.Timestamp, investable_value: float) -> dict:
        if self.strategy != "immunized":
            return {
                "target_duration": np.nan,
                "liability_pv": np.nan,
                "liability_dv01": np.nan,
                "funding_ratio": np.nan,
            }
        if self.target_mode == "liability":
            return self._liability_target(settle, investable_value)
        target = self.target_duration if self.target_duration is not None else 5.0
        return {
            "target_duration": float(target),
            "liability_pv": np.nan,
            "liability_dv01": np.nan,
            "funding_ratio": np.nan,
        }

    def _allocation_weights(self, settle: pd.Timestamp, cash_amount: float) -> tuple[list[dict], np.ndarray, dict]:
        candidates = self._candidate_risks(settle)
        if self.strategy != "immunized":
            weights = np.full(len(candidates), 1.0 / len(candidates))
            target_info = {
                "requested_duration": np.nan,
                "target_duration": np.nan,
                "status": "equal_weight",
                "liability_pv": np.nan,
                "liability_dv01": np.nan,
                "funding_ratio": np.nan,
            }
            return candidates, weights, target_info

        target_info = self._resolve_target(settle, cash_amount)
        durations = np.array([r["duration"] for r in candidates], dtype=float)
        solution = solve_duration_weights(
            durations,
            target_info["target_duration"],
            self.duration_tolerance,
        )
        target_info.update(
            {
                "requested_duration": solution.requested_duration,
                "target_duration": solution.target_duration,
                "status": solution.status,
                "min_duration": float(durations.min()),
                "max_duration": float(durations.max()),
            }
        )
        return candidates, solution.weights, target_info

    def _buy_weighted(
        self,
        settle: pd.Timestamp,
        cash_amount: float,
        action: str,
        source_lot_id: str = "",
    ) -> None:
        if cash_amount <= 1e-9:
            return
        candidates, weights, target_info = self._allocation_weights(settle, cash_amount)
        for candidate, weight in zip(candidates, weights):
            allocation = cash_amount * float(weight)
            if allocation <= 1e-9:
                continue
            price = candidate["price"]
            par_amount = allocation / price * 100.0
            lot_id = self._new_lot_id()
            self.holdings.append(
                LadderHolding(
                    candidate["bond"],
                    par_amount,
                    int(candidate["rung"]),
                    settle.strftime("%Y-%m-%d"),
                    price,
                    lot_id,
                    allocation,
                    float(weight),
                )
            )
            self.cash -= allocation
            self._append_log(
                settle,
                action,
                int(candidate["rung"]),
                candidate["bond"].maturity,
                candidate["bond"].coupon,
                par_amount,
                price,
                allocation,
                lot_id,
                source_lot_id,
                target_info.get("status", ""),
            )
            self.rung_analytics_records.append(
                {
                    "Date": settle.strftime("%Y-%m-%d"),
                    "Action": action,
                    "Rung": f"{candidate['rung']}Y",
                    "Weight": round(float(weight), 6),
                    "Allocation": round(allocation, 2),
                    "Par Amount": round(par_amount, 2),
                    "Price": round(price, 4),
                    "Coupon": round(candidate["bond"].coupon, 8),
                    "Duration": round(candidate["duration"], 6),
                    "DV01 per 100": round(candidate["dv01"], 8),
                    "Convexity per 100": round(candidate["convexity"], 8),
                    "Status": target_info.get("status", ""),
                }
            )

        if self.strategy == "immunized":
            self.target_log_records.append(
                {
                    "Date": settle.strftime("%Y-%m-%d"),
                    "Action": action,
                    "Requested Duration": target_info.get("requested_duration", np.nan),
                    "Target Duration": target_info.get("target_duration", np.nan),
                    "Min Feasible Duration": target_info.get("min_duration", np.nan),
                    "Max Feasible Duration": target_info.get("max_duration", np.nan),
                    "Status": target_info.get("status", ""),
                    "Liability PV": target_info.get("liability_pv", np.nan),
                    "Liability DV01": target_info.get("liability_dv01", np.nan),
                    "Funding Ratio": target_info.get("funding_ratio", np.nan),
                }
            )

    def _buy_single_rung(
        self,
        settle: pd.Timestamp,
        rung: int,
        cash_amount: float,
        action: str,
        source_lot_id: str = "",
    ) -> None:
        if cash_amount <= 1e-9:
            return
        curve = self.curve_history[settle]
        bond = self._make_bond(rung, settle)
        price = bond.dirty_price(settle, curve)
        par_amount = cash_amount / price * 100.0
        lot_id = self._new_lot_id()
        self.holdings.append(
            LadderHolding(
                bond,
                par_amount,
                rung,
                settle.strftime("%Y-%m-%d"),
                price,
                lot_id,
                cash_amount,
                1.0,
            )
        )
        self.cash -= cash_amount
        self._append_log(
            settle,
            action,
            rung,
            bond.maturity,
            bond.coupon,
            par_amount,
            price,
            cash_amount,
            lot_id,
            source_lot_id,
        )

    def _init(self, settle: pd.Timestamp):
        self._buy_weighted(settle, self.cash, "INIT_BUY")

    def _value_components(self, dt: pd.Timestamp) -> dict:
        curve = self.curve_history[dt]
        bond_value = 0.0
        par = 0.0
        for h in self.holdings:
            if dt < h.bond.maturity_dt:
                price = h.bond.dirty_price(dt, curve)
                bond_value += h.quantity * price / 100.0
                par += h.quantity
        return {
            "Cash Balance": float(self.cash),
            "Bond Market Value": float(bond_value),
            "Invested Value": float(bond_value),
            "Portfolio Value": float(self.cash + bond_value),
            "Total Par": float(par),
        }

    def _value(self, dt: pd.Timestamp) -> float:
        return self._value_components(dt)["Portfolio Value"]

    def _collect_coupons(self, dp: pd.Timestamp, dc: pd.Timestamp) -> float:
        total = 0.0
        for h in list(self.holdings):
            for cf_d, cf_a in h.bond.cashflow_schedule(dp):
                if dp < cf_d <= dc:
                    coupon_per_100 = cf_a - (h.bond.face if cf_d == h.bond.maturity_dt else 0.0)
                    if abs(coupon_per_100) <= 1e-12:
                        continue
                    amount = h.quantity * coupon_per_100 / 100.0
                    self.cash += amount
                    total += amount
                    self.cashflow_records.append(
                        {
                            "Date": dc.strftime("%Y-%m-%d"),
                            "Bond Maturity": h.bond.maturity,
                            "Rung": f"{h.rung_target}Y",
                            "Type": "Coupon",
                            "Amount": round(amount, 2),
                            "Par Amount": round(h.quantity, 2),
                            "Lot ID": h.lot_id,
                        }
                    )
                    self._append_log(
                        dc,
                        "COUPON",
                        h.rung_target,
                        h.bond.maturity,
                        h.bond.coupon,
                        h.quantity,
                        0.0,
                        amount,
                        h.lot_id,
                    )
        return total

    def _redeem_maturities(self, dp: pd.Timestamp, dc: pd.Timestamp) -> list[tuple[float, str]]:
        redemptions: list[tuple[float, str]] = []
        for h in list(self.holdings):
            if dp < h.bond.maturity_dt <= dc:
                amount = h.quantity
                self.cash += amount
                self.holdings.remove(h)
                redemptions.append((amount, h.lot_id))
                self.cashflow_records.append(
                    {
                        "Date": dc.strftime("%Y-%m-%d"),
                        "Bond Maturity": h.bond.maturity,
                        "Rung": f"{h.rung_target}Y",
                        "Type": "Principal",
                        "Amount": round(amount, 2),
                        "Par Amount": round(h.quantity, 2),
                        "Lot ID": h.lot_id,
                    }
                )
                self._append_log(
                    dc,
                    "REDEEM",
                    h.rung_target,
                    h.bond.maturity,
                    h.bond.coupon,
                    h.quantity,
                    100.0,
                    amount,
                    h.lot_id,
                )
        return redemptions

    def _build_withdrawals(self, dates: list[pd.Timestamp]) -> None:
        self._withdrawal_due_by_date = {}
        if self.strategy != "withdrawal" or self.withdrawal_amount <= 0:
            return
        months_by_freq = {
            "monthly": 1,
            "quarterly": 3,
            "semiannual": 6,
            "semi-annually": 6,
            "semiannualy": 6,
            "annual": 12,
            "annually": 12,
        }
        months = months_by_freq.get(str(self.withdrawal_frequency).lower(), 12)
        idx = pd.DatetimeIndex(dates)
        current = pd.Timestamp(self.first_withdrawal_date) if self.first_withdrawal_date else idx[0]
        while current < idx[0]:
            current += pd.DateOffset(months=months)
        while current <= idx[-1]:
            pos = idx.searchsorted(current)
            if pos < len(idx):
                due_dt = pd.Timestamp(idx[pos])
                self._withdrawal_due_by_date[due_dt] = (
                    self._withdrawal_due_by_date.get(due_dt, 0.0) + float(self.withdrawal_amount)
                )
            current += pd.DateOffset(months=months)

    def _pay_withdrawal(self, dt: pd.Timestamp, due: float) -> tuple[float, float]:
        if due <= 0:
            return 0.0, 0.0
        cash_before = self.cash
        paid = min(cash_before, due)
        shortfall = due - paid
        self.cash -= paid
        self._cumulative_withdrawn += paid
        self.withdrawal_records.append(
            {
                "Date": dt.strftime("%Y-%m-%d"),
                "Due": round(due, 2),
                "Paid": round(paid, 2),
                "Shortfall": round(shortfall, 2),
                "Cash Before": round(cash_before, 2),
                "Cash After": round(self.cash, 2),
            }
        )
        self._append_log(
            dt,
            "WITHDRAWAL",
            "",
            "",
            0.0,
            0.0,
            0.0,
            paid,
            "",
            "",
            f"shortfall={shortfall:.2f}",
        )
        return paid, shortfall

    def _portfolio_risk(self, dt: pd.Timestamp) -> dict:
        curve = self.curve_history[dt]
        bond_value = 0.0
        duration_value = 0.0
        dollar_dv01 = 0.0
        convexity = 0.0
        for h in self.holdings:
            if dt >= h.bond.maturity_dt:
                continue
            price = h.bond.dirty_price(dt, curve)
            mv = h.quantity * price / 100.0
            duration = h.bond.modified_duration(dt, curve)
            bond_value += mv
            duration_value += mv * duration
            dollar_dv01 += h.quantity * h.bond.dv01(dt, curve) / 100.0
            convexity += h.quantity * h.bond.convexity_dollar(dt, curve) / 100.0
        portfolio_duration = duration_value / bond_value if bond_value else 0.0
        return {
            "Portfolio Duration": portfolio_duration,
            "Portfolio DV01": dollar_dv01,
            "Portfolio Convexity": convexity,
            "Bond Market Value": bond_value,
        }

    def _record_match(self, dt: pd.Timestamp) -> None:
        risk = self._portfolio_risk(dt)
        comps = self._value_components(dt)
        if self.strategy == "immunized":
            target_info = self._resolve_target(dt, max(comps["Portfolio Value"], 1e-12))
            target_duration = target_info["target_duration"]
            target_dv01 = comps["Portfolio Value"] * target_duration / 10_000.0
            status = "tracked"
        else:
            target_info = {
                "liability_pv": np.nan,
                "liability_dv01": np.nan,
                "funding_ratio": np.nan,
            }
            target_duration = np.nan
            target_dv01 = np.nan
            status = self.strategy
        self.match_records.append(
            {
                "Date": dt.strftime("%Y-%m-%d"),
                "Portfolio Duration": round(risk["Portfolio Duration"], 6),
                "Target Duration": round(target_duration, 6) if np.isfinite(target_duration) else np.nan,
                "Duration Gap": (
                    round(risk["Portfolio Duration"] - target_duration, 6)
                    if np.isfinite(target_duration)
                    else np.nan
                ),
                "Portfolio DV01": round(risk["Portfolio DV01"], 6),
                "Target DV01": round(target_dv01, 6) if np.isfinite(target_dv01) else np.nan,
                "Liability PV": round(target_info["liability_pv"], 2)
                if np.isfinite(target_info["liability_pv"])
                else np.nan,
                "Liability DV01": round(target_info["liability_dv01"], 6)
                if np.isfinite(target_info["liability_dv01"])
                else np.nan,
                "Funding Ratio": round(target_info["funding_ratio"], 6)
                if np.isfinite(target_info["funding_ratio"])
                else np.nan,
                "Status": status,
            }
        )

    def _append_portfolio_row(
        self,
        dt: pd.Timestamp,
        prev_val: float,
        investment_pnl: float,
        coupon_cash: float,
        principal_cash: float,
        withdrawal_due: float,
        withdrawal_paid: float,
        withdrawal_shortfall: float,
    ) -> None:
        comps = self._value_components(dt)
        total_return_value = comps["Portfolio Value"] + self._cumulative_withdrawn
        daily_return = investment_pnl / prev_val if prev_val else 0.0
        cumulative_return = (total_return_value - self.capital) / self.capital
        self.portfolio_ts.append(
            {
                "Date": dt.strftime("%Y-%m-%d"),
                "Portfolio Value": round(comps["Portfolio Value"], 2),
                "Daily Return": round(daily_return, 6),
                "Cumulative Return": round(cumulative_return, 6),
                "Cash Balance": round(comps["Cash Balance"], 2),
                "Invested Value": round(comps["Invested Value"], 2),
                "Bond Market Value": round(comps["Bond Market Value"], 2),
                "Total Par": round(comps["Total Par"], 2),
                "Coupon Cashflow": round(coupon_cash, 2),
                "Principal Cashflow": round(principal_cash, 2),
                "Withdrawal Due": round(withdrawal_due, 2),
                "Withdrawal Paid": round(withdrawal_paid, 2),
                "Withdrawal Shortfall": round(withdrawal_shortfall, 2),
                "Cumulative Withdrawn": round(self._cumulative_withdrawn, 2),
            }
        )

    # -- main loop ----------------------------------------------------
    def run(self) -> dict:
        dates = [pd.Timestamp(d) for d in self.curve_history.business_dates(self.start, self.end)]
        if len(dates) < 2:
            raise ValueError("Need >=2 dates")

        self._build_withdrawals(dates)
        self._init(dates[0])
        self._last_date = dates[0]
        initial_value = self._value(dates[0])
        self._append_portfolio_row(dates[0], initial_value, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._record_match(dates[0])
        prev_val = initial_value

        for i in range(1, len(dates)):
            dp = dates[i - 1]
            dc = dates[i]

            sop_holdings = list(self.holdings)
            cp = self.curve_history[dp]
            cc = self.curve_history[dc]

            income = 0.0
            for h in sop_holdings:
                if h.bond.maturity_dt <= dp:
                    continue
                ai_prev = h.bond.accrued_interest(dp)
                ai_curr = h.bond.accrued_interest(dc)
                cpn_received = 0.0
                for cf_d, cf_a in h.bond.cashflow_schedule(dp):
                    if dp < cf_d <= dc:
                        cpn_received += cf_a - (h.bond.face if cf_d == h.bond.maturity_dt else 0.0)
                income += h.quantity * (ai_curr - ai_prev + cpn_received) / 100.0

            roll_pnl = 0.0
            rate_pnl = 0.0
            for h in sop_holdings:
                if h.bond.maturity_dt <= dp:
                    continue
                if h.bond.maturity_dt <= dc:
                    cp_clean = h.bond.clean_price(dp, cp)
                    roll_pnl += h.quantity * (h.bond.face - cp_clean) / 100.0
                else:
                    cp_clean = h.bond.clean_price(dp, cp)
                    cr_clean = h.bond.clean_price(dc, cp)
                    cc_clean = h.bond.clean_price(dc, cc)
                    roll_pnl += h.quantity * (cr_clean - cp_clean) / 100.0
                    rate_pnl += h.quantity * (cc_clean - cr_clean) / 100.0

            coupon_cash = self._collect_coupons(dp, dc)
            redemptions = self._redeem_maturities(dp, dc)
            principal_cash = sum(amount for amount, _ in redemptions)

            if self.strategy == "classic":
                for amount, source_lot_id in redemptions:
                    self._buy_single_rung(dc, max(self.rungs), amount, "ROLL_BUY", source_lot_id)
            elif self.strategy == "immunized" and principal_cash > 0:
                self._buy_weighted(dc, principal_cash, "ROLL_BUY", ",".join(src for _, src in redemptions))

            pre_external_value = self._value(dc)
            withdrawal_due = 0.0
            withdrawal_paid = 0.0
            withdrawal_shortfall = 0.0
            if self.strategy == "withdrawal":
                withdrawal_due = self._withdrawal_due_by_date.get(dc, 0.0)
                withdrawal_paid, withdrawal_shortfall = self._pay_withdrawal(dc, withdrawal_due)

            cv = self._value(dc)
            investment_pnl = pre_external_value - prev_val
            self._append_portfolio_row(
                dc,
                prev_val,
                investment_pnl,
                coupon_cash,
                principal_cash,
                withdrawal_due,
                withdrawal_paid,
                withdrawal_shortfall,
            )

            residual = investment_pnl - income - roll_pnl - rate_pnl
            self.attr_records.append(
                {
                    "Date": dc.strftime("%Y-%m-%d"),
                    "Total PnL": round(investment_pnl, 2),
                    "Income": round(income, 2),
                    "Rolldown": round(roll_pnl, 2),
                    "Rate Movement": round(rate_pnl, 2),
                    "Residual": round(residual, 2),
                }
            )
            self._record_match(dc)
            self._last_date = dc
            prev_val = cv

        trade_log = pd.DataFrame([r.__dict__ for r in self.rebalance_log])
        return {
            "portfolio_ts": pd.DataFrame(self.portfolio_ts).set_index("Date"),
            "attribution": pd.DataFrame(self.attr_records).set_index("Date"),
            "rebalance_log": trade_log,
            "trade_log": trade_log,
            "holdings": self._holdings_view(),
            "cashflows": pd.DataFrame(self.cashflow_records),
            "withdrawals": pd.DataFrame(self.withdrawal_records),
            "match_ts": pd.DataFrame(self.match_records).set_index("Date"),
            "rung_analytics": pd.DataFrame(self.rung_analytics_records),
            "target_log": pd.DataFrame(self.target_log_records),
            "summary": self._summary(),
        }

    def _summary(self) -> dict:
        dt = self._last_date or pd.Timestamp(self.end)
        comps = self._value_components(dt)
        risk = self._portfolio_risk(dt)
        total_due = sum(r["Due"] for r in self.withdrawal_records)
        total_paid = sum(r["Paid"] for r in self.withdrawal_records)
        total_shortfall = sum(r["Shortfall"] for r in self.withdrawal_records)
        return {
            "Strategy": self.strategy,
            "Final Value": round(comps["Portfolio Value"], 2),
            "Cash Balance": round(comps["Cash Balance"], 2),
            "Bond Market Value": round(comps["Bond Market Value"], 2),
            "Total Par": round(comps["Total Par"], 2),
            "Total Coupon Cashflow": round(sum(r["Amount"] for r in self.cashflow_records if r["Type"] == "Coupon"), 2),
            "Total Principal Cashflow": round(sum(r["Amount"] for r in self.cashflow_records if r["Type"] == "Principal"), 2),
            "Total Withdrawal Due": round(total_due, 2),
            "Total Withdrawal Paid": round(total_paid, 2),
            "Total Shortfall": round(total_shortfall, 2),
            "Coverage Ratio": round(total_paid / total_due, 6) if total_due else np.nan,
            "Portfolio Duration": round(risk["Portfolio Duration"], 6),
            "Portfolio DV01": round(risk["Portfolio DV01"], 6),
        }

    def _holdings_view(self) -> pd.DataFrame:
        rows = []
        dt = self._last_date or pd.Timestamp(self.end)
        curve = self.curve_history[dt]
        comps = self._value_components(dt)
        for h in self.holdings:
            if dt >= h.bond.maturity_dt:
                continue
            dirty = h.bond.dirty_price(dt, curve)
            clean = h.bond.clean_price(dt, curve)
            mv = h.quantity * dirty / 100.0
            rows.append(
                {
                    "Lot ID": h.lot_id,
                    "Rung": f"{h.rung_target}Y",
                    "Maturity": h.bond.maturity,
                    "Coupon": f"{h.bond.coupon * 100:.2f}%",
                    "Par Amount": round(h.quantity, 2),
                    "Purchase Date": h.purchase_date,
                    "Purchase Price": round(h.purchase_price, 4),
                    "Dirty Price": round(dirty, 4),
                    "Clean Price": round(clean, 4),
                    "Market Value": round(mv, 2),
                    "Weight": round(mv / comps["Portfolio Value"], 6) if comps["Portfolio Value"] else np.nan,
                    "Years To Maturity": round((h.bond.maturity_dt - dt).days / 365.25, 4),
                    "DV01": round(h.quantity * h.bond.dv01(dt, curve) / 100.0, 6),
                    "Modified Duration": round(h.bond.modified_duration(dt, curve), 6),
                    "Convexity Dollar": round(h.quantity * h.bond.convexity_dollar(dt, curve) / 100.0, 6),
                }
            )
        return pd.DataFrame(rows)
