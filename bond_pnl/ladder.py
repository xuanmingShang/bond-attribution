"""
ladder.py – Bond-ladder backtest (Optional Feature 2).
Five-rung *synthetic* Treasury ladder: construct, track, rebalance, attribute.

Design notes
------------
* Bonds are synthetic par bonds — the coupon is solved numerically so that
  dirty_price(settle) == face under the current yield curve.
* Attribution uses **clean prices** for rolldown and rate-movement buckets
  to avoid double-counting accrual already captured in the Income bucket.
* Holdings are **snapshotted at the start of each period** before any coupon
  collection, maturity processing, or rebalancing, so that the attribution
  explains the correct day's PnL.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from dataclasses import dataclass, field
from .bond import BondSpec
from .yield_curve import YieldCurveHistory

DEFAULT_RUNGS = [2, 5, 7, 10, 30]


@dataclass
class LadderHolding:
    bond: BondSpec
    quantity: float          # par amount
    rung_target: int
    purchase_date: str
    purchase_price: float


@dataclass
class RebalanceRecord:
    date: str; action: str; rung: int; bond_maturity: str
    coupon: float; par_amount: float; price: float


@dataclass
class LadderBacktest:
    curve_history: YieldCurveHistory
    start: str; end: str
    rungs: list[int] = field(default_factory=lambda: DEFAULT_RUNGS.copy())
    capital: float = 1_000_000.0
    rebal_months: int = 12
    financing_rate: float = 0.05

    def __post_init__(self):
        self.holdings: list[LadderHolding] = []
        self.cash: float = self.capital
        self.rebalance_log: list[RebalanceRecord] = []
        self.portfolio_ts: list[dict] = []
        self.attr_records: list[dict] = []

    # ── helpers ───────────────────────────────────────────────────────
    def _make_bond(self, rung: int, settle: pd.Timestamp) -> BondSpec:
        """Create a synthetic par bond for *rung* at *settle*.

        The coupon is solved numerically so that `dirty_price(settle) == 100`
        under the current curve, making this a true par bond.

        Supports negative-rate environments by allowing the coupon bracket
        to extend below zero.
        """
        c = self.curve_history[settle]
        mat = settle + pd.DateOffset(years=rung)
        guess = c.rate(rung)

        def _price_at_cpn(cpn):
            return BondSpec(maturity=mat.strftime("%Y-%m-%d"), face=100.0,
                            coupon=cpn, freq=2, day_count="ACT/ACT",
                            issue_date=settle.strftime("%Y-%m-%d")
                            ).dirty_price(settle, c)

        # Bracket the par coupon.  price(cpn) is monotonically increasing
        # in cpn, so we need lo where price < 100 and hi where price >= 100.
        lo = min(guess - 0.05, -0.05)
        hi = max(abs(guess) * 3, 0.10) + 0.01
        # Widen hi until price(hi) >= 100
        while _price_at_cpn(hi) < 100.0:
            hi *= 2
        # Widen lo until price(lo) < 100
        while _price_at_cpn(lo) >= 100.0:
            lo -= 0.05
        # Bisect
        for _ in range(80):
            mid_c = (lo + hi) / 2.0
            if _price_at_cpn(mid_c) < 100.0:
                lo = mid_c
            else:
                hi = mid_c
        par_cpn = round((lo + hi) / 2.0, 6)
        result = BondSpec(maturity=mat.strftime("%Y-%m-%d"), face=100.0,
                          coupon=par_cpn, freq=2, day_count="ACT/ACT",
                          issue_date=settle.strftime("%Y-%m-%d"))
        # Validate
        px = result.dirty_price(settle, c)
        if abs(px - 100.0) > 0.01:
            raise RuntimeError(
                f"Par bond solver failed for {rung}Y at {settle}: "
                f"price={px:.4f}, coupon={par_cpn:.6f}")
        return result

    def _init(self, settle: pd.Timestamp):
        per = self.capital / len(self.rungs)
        for rung in self.rungs:
            b = self._make_bond(rung, settle)
            c = self.curve_history[settle]
            p = b.dirty_price(settle, c)
            q = per / p * 100.0
            self.holdings.append(LadderHolding(b, q, rung,
                settle.strftime("%Y-%m-%d"), p))
            self.cash -= q * p / 100.0
            self.rebalance_log.append(RebalanceRecord(
                settle.strftime("%Y-%m-%d"), "INIT", rung,
                b.maturity, b.coupon, round(q, 2), round(p, 4)))

    def _value(self, dt: pd.Timestamp) -> float:
        c = self.curve_history[dt]
        v = self.cash
        for h in self.holdings:
            if dt < h.bond.maturity_dt:
                v += h.quantity * h.bond.dirty_price(dt, c) / 100.0
            else:
                v += h.quantity          # par redemption
        return v

    def _rebalance(self, dt: pd.Timestamp):
        c = self.curve_history[dt]
        for h in self.holdings:
            if dt < h.bond.maturity_dt:
                p = h.bond.dirty_price(dt, c)
                self.cash += h.quantity * p / 100.0
            else:
                p = 100.0
                self.cash += h.quantity
            self.rebalance_log.append(RebalanceRecord(
                dt.strftime("%Y-%m-%d"), "SELL", h.rung_target,
                h.bond.maturity, h.bond.coupon,
                round(h.quantity, 2), round(p, 4)))
        self.holdings.clear()
        per = self.cash / len(self.rungs)
        for rung in self.rungs:
            b = self._make_bond(rung, dt)
            p = b.dirty_price(dt, c)
            q = per / p * 100.0
            self.holdings.append(LadderHolding(b, q, rung,
                dt.strftime("%Y-%m-%d"), p))
            self.cash -= q * p / 100.0
            self.rebalance_log.append(RebalanceRecord(
                dt.strftime("%Y-%m-%d"), "BUY", rung,
                b.maturity, b.coupon, round(q, 2), round(p, 4)))

    def _collect_coupons(self, dp: pd.Timestamp, dc: pd.Timestamp):
        for h in self.holdings:
            for cf_d, cf_a in h.bond.cashflow_schedule(dp):
                if dp < cf_d <= dc:
                    cpn = cf_a - (h.bond.face
                                  if cf_d == h.bond.maturity_dt else 0.0)
                    self.cash += h.quantity * cpn / 100.0

    # ── main loop ─────────────────────────────────────────────────────
    def run(self) -> dict:
        dates = self.curve_history.business_dates(self.start, self.end)
        if len(dates) < 2:
            raise ValueError("Need ≥2 dates")
        self._init(pd.Timestamp(dates[0]))
        next_reb = pd.Timestamp(dates[0]) + pd.DateOffset(
            months=self.rebal_months)

        pv = self._value(pd.Timestamp(dates[0]))
        self.portfolio_ts.append({
            "Date": (dates[0].strftime("%Y-%m-%d")
                     if hasattr(dates[0], 'strftime') else str(dates[0])),
            "Portfolio Value": round(pv, 2),
            "Daily Return": 0.0,
            "Cumulative Return": 0.0,
        })
        prev_val = pv

        for i in range(1, len(dates)):
            dp = pd.Timestamp(dates[i - 1])
            dc = pd.Timestamp(dates[i])

            # === Snapshot start-of-period holdings for attribution ====
            sop_holdings = list(self.holdings)      # shallow copy of list
            cp = self.curve_history[dp]
            cc = self.curve_history[dc]
            dt_frac = (dc - dp).days / 365.25

            # --- Compute attribution on start-of-period holdings ------
            # Income: change in accrued interest + coupon received.
            # This exactly captures coupon-date resets and maturity.
            income = 0.0
            for h in sop_holdings:
                if h.bond.maturity_dt <= dp:
                    continue
                ai_prev = h.bond.accrued_interest(dp)
                ai_curr = h.bond.accrued_interest(dc)   # 0 after maturity
                cpn_received = 0.0
                for cf_d, cf_a in h.bond.cashflow_schedule(dp):
                    if dp < cf_d <= dc:
                        cpn_received += cf_a - (h.bond.face
                                                if cf_d == h.bond.maturity_dt else 0.0)
                income += h.quantity * (ai_curr - ai_prev + cpn_received) / 100.0

            # Rolldown & rate movement use CLEAN prices to avoid
            # double-counting the accrual already in Income.
            # Matured bonds get pull-to-par treatment in rolldown.
            roll_pnl = 0.0
            rate_pnl = 0.0
            for h in sop_holdings:
                if h.bond.maturity_dt <= dp:
                    continue  # already matured before this period
                if h.bond.maturity_dt <= dc:
                    # Matures in this period: pull-to-par
                    cp_clean = h.bond.clean_price(dp, cp)
                    roll_pnl += h.quantity * (h.bond.face - cp_clean) / 100.0
                    # rate movement negligible at maturity (price = par)
                else:
                    cp_clean = h.bond.clean_price(dp, cp)
                    cr_clean = h.bond.clean_price(dc, cp)
                    cc_clean = h.bond.clean_price(dc, cc)
                    roll_pnl += h.quantity * (cr_clean - cp_clean) / 100.0
                    rate_pnl += h.quantity * (cc_clean - cr_clean) / 100.0

            # === Now mutate state: coupons, maturity, rebalance =======
            self._collect_coupons(dp, dc)
            for h in list(self.holdings):
                if h.bond.maturity_dt <= dc and h.bond.maturity_dt > dp:
                    self.cash += h.quantity
                    self.holdings.remove(h)
            if dc >= next_reb and self.holdings:
                self._rebalance(dc)
                next_reb = dc + pd.DateOffset(months=self.rebal_months)

            cv = self._value(dc)
            dr = (cv - prev_val) / prev_val if prev_val else 0.0
            cr = (cv - self.capital) / self.capital

            self.portfolio_ts.append({
                "Date": dc.strftime("%Y-%m-%d"),
                "Portfolio Value": round(cv, 2),
                "Daily Return": round(dr, 6),
                "Cumulative Return": round(cr, 6),
            })

            res = (cv - prev_val) - income - roll_pnl - rate_pnl
            self.attr_records.append({
                "Date": dc.strftime("%Y-%m-%d"),
                "Total PnL": round(cv - prev_val, 2),
                "Income": round(income, 2),
                "Rolldown": round(roll_pnl, 2),
                "Rate Movement": round(rate_pnl, 2),
                "Residual": round(res, 2),
            })
            prev_val = cv

        return {
            "portfolio_ts": pd.DataFrame(self.portfolio_ts).set_index("Date"),
            "attribution": pd.DataFrame(self.attr_records).set_index("Date"),
            "rebalance_log": pd.DataFrame(
                [r.__dict__ for r in self.rebalance_log]),
            "holdings": self._holdings_view(),
        }

    def _holdings_view(self) -> pd.DataFrame:
        rows = []
        for h in self.holdings:
            rows.append({
                "Rung": f"{h.rung_target}Y",
                "Maturity": h.bond.maturity,
                "Coupon": f"{h.bond.coupon*100:.2f}%",
                "Par Amount": round(h.quantity, 2),
                "Purchase Date": h.purchase_date,
                "Purchase Price": round(h.purchase_price, 4),
            })
        return pd.DataFrame(rows)
