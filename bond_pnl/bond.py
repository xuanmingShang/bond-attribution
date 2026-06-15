"""
bond.py – Bond specification, cashflow scheduling, pricing, risk measures.

Unit conventions
----------------
- coupon  : DECIMAL  (0.04 = 4 %)
- YieldCurve.yields_pct  : PERCENT  (4.0 = 4 %)
- YieldCurve.yields      : DECIMAL  (0.04)
- DV01   : absolute price change per 1 bp parallel shift
- 1 bp = 0.0001 decimal = 0.01 %

Model assumptions
-----------------
- FRED constant-maturity Treasury (CMT) yields are used as a proxy for
  continuously-compounded zero-coupon spot rates.  CMT yields are actually
  par-style yields, so price levels will differ slightly from full
  bootstrap-derived zero-curve pricing.  For this pedagogical project the
  approximation is acceptable and documented.
- ACT/ACT day count is simplified to actual-days / 365.25 (not the true
  ISDA ACT/ACT or Treasury coupon-period-based convention).  The impact on
  accrued interest and year-fraction calculations is small for standard
  semi-annual Treasury bonds.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from dataclasses import dataclass, field

from .yield_curve import YieldCurve

# ── day-count ─────────────────────────────────────────────────────────
def _act_act(d1, d2):
    return (d2 - d1).days / 365.25

def _30_360(d1, d2):
    y1, m1, dd1 = d1.year, d1.month, min(d1.day, 30)
    y2, m2, dd2 = d2.year, d2.month, min(d2.day, 30)
    return (360*(y2-y1) + 30*(m2-m1) + (dd2-dd1)) / 360.0

DC = {"ACT/ACT": _act_act, "ACT/365": _act_act, "30/360": _30_360}

@dataclass
class BondSpec:
    maturity: str
    face: float = 100.0
    coupon: float = 0.04        # decimal
    freq: int = 2               # semi-annual
    day_count: str = "ACT/ACT"
    issue_date: str | None = None
    _cashflow_cache: dict[pd.Timestamp, tuple[tuple[pd.Timestamp, float], ...]] = field(
        init=False, repr=False, default_factory=dict
    )
    _cashflow_array_cache: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]] = field(
        init=False, repr=False, default_factory=dict
    )
    _accrued_cache: dict[pd.Timestamp, float] = field(init=False, repr=False, default_factory=dict)
    _all_cf_dates: pd.DatetimeIndex | None = field(init=False, repr=False, default=None)
    _all_cf_amounts: np.ndarray | None = field(init=False, repr=False, default=None)

    def __post_init__(self):
        self.maturity_dt = pd.Timestamp(self.maturity)
        self.issue_dt = pd.Timestamp(self.issue_date) if self.issue_date else None
        self._dcf = DC[self.day_count]
        self._build_full_cashflows()

    def _build_full_cashflows(self):
        if self.issue_dt is None:
            return
        cpn = self.face * self.coupon / self.freq
        mo = 12 // self.freq
        dates, d = [], self.maturity_dt
        while d > self.issue_dt:
            dates.append(d)
            d -= pd.DateOffset(months=mo)
        dates.sort()
        amounts = np.full(len(dates), cpn, dtype=float)
        if len(amounts):
            amounts[-1] += self.face
        self._all_cf_dates = pd.DatetimeIndex(dates)
        self._all_cf_amounts = amounts

    # ── cashflows after settle ────────────────────────────────────────
    def cashflow_schedule(self, settle):
        settle = pd.Timestamp(settle)
        cached = self._cashflow_cache.get(settle)
        if cached is not None:
            return list(cached)

        if self._all_cf_dates is not None and self._all_cf_amounts is not None:
            pos = self._all_cf_dates.searchsorted(settle, side="right")
            out = [
                (pd.Timestamp(dt), float(cf))
                for dt, cf in zip(self._all_cf_dates[pos:], self._all_cf_amounts[pos:])
            ]
            self._cashflow_cache[settle] = tuple(out)
            return out

        cpn = self.face * self.coupon / self.freq
        mo  = 12 // self.freq
        dates, d = [], self.maturity_dt
        while d > settle:
            dates.append(d); d -= pd.DateOffset(months=mo)
        dates.sort()
        out = []
        for i, dt in enumerate(dates):
            cf = cpn + (self.face if i == len(dates)-1 else 0.0)
            out.append((dt, cf))
        self._cashflow_cache[settle] = tuple(out)
        return out

    def _cashflow_arrays(self, settle):
        settle = pd.Timestamp(settle)
        cached = self._cashflow_array_cache.get(settle)
        if cached is not None:
            return cached

        if self._all_cf_dates is not None and self._all_cf_amounts is not None:
            pos = self._all_cf_dates.searchsorted(settle, side="right")
            dates = self._all_cf_dates[pos:]
            cashflows = self._all_cf_amounts[pos:]
            if len(dates) == 0:
                arrays = (np.array([], dtype=float), np.array([], dtype=float))
                self._cashflow_array_cache[settle] = arrays
                return arrays
            if self.day_count in {"ACT/ACT", "ACT/365"}:
                times = np.array((dates - settle).days, dtype=float) / 365.25
            else:
                times = np.array([self._dcf(settle, dt) for dt in dates], dtype=float)
            arrays = (times, np.asarray(cashflows, dtype=float))
            self._cashflow_array_cache[settle] = arrays
            return arrays

        schedule = self.cashflow_schedule(settle)
        if not schedule:
            arrays = (np.array([], dtype=float), np.array([], dtype=float))
            self._cashflow_array_cache[settle] = arrays
            return arrays

        times = np.array([self._dcf(settle, dt) for dt, _ in schedule], dtype=float)
        cashflows = np.array([cf for _, cf in schedule], dtype=float)
        valid = times > 0
        arrays = (times[valid], cashflows[valid])
        self._cashflow_array_cache[settle] = arrays
        return arrays

    # ── pricing (continuous discounting) ──────────────────────────────
    def dirty_price(self, settle, curve):
        times, cashflows = self._cashflow_arrays(settle)
        if len(times) == 0:
            return 0.0   # after maturity, principal is a cashflow, not a price
        rates = curve.rates(times)
        return float(np.sum(cashflows * np.exp(-rates * times)))

    def accrued_interest(self, settle):
        settle = pd.Timestamp(settle)
        cached = self._accrued_cache.get(settle)
        if cached is not None:
            return cached

        # After maturity no accrual is meaningful
        if settle >= self.maturity_dt:
            self._accrued_cache[settle] = 0.0
            return 0.0
        if self.issue_dt is not None and settle <= self.issue_dt:
            self._accrued_cache[settle] = 0.0
            return 0.0
        cpn = self.face * self.coupon / self.freq
        mo  = 12 // self.freq
        if self._all_cf_dates is not None:
            pos = self._all_cf_dates.searchsorted(settle, side="right")
            if pos < len(self._all_cf_dates):
                nxt = self._all_cf_dates[pos]
                prev = self._all_cf_dates[pos - 1] if pos > 0 else self.issue_dt
                denom = self._dcf(prev, nxt)
                accrued = cpn * self._dcf(prev, settle) / denom if denom else 0.0
                self._accrued_cache[settle] = accrued
                return accrued

        schedule = self.cashflow_schedule(settle)
        if schedule:
            nxt = schedule[0][0]
            prev = nxt - pd.DateOffset(months=mo)
            if self.issue_dt is not None and prev < self.issue_dt:
                prev = self.issue_dt
            denom = self._dcf(prev, nxt)
            accrued = cpn * self._dcf(prev, settle) / denom if denom else 0.0
            self._accrued_cache[settle] = accrued
            return accrued
        self._accrued_cache[settle] = 0.0
        return 0.0

    def clean_price(self, settle, curve):
        # After maturity both dirty and clean are 0
        if settle >= self.maturity_dt:
            return 0.0
        return self.dirty_price(settle, curve) - self.accrued_interest(settle)

    def ytm_tenor(self, settle):
        return self._dcf(settle, self.maturity_dt)

    # ── DV01 (per 1 bp) ──────────────────────────────────────────────
    def dv01(self, settle, curve, bump_bp=1.0):
        return self.risk_measures(settle, curve, bump_bp)["dv01"]

    # ── dollar convexity (per bp^2) ───────────────────────────────────
    def convexity_dollar(self, settle, curve, bump_bp=1.0):
        return self.risk_measures(settle, curve, bump_bp)["convexity_dollar"]

    def modified_duration(self, settle, curve):
        return self.risk_measures(settle, curve)["modified_duration"]

    def risk_measures(self, settle, curve, bump_bp=1.0):
        """Return dirty price, DV01, convexity, and modified duration together."""
        times, cashflows = self._cashflow_arrays(settle)
        if len(times) == 0:
            return {
                "dirty_price": 0.0,
                "dv01": 0.0,
                "convexity_dollar": 0.0,
                "modified_duration": 0.0,
            }

        rates = curve.rates(times)
        bump = bump_bp * 0.0001
        dirty = float(np.sum(cashflows * np.exp(-rates * times)))
        dirty_up = float(np.sum(cashflows * np.exp(-(rates + bump) * times)))
        dirty_dn = float(np.sum(cashflows * np.exp(-(rates - bump) * times)))
        dv01 = (dirty_dn - dirty_up) / (2.0 * bump_bp)
        convexity = (dirty_up + dirty_dn - 2.0 * dirty) / (bump_bp**2)
        duration = dv01 / dirty * 10_000.0 if dirty else 0.0
        return {
            "dirty_price": dirty,
            "dv01": dv01,
            "convexity_dollar": convexity,
            "modified_duration": duration,
        }

    def summary(self):
        return {"Maturity": self.maturity, "Face": self.face,
                "Coupon": f"{self.coupon*100:.2f}%",
                "Frequency": f"{self.freq}x/year",
                "Day Count": self.day_count}
