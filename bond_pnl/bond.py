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
from dataclasses import dataclass

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

    def __post_init__(self):
        self.maturity_dt = pd.Timestamp(self.maturity)
        self.issue_dt = pd.Timestamp(self.issue_date) if self.issue_date else None
        self._dcf = DC[self.day_count]

    # ── cashflows after settle ────────────────────────────────────────
    def cashflow_schedule(self, settle):
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
        return out

    # ── pricing (continuous discounting) ──────────────────────────────
    def dirty_price(self, settle, curve):
        pv = 0.0
        for dt, cf in self.cashflow_schedule(settle):
            t = self._dcf(settle, dt)
            if t <= 0:
                continue
            pv += cf * np.exp(-curve.rate(t) * t)
        return pv   # 0.0 after maturity (principal is a cashflow, not a price)

    def accrued_interest(self, settle):
        # After maturity no accrual is meaningful
        if settle >= self.maturity_dt:
            return 0.0
        cpn = self.face * self.coupon / self.freq
        mo  = 12 // self.freq
        d   = self.maturity_dt
        earliest = self.issue_dt or pd.Timestamp("1900-01-01")
        while d >= earliest:
            if d <= settle:
                nxt = d + pd.DateOffset(months=mo)
                denom = self._dcf(d, nxt)
                return cpn * self._dcf(d, settle) / denom if denom else 0.0
            d -= pd.DateOffset(months=mo)
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
        from .yield_curve import YieldCurve as YC
        s = bump_bp * 0.01                 # 1 bp = 0.01 in %-space
        up = YC(curve.date, curve.yields_pct + s)
        dn = YC(curve.date, curve.yields_pct - s)
        return (self.dirty_price(settle, dn) - self.dirty_price(settle, up)) / (2.0 * bump_bp)

    # ── dollar convexity (per bp^2) ───────────────────────────────────
    def convexity_dollar(self, settle, curve, bump_bp=1.0):
        from .yield_curve import YieldCurve as YC
        s = bump_bp * 0.01
        up = YC(curve.date, curve.yields_pct + s)
        dn = YC(curve.date, curve.yields_pct - s)
        mid = self.dirty_price(settle, curve)
        return (self.dirty_price(settle, up) + self.dirty_price(settle, dn) - 2*mid) / (bump_bp**2)

    def modified_duration(self, settle, curve):
        pv = self.dirty_price(settle, curve)
        return self.dv01(settle, curve) / pv * 10_000.0 if pv else 0.0

    def summary(self):
        return {"Maturity": self.maturity, "Face": self.face,
                "Coupon": f"{self.coupon*100:.2f}%",
                "Frequency": f"{self.freq}x/year",
                "Day Count": self.day_count}
