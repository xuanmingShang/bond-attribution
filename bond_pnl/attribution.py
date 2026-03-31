"""
attribution.py – Campisi-style daily PnL attribution (exact decomposition).

Follows the sequence from the lesson:

    Step 0: PV(T-1, Curve_{T-1})              ← base
    Step 1: PV(T-1, Curve_T)                  ← Market Impact = Step1 − Step0
    Step 2: PV(T, Curve_T)                    ← Time Impact   = Step2 − Step1

    Per the lesson:
      Actual   ≈ PV(T) − PV(T-1) + CashFlow(T) − Funding(T)
      Actual   = Market Impact + Time Impact  (exact, net of funding)

    CashFlow includes both coupon *and* principal at maturity, so that
    the terminal-day PnL is correct (dirty_price returns 0 after maturity).

Market Impact is further decomposed:
    Duration   = −DV01 × Δr_local  (bp)  [Gaussian-weighted local shift]
    Convexity  = ½ × ConvDollar × Δr² (bp²)
    Reshape    = exact_market − PV(T-1, curve_local_shifted) + PV(T-1, curve_{T-1})
               = non-local-shift portion of the curve move
    Rate Residual = exact_market − duration − convexity − reshape

Time Impact is decomposed (net of funding):
    Accrual    = coupon income earned
    Rolldown   = clean-price change from shortened maturity on Curve_T
    Funding    = DirtyPV(T-1) × financing_rate × dt
    Carry      = Accrual + Rolldown − Funding
    Time Residual = time_impact − carry  (should be very small)

Total Residual = Rate Residual + Time Residual  (should be very small)
"""
from __future__ import annotations
import numpy as np, pandas as pd
from dataclasses import dataclass
from .bond import BondSpec
from .yield_curve import YieldCurve, YieldCurveHistory, TENOR_YEARS


@dataclass
class DailyAttribution:
    date: pd.Timestamp
    prev_date: pd.Timestamp
    dirty_prev: float          # PV(T-1, curve_{T-1})
    dirty_curr: float          # PV(T,   curve_T)
    actual_pnl: float          # dirty_curr − dirty_prev  (+ any coupon cf)

    # --- Market Impact breakdown ---
    market_impact: float       # exact: PV(T-1, curve_T) − PV(T-1, curve_{T-1})
    duration_effect: float
    convexity_effect: float
    curve_reshape: float
    rate_residual: float       # market − (dur + conv + reshape)

    # --- Time Impact breakdown ---
    time_impact: float         # exact: PV(T, curve_T) − PV(T-1, curve_T)
    accrual: float
    rolldown: float
    funding: float
    carry: float
    time_residual: float       # time_impact − carry

    # --- Combined ---
    residual: float            # rate_residual + time_residual
    coupon_cf: float           # coupon cash received in period (added to actual)
    ytm: float
    dv01: float
    delta_r_bp: float          # local (Gaussian-weighted) shift used (bp)

    def to_dict(self):
        r = lambda x: round(x, 6)
        return {
            "Date":           self.date.strftime("%Y-%m-%d"),
            "Prev Date":      self.prev_date.strftime("%Y-%m-%d"),
            "Dirty PV(T-1)":  r(self.dirty_prev),
            "Dirty PV(T)":    r(self.dirty_curr),
            "Actual PnL":     r(self.actual_pnl),
            "Market Impact":  r(self.market_impact),
            "Duration":       r(self.duration_effect),
            "Convexity":      r(self.convexity_effect),
            "Curve Reshape":  r(self.curve_reshape),
            "Rate Residual":  r(self.rate_residual),
            "Time Impact":    r(self.time_impact),
            "Accrual":        r(self.accrual),
            "Rolldown":       r(self.rolldown),
            "Funding":        r(self.funding),
            "Carry":          r(self.carry),
            "Time Residual":  r(self.time_residual),
            "Residual":       r(self.residual),
            "Coupon CF":      r(self.coupon_cf),
            "TTM":            r(self.ytm),
            "DV01":           r(self.dv01),
            "Δr (bp)":        r(self.delta_r_bp),
        }


def _local_shift_bp(curve_prev: YieldCurve, curve_curr: YieldCurve,
                        bond_ttm: float) -> float:
    """Gaussian-weighted average yield change (bp) centred at bond TTM.

    This is a *local* (not parallel) shift: it weights tenors near the
    bond's maturity more heavily, giving a representative scalar rate move
    for the Duration effect.  The Curve Reshape bucket captures the
    difference between this local approximation and the full curve change.
    """
    delta_pct = curve_curr.yields_pct - curve_prev.yields_pct
    t = curve_prev.tenors
    sig = max(bond_ttm * 0.3, 1.0)
    w = np.exp(-0.5 * ((t - bond_ttm) / sig) ** 2)
    w /= w.sum()
    return float(np.dot(w, delta_pct)) * 100.0   # % → bp


def compute_daily_attribution(
    bond: BondSpec,
    curve_prev: YieldCurve,
    curve_curr: YieldCurve,
    d_prev: pd.Timestamp,
    d_curr: pd.Timestamp,
    financing_rate: float = 0.05,
) -> DailyAttribution:

    dcf = bond._dcf
    dt  = dcf(d_prev, d_curr)          # year-fraction

    # ── Step 0  PV(T-1, curve_{T-1}) ─────────────────────────────────
    pv_prev = bond.dirty_price(d_prev, curve_prev)

    # ── Step 1  PV(T-1, curve_T)  →  Market Impact ───────────────────
    pv_rate_only = bond.dirty_price(d_prev, curve_curr)
    market_impact = pv_rate_only - pv_prev

    # ── Step 2  PV(T, curve_T)    →  Time Impact ─────────────────────
    pv_curr = bond.dirty_price(d_curr, curve_curr)

    # Cashflows received in (d_prev, d_curr] — includes coupon AND principal
    # at maturity.  dirty_price() returns 0 after maturity, so the principal
    # must appear here to keep PnL correct on the terminal day.
    coupon_cf = 0.0
    for cf_date, cf_amt in bond.cashflow_schedule(d_prev):
        if d_prev < cf_date <= d_curr:
            coupon_cf += cf_amt          # full cashflow (coupon + principal at mat)

    # Funding cost: lesson says DirtyPV(T-1) × FinancingRate × DayCount
    funding = pv_prev * financing_rate * dt

    # Per lesson: Actual ≈ PV(T) - PV(T-1) + CashFlow(T) - Funding(T)
    # Time Impact is NET of funding so that Actual = Market + Time (exact)
    time_impact_gross = pv_curr - pv_rate_only + coupon_cf
    time_impact = time_impact_gross - funding

    actual_pnl = market_impact + time_impact   # ≡ pv_curr − pv_prev + coupon_cf − funding

    # ── Market Impact decomposition ───────────────────────────────────
    ttm      = bond.ytm_tenor(d_prev)
    dv01_val = bond.dv01(d_prev, curve_prev)
    delta_bp = _local_shift_bp(curve_prev, curve_curr, ttm)

    duration_effect  = -dv01_val * delta_bp
    conv             = bond.convexity_dollar(d_prev, curve_prev)
    convexity_effect = 0.5 * conv * delta_bp ** 2

    # Reshape: difference between exact market impact under full new curve
    # vs under a parallel-shifted old curve
    from .yield_curve import YieldCurve as YC
    shifted_pct = curve_prev.yields_pct + delta_bp * 0.01   # bp → %
    curve_par   = YC(curve_curr.date, shifted_pct)
    pv_parallel = bond.dirty_price(d_prev, curve_par)
    # reshape = (exact under full curve) − (under parallel shift)
    curve_reshape = market_impact - (pv_parallel - pv_prev)

    # What parallel shift explains via Taylor:
    taylor_parallel = duration_effect + convexity_effect
    # The parallel-shift exact price change:
    parallel_exact  = pv_parallel - pv_prev
    # rate_residual captures both Taylor truncation and reshape approximation
    rate_residual = market_impact - (taylor_parallel + curve_reshape)

    # ── Time Impact decomposition ─────────────────────────────────────
    accrual = bond.face * bond.coupon * dt

    # Rolldown on the NEW curve (curve_T)
    # If bond matures in (d_prev, d_curr], clean price "at maturity" is face
    # (the redemption value), not 0; otherwise use normal clean price.
    matures_in_period = d_prev < bond.maturity_dt <= d_curr
    clean_at_prev = bond.clean_price(d_prev, curve_curr)
    clean_at_curr = bond.face if matures_in_period else bond.clean_price(d_curr, curve_curr)
    rolldown = clean_at_curr - clean_at_prev

    # funding already computed above (pv_prev * financing_rate * dt)
    carry   = accrual + rolldown - funding

    # time_impact is already net of funding, so residual is the small
    # mismatch between exact time passage and carry approximation
    time_residual = time_impact - carry

    residual = rate_residual + time_residual

    return DailyAttribution(
        date=d_curr, prev_date=d_prev,
        dirty_prev=pv_prev, dirty_curr=pv_curr,
        actual_pnl=actual_pnl,
        market_impact=market_impact,
        duration_effect=duration_effect,
        convexity_effect=convexity_effect,
        curve_reshape=curve_reshape,
        rate_residual=rate_residual,
        time_impact=time_impact,
        accrual=accrual, rolldown=rolldown,
        funding=funding, carry=carry,
        time_residual=time_residual,
        residual=residual,
        coupon_cf=coupon_cf,
        ytm=ttm, dv01=dv01_val,
        delta_r_bp=delta_bp,
    )


def run_attribution(bond, curve_history, start, end,
                    financing_rate=0.05) -> pd.DataFrame:
    dates = curve_history.business_dates(start, end)
    if len(dates) < 2:
        raise ValueError("Need ≥2 business dates")
    rows = []
    for i in range(1, len(dates)):
        dp, dc = dates[i - 1], dates[i]
        if pd.Timestamp(dp) >= bond.maturity_dt:
            break
        a = compute_daily_attribution(
            bond, curve_history[dp], curve_history[dc],
            pd.Timestamp(dp), pd.Timestamp(dc), financing_rate)
        rows.append(a.to_dict())
    df = pd.DataFrame(rows)
    if not df.empty:
        df.set_index("Date", inplace=True)
    return df


def attribution_summary(df: pd.DataFrame) -> pd.Series:
    cols = ["Actual PnL", "Market Impact", "Duration", "Convexity",
            "Curve Reshape", "Rate Residual",
            "Time Impact", "Accrual", "Rolldown", "Funding", "Carry",
            "Time Residual", "Residual", "Coupon CF"]
    return df[[c for c in cols if c in df.columns]].sum().rename("Total")
