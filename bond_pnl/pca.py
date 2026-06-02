"""
pca.py – PCA on yield-curve changes & PCA-based bond PnL attribution.
Optional Feature 1.

PCA attribution uses full repricing under each PC shock rather than a single
scalar DV01 approximation, so that slope/curvature factors are correctly captured.

PnL identity is aligned with the core module:
    Actual = PV(T) − PV(T-1) + CouponCF − Funding
"""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from dataclasses import dataclass
from .yield_curve import YieldCurveHistory, YieldCurve, TENOR_LABELS, TENOR_YEARS
from .bond import BondSpec


@dataclass
class PCAResult:
    n_components: int
    explained_variance_ratio: np.ndarray
    cumulative_variance: np.ndarray
    loadings: pd.DataFrame          # (n_components, n_tenors) in % units
    scores: pd.DataFrame            # (n_dates, n_components)
    mean_change: np.ndarray         # mean daily Δyield (%)
    pca_model: PCA

    def summary(self):
        rows = []
        interp = ["Level", "Slope", "Curvature"]
        for i in range(self.n_components):
            rows.append({
                "Component": f"PC{i+1}",
                "Var Explained": f"{self.explained_variance_ratio[i]*100:.2f}%",
                "Cumulative":    f"{self.cumulative_variance[i]*100:.2f}%",
                "Interpretation": interp[i] if i < 3 else f"PC{i+1}",
            })
        return pd.DataFrame(rows)


def fit_pca(curve_hist, start=None, end=None, n_components=3):
    """Fit PCA on daily yield-curve changes (in %).

    Returns PCAResult with loadings in % units matching the yield-change space.
    """
    ch = curve_hist.changes()
    if start: ch = ch[ch.index >= pd.Timestamp(start)]
    if end:   ch = ch[ch.index <= pd.Timestamp(end)]
    X = ch.values           # (n_dates, n_tenors) in % units
    pca = PCA(n_components=n_components)
    sc  = pca.fit_transform(X)
    ld  = pd.DataFrame(pca.components_,
                        index=[f"PC{i+1}" for i in range(n_components)],
                        columns=TENOR_LABELS[:X.shape[1]])
    sd  = pd.DataFrame(sc, index=ch.index,
                        columns=[f"PC{i+1}" for i in range(n_components)])
    return PCAResult(n_components, pca.explained_variance_ratio_,
                     np.cumsum(pca.explained_variance_ratio_),
                     ld, sd, pca.mean_, pca)


def pca_attribution(bond: BondSpec, curve_hist: YieldCurveHistory,
                    pca_res: PCAResult, start, end,
                    financing_rate: float = 0.05):
    """PCA-based bond PnL attribution via full repricing under each PC shock.

    For each day:
      1. Compute Actual PnL (same basis as core: includes coupon CF − funding)
      2. Compute Carry (accrual + rolldown − funding)
      3. For each PC k:
           - Reconstruct the curve change due to PC k alone:   Δy_k = score_k * loading_k
           - Build shifted curve:  curve_prev + Σ_{j≤k} Δy_j
           - PC k PnL = bond price under cumulative-up-to-k − under cumulative-up-to-(k-1)
         This ensures the total PC PnL sums to the full market impact when
         the retained PCs explain all the yield change.
      4. Residual = Actual − Carry − PC Total

    PCA is fitted on centered yield changes. The sample mean curve drift is
    intentionally not displayed as a separate PnL bucket; it remains in the
    residual together with omitted components and non-linear effects.
    """
    dates = curve_hist.business_dates(start, end)
    changes_df = curve_hist.changes()
    loadings = pca_res.loadings.values        # (n_components, n_tenors) in %
    rows = []
    for i in range(1, len(dates)):
        dp, dc = dates[i - 1], dates[i]
        if pd.Timestamp(dp) >= bond.maturity_dt:
            break
        cp = curve_hist[dp]
        cc = curve_hist[dc]
        d_prev = pd.Timestamp(dp)
        d_curr = pd.Timestamp(dc)
        dt = bond._dcf(d_prev, d_curr)

        pv_prev = bond.dirty_price(d_prev, cp)
        pv_curr = bond.dirty_price(d_curr, cc)

        # --- coupon CF in (d_prev, d_curr] ---
        # Full cashflow including principal at maturity (dirty_price=0 after mat)
        coupon_cf = 0.0
        for cf_date, cf_amt in bond.cashflow_schedule(d_prev):
            if d_prev < cf_date <= d_curr:
                coupon_cf += cf_amt

        # --- Aligned Actual PnL (same basis as core) ---
        funding = pv_prev * financing_rate * dt
        actual = pv_curr - pv_prev + coupon_cf - funding

        # --- Carry (accrual + rolldown on prev curve − funding) ---
        accrual = bond.face * bond.coupon * dt
        cl_p = bond.clean_price(d_prev, cp)
        # If bond matures in this interval, clean at maturity = face
        matures_in_period = d_prev < bond.maturity_dt <= d_curr
        cl_r = bond.face if matures_in_period else bond.clean_price(d_curr, cp)
        rolldown = cl_r - cl_p
        carry = accrual + rolldown - funding

        # --- PCA decomposition via full repricing ---
        # daily yield change vector (%)
        if dc in changes_df.index:
            dy = changes_df.loc[dc].values     # (n_tenors,) in %
        else:
            dy = cc.yields_pct - cp.yields_pct

        # project onto retained PCs (centered)
        dy_centered = dy - pca_res.mean_change
        scores = loadings @ dy_centered          # (n_components,)

        # Sequential repricing:
        #   1) Start from curve_prev
        #   2) Apply each retained centered PC shift cumulatively
        #   3) Residual absorbs mean drift, omitted PCs, and non-linearities
        pv_base = pv_prev
        row = {"Date": d_curr.strftime("%Y-%m-%d"),
               "Actual PnL": round(actual, 6),
               "Carry": round(carry, 6)}

        # PC components
        pc_tot = 0.0
        cumul_shift_pct = np.zeros_like(pca_res.mean_change)
        for k in range(pca_res.n_components):
            pc_shift_pct = scores[k] * loadings[k]  # (n_tenors,) in %
            cumul_shift_pct = cumul_shift_pct + pc_shift_pct
            curve_k = YieldCurve(cp.date, cp.yields_pct + cumul_shift_pct)
            pv_k = bond.dirty_price(d_prev, curve_k)
            pc_pnl = pv_k - pv_base
            row[f"PC{k+1} PnL"] = round(pc_pnl, 6)
            pc_tot += pc_pnl
            pv_base = pv_k

        row["PC Total"] = round(pc_tot, 6)
        row["Residual"] = round(actual - carry - pc_tot, 6)
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df.set_index("Date", inplace=True)
    return df
