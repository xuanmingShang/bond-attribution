"""
Tests for pca.py – PCA yield-curve analysis.
"""
import pytest, numpy as np, pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bond_pnl.bond import BondSpec
from bond_pnl.pca import fit_pca, pca_attribution
from bond_pnl.yield_curve import YieldCurveHistory, TENOR_YEARS, TENOR_LABELS


def _random_curve_hist(n_days=201, seed=42):
    """Build a YieldCurveHistory with random yields so .changes() works."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-06-01", periods=n_days)
    # base level + small random walk
    levels = 4.0 + np.cumsum(rng.normal(0, 0.05, (n_days, len(TENOR_YEARS))), axis=0)
    df = pd.DataFrame(levels, index=dates, columns=TENOR_LABELS)
    return YieldCurveHistory(df)


class TestFitPCA:

    def test_variance_explained_sums_le_1(self):
        ch = _random_curve_hist()
        res = fit_pca(ch, n_components=3)
        assert res.explained_variance_ratio.sum() <= 1.0 + 1e-9

    def test_loadings_shape(self):
        ch = _random_curve_hist()
        res = fit_pca(ch, n_components=3)
        assert res.loadings.shape == (3, len(TENOR_YEARS))

    def test_more_components_more_variance(self):
        ch = _random_curve_hist()
        r2 = fit_pca(ch, n_components=2)
        r3 = fit_pca(ch, n_components=3)
        assert r3.explained_variance_ratio.sum() >= r2.explained_variance_ratio.sum() - 1e-9

    def test_loadings_orthogonal(self):
        ch = _random_curve_hist()
        res = fit_pca(ch, n_components=3)
        C = res.loadings.values  # (3, n_tenors)
        gram = C @ C.T
        off_diag = gram - np.diag(np.diag(gram))
        assert np.allclose(off_diag, 0, atol=1e-10)


class TestPCAAttribution:
    def test_mean_pnl_is_not_exposed_and_identity_holds(self):
        ch = _random_curve_hist(n_days=90)
        start = "2023-06-01"
        end = "2023-09-29"
        pr = fit_pca(ch, start, end, n_components=3)
        bond = BondSpec(
            maturity="2028-06-01",
            face=100.0,
            coupon=0.04,
            freq=2,
            day_count="ACT/ACT",
            issue_date=start,
        )

        attr = pca_attribution(bond, ch, pr, start, end, financing_rate=0.03)

        assert "Mean PnL" not in attr.columns
        assert {"PC1 PnL", "PC2 PnL", "PC3 PnL", "PC Total", "Residual"}.issubset(attr.columns)

        pc_sum = attr[["PC1 PnL", "PC2 PnL", "PC3 PnL"]].sum(axis=1)
        assert np.allclose(attr["PC Total"], pc_sum, atol=3e-6)

        explained = attr["Carry"] + attr["PC Total"] + attr["Residual"]
        assert np.allclose(attr["Actual PnL"], explained, atol=3e-6)
