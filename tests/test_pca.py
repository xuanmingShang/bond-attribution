"""
Tests for pca.py – PCA yield-curve analysis.
"""
import pytest, numpy as np, pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bond_pnl.pca import fit_pca
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
