"""
main.py – CLI entry-point for Bond PnL Attribution System.

Usage:
    python main.py                          # full demo (2023-06-01 → 2024-06-30)
    python main.py --start 2023-01-01 --end 2024-12-31
    python main.py --core-only
    python main.py --coupon 4.5 --maturity-years 10
"""
from __future__ import annotations
import argparse, sys, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bond_pnl.yield_curve import fetch_yields, YieldCurveHistory
from bond_pnl.bond import BondSpec
from bond_pnl.attribution import run_attribution, attribution_summary
from bond_pnl.pca import fit_pca, pca_attribution
from bond_pnl.ladder import DEFAULT_RUNGS, LadderBacktest

OUT = Path(__file__).resolve().parent / "output"

def _ensure(): OUT.mkdir(parents=True, exist_ok=True)
def _hdr(t): print("\n" + "="*70 + f"\n  {t}\n" + "="*70)
def _tbl(df, t="", n=20):
    if t: print(f"\n--- {t} ---")
    if isinstance(df, pd.Series): print(df.to_string()); return
    print(tabulate(df.head(n), headers="keys", tablefmt="simple", floatfmt=".6f"))
    if len(df)>n: print(f"  ... ({len(df)} rows, showing {n})")


# ═══════════════════════════════════════════════════════════════════════
#  CORE
# ═══════════════════════════════════════════════════════════════════════
def run_core(ch, start, end, bspec=None):
    _hdr("CORE: Bond PnL Attribution")
    if bspec is None:
        sd = pd.Timestamp(start)
        md = sd + pd.DateOffset(years=5)
        py = ch[start].rate(5.0)
        bspec = dict(maturity=md.strftime("%Y-%m-%d"), face=100.0,
                     coupon=round(py,4), freq=2, day_count="ACT/ACT",
                     issue_date=start)
    bond = BondSpec(**bspec)
    print("\nBond Specification:")
    for k,v in bond.summary().items(): print(f"  {k}: {v}")
    fr = ch[start].rate(0.25)
    print(f"\nFinancing Rate (3M proxy): {fr*100:.2f}%")
    print(f"Window: {start} → {end}")

    df = run_attribution(bond, ch, start, end, fr)
    if df.empty: print("  No data."); return df, bond

    _tbl(df.head(10), "Daily Attribution (first 10)")
    if len(df)>10: _tbl(df.tail(5), "Daily Attribution (last 5)")

    s = attribution_summary(df)
    print(f"\n--- Aggregate Summary ---"); print(s.to_string())

    ta = df["Actual PnL"].sum()
    tm = df["Market Impact"].sum()
    tt = df["Time Impact"].sum()
    tr = df["Residual"].sum()
    print(f"\n  Actual PnL:     {ta:.6f}")
    print(f"  Market Impact:  {tm:.6f}")
    print(f"  Time Impact:    {tt:.6f}")
    print(f"  Market+Time:    {tm+tt:.6f}  (should == Actual)")
    print(f"  Total Residual: {tr:.6f}")
    if ta: print(f"  |Res/Act|:      {abs(tr/ta)*100:.2f}%")

    _ensure()
    df.to_csv(OUT/"core_daily_attribution.csv")
    s.to_csv(OUT/"core_summary.csv")
    _plot_core(df, bond)
    print(f"  Saved: output/core_*.csv, core_attribution.png")
    return df, bond

def _plot_core(df, bond):
    _ensure()
    fig, ax = plt.subplots(2,2,figsize=(14,10))
    cum = df[["Actual PnL"]].cumsum()
    ax[0,0].plot(range(len(cum)), cum.values, "b-", lw=1.5)
    ax[0,0].set_title("Cumulative Actual PnL"); ax[0,0].grid(True, alpha=.3)
    cs = ["Carry","Duration","Convexity","Curve Reshape","Market Impact","Time Impact"]
    avail = [c for c in cs if c in df.columns]
    cc = df[avail].cumsum()
    for c in avail: ax[0,1].plot(range(len(cc)), cc[c], label=c, lw=1)
    ax[0,1].set_title("Cumulative Components"); ax[0,1].legend(fontsize=7); ax[0,1].grid(True, alpha=.3)
    step = max(1, len(df)//60)
    sub = df.iloc[::step]
    ax[1,0].bar(range(len(sub)), sub["Actual PnL"], color="steelblue", alpha=.7)
    ax[1,0].set_title(f"Daily PnL (every {step}d)"); ax[1,0].grid(True, alpha=.3)
    bar_cs = ["Carry","Duration","Convexity","Curve Reshape","Rate Residual","Time Residual"]
    bar_avail = [c for c in bar_cs if c in df.columns]
    tots = df[bar_avail].sum()
    colors = ["#2ecc71","#3498db","#e74c3c","#f39c12","#9b59b6","#95a5a6"]
    ax[1,1].bar(range(len(tots)), tots.values, color=colors[:len(tots)])
    ax[1,1].set_xticks(range(len(tots))); ax[1,1].set_xticklabels(tots.index, rotation=45, ha="right")
    ax[1,1].set_title("Total Breakdown"); ax[1,1].grid(True, alpha=.3)
    plt.suptitle(f"Bond PnL – {bond.coupon*100:.1f}% {bond.maturity}", fontsize=13, weight="bold")
    plt.tight_layout(); plt.savefig(OUT/"core_attribution.png", dpi=150, bbox_inches="tight"); plt.close()


# ═══════════════════════════════════════════════════════════════════════
#  PCA
# ═══════════════════════════════════════════════════════════════════════
def run_pca_feature(ch, bond, start, end):
    _hdr("OPTIONAL 1: PCA Analysis & Attribution")
    pr = fit_pca(ch, start, end, 3)
    _tbl(pr.summary(), "Variance Explained")
    _tbl(pr.loadings.T, "Factor Loadings")
    _tbl(pr.scores.head(10), "Factor Scores (first 10)")
    fr = ch[start].rate(0.25)
    pa = pca_attribution(bond, ch, pr, start, end, fr)
    if not pa.empty:
        _tbl(pa.head(10), "PCA Attribution (first 10)")
        print("\n--- PCA Summary ---"); print(pa.sum().rename("Total").to_string())
    _ensure()
    pr.loadings.to_csv(OUT/"pca_loadings.csv")
    pr.scores.to_csv(OUT/"pca_scores.csv")
    pr.summary().to_csv(OUT/"pca_summary.csv", index=False)
    if not pa.empty: pa.to_csv(OUT/"pca_attribution.csv")
    _plot_pca(pr, pa)
    print(f"  Saved: output/pca_*.csv, pca_analysis.png")
    return pr, pa

def _plot_pca(pr, pa):
    _ensure()
    fig, ax = plt.subplots(2,2,figsize=(14,10))
    x = range(pr.n_components)
    ax[0,0].bar(x, pr.explained_variance_ratio*100, color="steelblue")
    ax[0,0].plot(x, pr.cumulative_variance*100, "ro-")
    ax[0,0].set_xticks(list(x)); ax[0,0].set_xticklabels([f"PC{i+1}" for i in x])
    ax[0,0].set_title("Variance Explained (%)"); ax[0,0].grid(True, alpha=.3)
    ts = pr.loadings.columns
    for i in range(pr.n_components):
        ax[0,1].plot(range(len(ts)), pr.loadings.iloc[i], "o-",
                     label=["Level","Slope","Curvature"][i] if i<3 else f"PC{i+1}")
    ax[0,1].set_xticks(range(len(ts))); ax[0,1].set_xticklabels(ts, rotation=45, fontsize=7)
    ax[0,1].set_title("Factor Loadings"); ax[0,1].legend(); ax[0,1].grid(True, alpha=.3)
    for c in pr.scores.columns:
        ax[1,0].plot(range(len(pr.scores)), pr.scores[c], lw=.8, label=c)
    ax[1,0].set_title("Factor Scores"); ax[1,0].legend(); ax[1,0].grid(True, alpha=.3)
    if not pa.empty:
        pcc = [c for c in pa.columns if (c.startswith("PC") or c == "Mean PnL") and "Total" not in c]
        cum = pa[pcc+["Carry","Residual"]].cumsum()
        for c in cum.columns: ax[1,1].plot(range(len(cum)), cum[c], lw=1, label=c)
        ax[1,1].set_title("Cumul. PCA Attribution"); ax[1,1].legend(fontsize=7); ax[1,1].grid(True, alpha=.3)
    plt.suptitle("PCA Analysis", fontsize=13, weight="bold")
    plt.tight_layout(); plt.savefig(OUT/"pca_analysis.png", dpi=150, bbox_inches="tight"); plt.close()


# ═══════════════════════════════════════════════════════════════════════
#  LADDER
# ═══════════════════════════════════════════════════════════════════════
def run_ladder(ch, start, end):
    _hdr("OPTIONAL 2: Bond Ladder Backtest")
    bt = LadderBacktest(ch, start, end, DEFAULT_RUNGS.copy(), 1_000_000.0, 12,
                        ch[start].rate(0.25))
    rs = bt.run()
    ts = rs["portfolio_ts"]
    _tbl(ts.head(10), "Portfolio (first 10)")
    _tbl(ts.tail(5), "Portfolio (last 5)")
    fv = ts["Portfolio Value"].iloc[-1]
    tr = ts["Cumulative Return"].iloc[-1]
    elapsed_years = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    elapsed_years = max(elapsed_years, 0.01)
    ar = (1+tr)**(1/elapsed_years)-1
    print(f"\n--- Period Summary ---")
    print(f"  Initial: $1,000,000  Final: ${fv:,.2f}")
    print(f"  Return: {tr*100:.2f}%  Annualised: {ar*100:.2f}%")
    at = rs["attribution"]
    print("\n--- Attribution Summary ---")
    print(at[["Total PnL","Income","Rolldown","Rate Movement","Residual"]].sum().to_string())
    if not rs["holdings"].empty: _tbl(rs["holdings"], "Holdings")
    _tbl(rs["rebalance_log"].head(20), f"Rebalance Log ({len(rs['rebalance_log'])} records)")
    _ensure()
    ts.to_csv(OUT/"ladder_portfolio.csv"); at.to_csv(OUT/"ladder_attribution.csv")
    rs["rebalance_log"].to_csv(OUT/"ladder_rebalance.csv", index=False)
    rs["holdings"].to_csv(OUT/"ladder_holdings.csv", index=False)
    _plot_ladder(ts, at)
    print(f"  Saved: output/ladder_*.csv, ladder_backtest.png")
    return rs

def _plot_ladder(ts, at):
    _ensure()
    fig, ax = plt.subplots(2,2,figsize=(14,10))
    ax[0,0].plot(range(len(ts)), ts["Portfolio Value"], "b-", lw=1.5)
    ax[0,0].set_title("Portfolio Value ($)"); ax[0,0].grid(True, alpha=.3)
    ax[0,1].plot(range(len(ts)), ts["Cumulative Return"]*100, "g-", lw=1.5)
    ax[0,1].set_title("Cumulative Return (%)"); ax[0,1].grid(True, alpha=.3)
    acs = ["Income","Rolldown","Rate Movement","Residual"]
    cum = at[acs].cumsum()
    for c in acs: ax[1,0].plot(range(len(cum)), cum[c], lw=1, label=c)
    ax[1,0].set_title("Cumul. Attribution"); ax[1,0].legend(); ax[1,0].grid(True, alpha=.3)
    tots = at[acs].sum()
    ax[1,1].bar(range(len(tots)), tots.values, color=["#2ecc71","#3498db","#e74c3c","#95a5a6"])
    ax[1,1].set_xticks(range(len(tots))); ax[1,1].set_xticklabels(tots.index, rotation=30, ha="right")
    ax[1,1].set_title("Total Attribution ($)"); ax[1,1].grid(True, alpha=.3)
    plt.suptitle("Bond Ladder Backtest", fontsize=13, weight="bold")
    plt.tight_layout(); plt.savefig(OUT/"ladder_backtest.png", dpi=150, bbox_inches="tight"); plt.close()


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════
def should_customize_bond(coupon, maturity_years: int) -> bool:
    """Return True if the CLI args request a non-default bond specification."""
    return coupon is not None or maturity_years != 5


def main():
    ap = argparse.ArgumentParser(description="Bond PnL Attribution System")
    ap.add_argument("--start", default="2023-06-01")
    ap.add_argument("--end",   default="2024-06-30")
    ap.add_argument("--core-only", action="store_true")
    ap.add_argument("--coupon", type=float, default=None, help="Coupon rate in percent")
    ap.add_argument("--maturity-years", type=int, default=5)
    a = ap.parse_args()

    _hdr("Bond PnL Attribution System")
    print(f"  Window: {a.start} → {a.end}")

    print("\n[1] Loading yields from FRED ...")
    ydf = fetch_yields(a.start, a.end)
    ch  = YieldCurveHistory(ydf)
    print(f"  {len(ydf)} dates × {len(ydf.columns)} tenors")

    # Snap start/end to actual business days in the data
    avail = ch.business_dates(a.start, a.end)
    if len(avail) < 2:
        print("  ERROR: Need at least 2 business dates in window."); return
    start_snap = str(avail[0].date()) if hasattr(avail[0], 'date') else str(avail[0])
    end_snap = str(avail[-1].date()) if hasattr(avail[-1], 'date') else str(avail[-1])
    if start_snap != a.start or end_snap != a.end:
        print(f"  Dates snapped to business days: {start_snap} → {end_snap}")

    print("\n[2] Core Attribution ...")
    bs = None
    if should_customize_bond(a.coupon, a.maturity_years):
        sd = pd.Timestamp(start_snap)
        md = sd + pd.DateOffset(years=a.maturity_years)
        if a.coupon is not None:
            cpn = a.coupon / 100.0
        else:
            cpn = round(ch[start_snap].rate(float(a.maturity_years)), 4)
        bs = dict(maturity=md.strftime("%Y-%m-%d"), face=100.0,
                  coupon=cpn, freq=2, day_count="ACT/ACT",
                  issue_date=start_snap)
    cdf, bond = run_core(ch, start_snap, end_snap, bs)
    if a.core_only: _hdr("DONE"); return

    print("\n[3] PCA ...")
    pr, pa = run_pca_feature(ch, bond, start_snap, end_snap)

    print("\n[4] Ladder ...")
    lr = run_ladder(ch, start_snap, end_snap)

    # comparison
    if not cdf.empty and not pa.empty:
        _hdr("Traditional vs PCA Comparison")
        trad = cdf[["Carry","Duration","Convexity","Curve Reshape","Residual",
                     "Market Impact","Time Impact","Actual PnL"]].sum()
        pca_s = pa.sum()
        # Build side-by-side comparison
        comp = pd.DataFrame({"Traditional": trad})
        pca_df = pd.DataFrame({"PCA": pca_s})
        comp = pd.concat([comp, pca_df], axis=1)
        comp.to_csv(OUT/"comparison.csv")
        print("Traditional:\n" + trad.to_string())
        print("\nPCA-based:\n" + pca_s.to_string())
        print(f"\n  Saved: output/comparison.csv")

    _hdr("ALL COMPLETE")
    print(f"  Output: {OUT}")
    for f in sorted(OUT.glob("*")): print(f"    {f.name}")

if __name__ == "__main__":
    main()
