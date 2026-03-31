"""
run_multi_year.py – Run Bond PnL Attribution across multiple year windows.

Generates a comparative summary showing how attribution components change
across different market regimes (pre-COVID, COVID, hiking cycle, etc.).

Usage:
    python run_multi_year.py                  # Run all predefined windows
    python run_multi_year.py --windows 0 2 4  # Run specific windows by index
"""
from __future__ import annotations
import argparse, sys, os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tabulate import tabulate as _tabulate

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bond_pnl.yield_curve import fetch_yields, YieldCurveHistory
from bond_pnl.bond import BondSpec
from bond_pnl.attribution import run_attribution, attribution_summary
from bond_pnl.pca import fit_pca, pca_attribution
from bond_pnl.ladder import LadderBacktest

OUT = Path(__file__).resolve().parent / "output" / "multi_year"

WINDOWS = [
    ("2018-01-01", "2019-12-31", "2018-2019 (Pre-COVID, late tightening)"),
    ("2020-01-01", "2021-12-31", "2020-2021 (COVID / Zero-Rate)"),
    ("2022-01-01", "2023-12-31", "2022-2023 (Fed Hiking)"),
    ("2023-06-01", "2024-06-30", "2023H2-2024H1 (Default)"),
    ("2024-01-01", "2025-06-30", "2024-2025 (Recent)"),
]


def run_window(start: str, end: str, label: str):
    """Run full analysis for a single window. Returns summary dict."""
    print(f"\n{'='*70}")
    print(f"  {label}  ({start} → {end})")
    print(f"{'='*70}")

    ydf = fetch_yields(start, end)
    ch = YieldCurveHistory(ydf)
    n_dates = len(ydf)
    print(f"  Loaded {n_dates} dates × {len(ydf.columns)} tenors")

    # Use first actual trading day (may not be Jan 1)
    actual_start = str(ydf.index[0].date())
    actual_end = str(ydf.index[-1].date())

    # Bond: 5Y par bond issued at actual start
    sd = pd.Timestamp(actual_start)
    md = sd + pd.DateOffset(years=5)
    py = ch[actual_start].rate(5.0)
    bspec = dict(maturity=md.strftime("%Y-%m-%d"), face=100.0,
                 coupon=round(py, 4), freq=2, day_count="ACT/ACT",
                 issue_date=actual_start)
    bond = BondSpec(**bspec)
    fr = ch[actual_start].rate(0.25)

    # Core attribution
    df = run_attribution(bond, ch, actual_start, actual_end, fr)
    if df.empty:
        print("  ⚠ No data for this window.")
        return None
    s = attribution_summary(df)
    actual = s.get("Actual PnL", 0)
    residual = s.get("Residual", 0)
    res_pct = abs(residual / actual) * 100 if actual else 0
    print(f"  Core: Actual={actual:.4f}  |Res|={res_pct:.2f}%")

    # PCA
    pr = fit_pca(ch, actual_start, actual_end, 3)
    pa = pca_attribution(bond, ch, pr, actual_start, actual_end, fr)
    pca_res = pa.sum().get("Residual", 0) if not pa.empty else None
    var_expl = pr.cumulative_variance[-1] * 100
    print(f"  PCA: {var_expl:.1f}% explained  Res={pca_res:.4f}" if pca_res else "  PCA: N/A")

    # Ladder
    bt = LadderBacktest(ch, actual_start, actual_end, [2, 5, 7, 10, 30], 1_000_000, 12, fr)
    lr = bt.run()
    ts = lr["portfolio_ts"]
    lret = ts["Cumulative Return"].iloc[-1] * 100
    # Use elapsed calendar year fraction for annualization (more accurate than 252/n_dates)
    elapsed_years = (pd.Timestamp(actual_end) - pd.Timestamp(actual_start)).days / 365.25
    elapsed_years = max(elapsed_years, 0.01)  # avoid division by zero
    ladder_ann = ((1 + lret / 100) ** (1 / elapsed_years) - 1) * 100
    print(f"  Ladder: Return={lret:.2f}%  Ann={ladder_ann:.2f}%")

    # Annualized metrics: scale by 1/elapsed_years for fair cross-window comparison
    ann_actual = actual / elapsed_years
    ann_carry = s.get("Carry", 0) / elapsed_years
    ann_market = s.get("Market Impact", 0) / elapsed_years

    return {
        "Window": label,
        "Start": actual_start, "End": actual_end, "Days": n_dates,
        "Coupon": f"{py*100:.2f}%",
        "3M Rate": f"{fr*100:.2f}%",
        "Actual PnL": round(actual, 4),
        "Ann. PnL": round(ann_actual, 4),
        "Carry": round(s.get("Carry", 0), 4),
        "Ann. Carry": round(ann_carry, 4),
        "Duration": round(s.get("Duration", 0), 4),
        "Convexity": round(s.get("Convexity", 0), 4),
        "Curve Reshape": round(s.get("Curve Reshape", 0), 4),
        "Market Impact": round(s.get("Market Impact", 0), 4),
        "Ann. Market": round(ann_market, 4),
        "Time Impact": round(s.get("Time Impact", 0), 4),
        "|Res/Act|": f"{res_pct:.2f}%",
        "PCA Var Expl": f"{var_expl:.1f}%",
        "PCA Residual": round(pca_res, 4) if pca_res is not None else "N/A",
        "Ladder Return": f"{lret:.2f}%",
        "Ladder Ann.": f"{ladder_ann:.2f}%",
        # Raw for plotting
        "_core_df": df,
        "_pca_attr": pa,
        "_ladder_ts": ts,
        "_ladder_attr": lr["attribution"],
    }


def plot_multi_comparison(results: list[dict]):
    """Create comparison charts across all windows."""
    OUT.mkdir(parents=True, exist_ok=True)

    labels = [r["Window"] for r in results]
    n = len(labels)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 1. Annualized PnL (fair comparison across unequal windows)
    ax = axes[0, 0]
    vals = [r["Ann. PnL"] for r in results]
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in vals]
    ax.barh(range(n), vals, color=colors)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=9)
    ax.set_title("Annualized PnL (per year)", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    # 2. Annualized component breakdown
    ax = axes[0, 1]
    comps = ["Ann. Carry", "Ann. Market"]
    comp_labels = ["Carry (ann.)", "Market (ann.)"]
    comp_colors = ["#2ecc71", "#3498db"]
    bottom = np.zeros(n)
    for comp, clr, clabel in zip(comps, comp_colors, comp_labels):
        vals = [r[comp] for r in results]
        ax.barh(range(n), vals, left=bottom, color=clr, label=clabel, alpha=0.85)
        bottom += np.array(vals)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=9)
    ax.set_title("Annualized Components", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")

    # 3. Annualized Market vs Time
    ax = axes[0, 2]
    x = np.arange(n)
    w = 0.35
    market = [r["Ann. Market"] for r in results]
    time_ann = [r["Ann. PnL"] - r["Ann. Market"] for r in results]
    ax.barh(x - w/2, market, w, label="Market (ann.)", color="#3498db", alpha=0.85)
    ax.barh(x + w/2, time_ann, w, label="Time (ann.)", color="#2ecc71", alpha=0.85)
    ax.set_yticks(x); ax.set_yticklabels(labels, fontsize=9)
    ax.set_title("Ann. Market vs Time Impact", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="x")

    # 4. Cumulative PnL overlay
    ax = axes[1, 0]
    for r in results:
        cum = r["_core_df"]["Actual PnL"].cumsum()
        ax.plot(range(len(cum)), cum.values, label=r["Window"], lw=1.5)
    ax.set_title("Cumulative PnL Paths", fontweight="bold")
    ax.set_xlabel("Trading Day"); ax.set_ylabel("Cumulative PnL")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    # 5. Ladder returns
    ax = axes[1, 1]
    for r in results:
        ts = r["_ladder_ts"]
        ax.plot(range(len(ts)), ts["Cumulative Return"].values * 100,
                label=r["Window"], lw=1.5)
    ax.set_title("Ladder Cumulative Return (%)", fontweight="bold")
    ax.set_xlabel("Trading Day"); ax.set_ylabel("Return (%)")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    # 6. Residual quality
    ax = axes[1, 2]
    res_pcts = [float(r["|Res/Act|"].rstrip("%")) for r in results]
    pca_var = [float(r["PCA Var Expl"].rstrip("%")) for r in results]
    ax.barh(range(n), res_pcts, color="#9b59b6", alpha=0.7, label="|Res/Act| %")
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=9)
    ax.set_title("Model Quality", fontweight="bold")
    ax.set_xlabel("|Residual/Actual| %")
    ax2 = ax.twiny()
    ax2.plot(pca_var, range(n), "ro-", label="PCA Var Expl %")
    ax2.set_xlabel("PCA Variance Explained %")
    ax.legend(fontsize=8, loc="lower right")
    ax2.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, axis="x")

    plt.suptitle("Multi-Year Bond PnL Attribution Comparison",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(OUT / "multi_year_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {OUT / 'multi_year_comparison.png'}")


def main():
    ap = argparse.ArgumentParser(description="Multi-Year Bond PnL Analysis")
    ap.add_argument("--windows", nargs="*", type=int, default=None,
                    help="Indices of windows to run (0-based). Default: all.")
    a = ap.parse_args()

    selected = WINDOWS if a.windows is None else [WINDOWS[i] for i in a.windows]
    
    print("=" * 70)
    print("  Multi-Year Bond PnL Attribution Analysis")
    print("=" * 70)
    print(f"  Running {len(selected)} windows")

    results = []
    for start, end, label in selected:
        r = run_window(start, end, label)
        if r:
            results.append(r)

    if not results:
        print("\n  No valid results.")
        return

    # Summary table
    OUT.mkdir(parents=True, exist_ok=True)
    display_cols = [c for c in results[0] if not c.startswith("_")]
    summary_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                                for r in results])
    summary_df.to_csv(OUT / "multi_year_summary.csv", index=False)

    print(f"\n{'='*70}")
    print("  MULTI-YEAR SUMMARY")
    print(f"{'='*70}")
    print(_tabulate(summary_df, headers="keys", tablefmt="simple", showindex=False))

    plot_multi_comparison(results)

    print(f"\n  Output: {OUT}")
    for f in sorted(OUT.glob("*")):
        print(f"    {f.name}")
    print(f"\n{'='*70}")
    print("  ALL COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
