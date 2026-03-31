# Bond PnL Attribution System — Executive Summary

## Overview

This project implements a **complete bond profit-and-loss (PnL) attribution system** for U.S. Treasury fixed-rate bonds. Given a date range, it downloads daily yield-curve data from FRED and decomposes bond returns into interpretable components.

## Core Components

### 1. Core Attribution (Campisi Framework)

Decomposes daily bond PnL into an **exact identity**:

$$\text{Actual PnL} = \underbrace{\text{Market Impact}}_{\text{curve changes}} + \underbrace{\text{Time Impact}}_{\text{passage of time}}$$

Market Impact is further broken down into:
- **Duration** — first-order rate sensitivity (≈ gradient in ML)
- **Convexity** — second-order rate sensitivity (≈ Hessian)
- **Curve Reshape** — non-parallel curve movements

Time Impact is decomposed into:
- **Accrual** — coupon income earned
- **Rolldown** — clean-price change from shortened maturity
- **Funding** — cost of financing the position

**Carry = Accrual + Rolldown − Funding**

### 2. PCA Analysis (Optional Feature 1)

Applies PCA to daily yield-curve changes:
- **PC1** ≈ Level (parallel shift) — typically explains ~90% of variance
- **PC2** ≈ Slope (twist) — ~8%
- **PC3** ≈ Curvature (butterfly) — ~1%

Uses **full repricing** under each PC shock for attribution, not Taylor approximation.

### 3. Bond Ladder Backtest (Optional Feature 2)

Constructs a 5-rung synthetic Treasury ladder (2Y, 5Y, 7Y, 10Y, 30Y):
- **Par bond solver** — numerically finds coupon so `dirty_price = face`
- **Annual rebalancing** — sells all, reinvests equally
- **Attribution** — Income + Rolldown + Rate Movement + Residual (residual ≈ 0)

## Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| Continuous discounting | Mathematically clean, standard in quant finance |
| Cubic spline interpolation | Smooth, arbitrage-free term structure |
| CMT yields as zero-rate proxy | FRED data freely available; documented approximation |
| ACT/ACT (simplified) | `days / 365.25` — small impact for Treasury bonds |
| Gaussian-weighted local shift | Better Duration estimate than parallel shift for non-5Y bonds |

## Model Quality

Across 5 market regimes (2018–2025):
- **Core residual**: 0.04% – 0.67% of actual PnL
- **PCA variance explained**: >99.5%
- **Ladder residual**: ≈ 0.00

## Project Structure

```
bond_pnl/
  yield_curve.py    — FRED data loading, caching, interpolation
  bond.py           — Bond specification, pricing, risk measures
  attribution.py    — Campisi daily attribution engine
  pca.py            — PCA decomposition and attribution
  ladder.py         — Bond ladder backtest
main.py             — CLI entry point
dashboard.py        — Streamlit interactive dashboard
run_multi_year.py   — Cross-regime comparison runner
download_data.py    — Batch FRED data download
tests/              — 56 automated tests (pytest)
docs/EXPLANATION.md — Detailed Chinese documentation
```

## Quick Start

```bash
# Set API key (option 1: env var, option 2: file)
# Linux / macOS:
export FRED_API_KEY=your_key_here
# Windows PowerShell:
#   $env:FRED_API_KEY = "your_key_here"
# Or simply create a file:
#   echo "your_key_here" > fred-api-key.txt

# Install dependencies
pip install -r requirements.txt

# Run full analysis
python main.py

# Launch interactive dashboard
streamlit run dashboard.py

# Multi-year cross-regime comparison
python run_multi_year.py
```

## Dashboard Features

- **Yield Curve**: 3D surface, snapshots, heatmap, daily changes
- **Core Attribution**: Cumulative PnL, waterfall, components, daily distribution
- **PCA Analysis**: Variance explained, factor loadings, PCA attribution waterfall
- **Ladder Backtest**: Portfolio value, attribution, holdings, rebalance log
- **Comparison**: Side-by-side Traditional vs PCA on unified categories

## Test Coverage

56 tests covering:
- Bond pricing (par, off-par, maturity, accrued interest)
- Attribution identity (Market + Time ≡ Actual)
- PCA decomposition (variance, loadings, attribution)
- Ladder (par solver, rebalance, attribution, residual = 0)
- Integration (date snapping, annualization, drawdown, API key, stale-state fingerprint)
- Utility functions (`get_dates`, `snap_to_business_day`, `input_fingerprint`)
