# Bond PnL Attribution System

A Python project for **U.S. Treasury bond PnL attribution**, including:

- **Core attribution** under a Campisi-style framework
- **PCA-based yield curve decomposition**
- **Classic, Withdrawal, and Immunized bond ladder backtests**
- **Multi-year cross-regime analysis**
- **Interactive Streamlit dashboard**

The project is designed to be reproducible from public FRED data and usable from both command-line scripts and an interactive dashboard.

## Features

### 1. Core Bond PnL Attribution
Decomposes bond PnL into:

- **Duration**
- **Convexity**
- **Curve Reshape**
- **Accrual**
- **Rolldown**
- **Funding**
- **Residual**

Identity used in the project:

$$
\text{Actual PnL} = \text{Market Impact} + \text{Time Impact}
$$

where:

- $\text{Market Impact} = \text{Duration} + \text{Convexity} + \text{Curve Reshape}$
- $\text{Time Impact} = \text{Accrual} + \text{Rolldown} - \text{Funding}$

### 2. PCA Analysis
Applies PCA to daily Treasury yield-curve changes and attributes PnL to:

- **PC1**: level shift
- **PC2**: slope change
- **PC3**: curvature change
- **Carry** and **Residual**

The PCA model is fit on centered yield changes; mean curve drift is absorbed into
Residual so the displayed components still add back to Actual PnL.

### 3. Bond Ladder Backtest
Builds synthetic Treasury ladders and evaluates:

- **Classic Roll**: 1Y-5Y hold-to-maturity ladder; matured principal rolls into the longest rung.
- **Withdrawal**: periodic spending schedule using coupon and principal cash flows, with shortfall tracking.
- **Immunized**: duration- or liability-targeted ladder using wider 1Y-30Y candidate rungs.
- Portfolio value, cash balance, attribution, holdings, activity log, cashflow log, and target-match diagnostics.

### 4. Multi-Year Analysis
Runs the same framework across multiple historical market windows for regime comparison.

### 5. Interactive Dashboard
The `Streamlit` dashboard provides:

- **Field**: 3D yield-curve surface and calendar-selected curve snapshot
- **Attribution**: Core, PCA, and Compare views in one workflow
- **Ladder**: single-select switching between Classic Roll, Withdrawal, and Immunized views

## Repository Structure

```text
CS7320/
├─ bond_pnl/
│  ├─ attribution.py
│  ├─ bond.py
│  ├─ ladder.py
│  ├─ pca.py
│  ├─ utils.py
│  └─ yield_curve.py
├─ tests/
├─ dashboard.py
├─ download_data.py
├─ LICENSE
├─ main.py
├─ requirements.txt
└─ run_multi_year.py
```

## Requirements

- Python 3.10+
- A FRED API key

Main dependencies are listed in `requirements.txt`, including:

- `fredapi`
- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `scikit-learn`
- `streamlit`
- `plotly`
- `pytest`

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/xuanmingShang/bond-attribution.git
cd bond-attribution
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure FRED API key

Choose one of the following:

#### Option A: environment variable

**PowerShell**

```powershell
$env:FRED_API_KEY = "your_key_here"
```

**bash / zsh**

```bash
export FRED_API_KEY=your_key_here
```

#### Option B: local file

Create a file named `fred-api-key.txt` in the repository root:

```text
your_key_here
```

> `fred-api-key.txt` is ignored by Git and should never be committed.

## Usage

### Run the main analysis

```bash
python main.py
```

Optional arguments:

```bash
python main.py --start 2023-06-01 --end 2024-06-30
python main.py --maturity-years 10
python main.py --coupon 3.5
python main.py --core-only
```

### Launch the dashboard

```bash
python -m streamlit run dashboard.py
```

### Download more FRED data

```bash
python download_data.py
```

### Run multi-year comparison

```bash
python run_multi_year.py
```

## Outputs

Generated analysis files are typically written to:

- `output/`
- `data/`

These directories are intentionally ignored in Git because they are reproducible artifacts.

## Testing

Run the test suite with:

```bash
python -m pytest tests/ -q
```

Current status:

- **63 tests passing**

## Changelog

### 2026-06-02

- Reorganized the Streamlit dashboard into cleaner Field, Attribution, and Ladder workflows.
- Improved Field snapshot selection with a calendar input and stabilized dashboard controls.
- Removed separate PCA mean PnL handling by folding mean drift into Residual.
- Updated Core cumulative attribution so Actual PnL renders as a solid line.
- Rebuilt Ladder around Classic Roll, Withdrawal, and Immunized strategy modes with switchable UI support.

## License

This project is released under the MIT License. See `LICENSE` for details.
