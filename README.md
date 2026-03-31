# Bond PnL Attribution System

A Python project for **U.S. Treasury bond PnL attribution**, including:

- **Core attribution** under a Campisi-style framework
- **PCA-based yield curve decomposition**
- **Bond ladder backtest**
- **Multi-year cross-regime analysis**
- **Interactive Streamlit dashboard**

This repository was built for a course/project setting and is designed to be both **analytically correct** and **presentation-friendly**.

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

### 3. Bond Ladder Backtest
Builds a synthetic Treasury ladder and evaluates:

- portfolio value evolution
- attribution by component
- rebalance behavior
- residual quality

### 4. Multi-Year Analysis
Runs the same framework across multiple historical market windows for regime comparison.

### 5. Interactive Dashboard
The `Streamlit` dashboard provides:

- yield curve visualization
- attribution charts
- PCA charts
- ladder backtest charts
- side-by-side comparison panels

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
├─ docs/
│  ├─ EXPLANATION.md
│  └─ SUMMARY.md
├─ tests/
├─ dashboard.py
├─ download_data.py
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
git clone <your-github-repo-url>
cd CS7320
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
streamlit run dashboard.py
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

- **56 tests passing**

## Documentation

- `docs/EXPLANATION.md`: detailed Chinese explanation from basics to implementation details
- `docs/SUMMARY.md`: English executive summary
