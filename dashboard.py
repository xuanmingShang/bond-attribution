"""
dashboard.py – Streamlit Interactive Dashboard for Bond PnL Attribution System.

Usage:
    streamlit run dashboard.py
"""
from __future__ import annotations
import sys, os
from io import StringIO
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from bond_pnl.yield_curve import fetch_yields, YieldCurveHistory, TENOR_LABELS, TENOR_YEARS
from bond_pnl.bond import BondSpec
from bond_pnl.attribution import run_attribution, attribution_summary
from bond_pnl.pca import fit_pca, pca_attribution
from bond_pnl.ladder import LadderBacktest
from bond_pnl.utils import get_dates, snap_to_business_day, input_fingerprint

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Page Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="Bond PnL Attribution System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for professional look ──────────────────────────────────
st.markdown("""
<style>
    /* Overall page */
    .block-container { padding-top: 1rem; }
    
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea11, #764ba211);
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetric"] label {
        font-size: 0.85rem !important;
        color: #555 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #1a1a2e, #16213e);
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        margin: 10px 0;
        font-size: 1.1rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    /* Info boxes */
    .info-box {
        background: #f0f7ff;
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        font-size: 0.9rem;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e, #16213e);
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #e0e0e0;
    }
    
    /* Tables */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Color Palette
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COLORS = {
    "primary": "#3b82f6",
    "success": "#10b981",
    "danger": "#ef4444",
    "warning": "#f59e0b",
    "purple": "#8b5cf6",
    "gray": "#6b7280",
    "carry": "#10b981",
    "duration": "#3b82f6",
    "convexity": "#ef4444",
    "reshape": "#f59e0b",
    "residual": "#6b7280",
    "rolldown": "#8b5cf6",
    "funding": "#ec4899",
    "accrual": "#14b8a6",
    "market": "#3b82f6",
    "time": "#10b981",
    "pc1": "#3b82f6",
    "pc2": "#ef4444",
    "pc3": "#f59e0b",
}

WATERFALL_COLORS = ["#10b981", "#3b82f6", "#ef4444", "#f59e0b", "#8b5cf6", "#6b7280"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Helper: Section header
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def section_header(text: str):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)

def info_box(text: str):
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Sidebar Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown("## 📊 Bond PnL Attribution")
    st.markdown("---")
    
    st.markdown("### ⚙️ Analysis Window")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", pd.Timestamp("2023-06-01"))
    with col2:
        end_date = st.date_input("End", pd.Timestamp("2024-06-30"))
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    st.markdown("### 🏦 Bond Specification")
    maturity_years = st.slider("Maturity (years)", 1, 30, 5)
    coupon_pct = st.number_input("Coupon Rate (%)", 0.0, 15.0, 0.0,
                                  step=0.25, help="0 = auto (par coupon)")
    face_value = st.number_input("Face Value ($)", 10.0, 10000.0, 100.0, step=10.0)
    freq = st.selectbox("Coupon Frequency", [2, 1, 4], index=0,
                        format_func=lambda x: {1: "Annual", 2: "Semi-annual", 4: "Quarterly"}[x])
    
    st.markdown("### 📑 Features")
    run_pca = st.checkbox("PCA Analysis", value=True)
    run_ladder_opt = st.checkbox("Ladder Backtest", value=True)
    
    if run_ladder_opt:
        st.markdown("### 🪜 Ladder Settings")
        ladder_capital = st.number_input("Capital ($)", 100_000, 100_000_000,
                                         1_000_000, step=100_000)
        ladder_rebal = st.selectbox("Rebalance (months)", [6, 12, 24], index=1)
    
    st.markdown("---")
    run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)


def _snap_to_business_day(ydf: pd.DataFrame, date_str: str,
                          direction: str = "forward") -> str:
    """Delegate to shared util – kept as thin wrapper for back-compat."""
    return snap_to_business_day(ydf, date_str, direction)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Title
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
# 📊 Bond PnL Attribution System
**U.S. Treasury Fixed-Rate Bond Analytics** &nbsp;|&nbsp; Core Attribution · PCA Decomposition · Ladder Backtest
""")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Data Loading & Computation (cached)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_data(show_spinner=False)
def load_yields(start: str, end: str):
    return fetch_yields(start, end)

@st.cache_data(show_spinner=False)
def compute_core(ydf_json: str, start: str, end: str,
                 maturity_years: int, coupon: float, face: float, freq: int):
    ydf = pd.read_json(StringIO(ydf_json))
    ch = YieldCurveHistory(ydf)
    sd = pd.Timestamp(start)
    md = sd + pd.DateOffset(years=maturity_years)
    if coupon <= 0:
        coupon = round(ch[start].rate(float(maturity_years)), 4)
    else:
        coupon = coupon / 100.0
    bspec = dict(maturity=md.strftime("%Y-%m-%d"), face=face,
                 coupon=coupon, freq=freq, day_count="ACT/ACT",
                 issue_date=start)
    bond = BondSpec(**bspec)
    fr = ch[start].rate(0.25)
    df = run_attribution(bond, ch, start, end, fr)
    s = attribution_summary(df)
    return df, s, bond.summary(), coupon, fr

@st.cache_data(show_spinner=False)
def compute_pca(ydf_json: str, start: str, end: str,
                bond_spec_json: str, coupon: float, face: float, freq: int):
    ydf = pd.read_json(StringIO(ydf_json))
    ch = YieldCurveHistory(ydf)
    sd = pd.Timestamp(start)
    # Reconstruct bond for PCA attribution
    bond_info = pd.read_json(StringIO(bond_spec_json), typ="series")
    bond = BondSpec(
        maturity=bond_info["Maturity"],
        face=face,
        coupon=coupon,
        freq=freq,
        day_count="ACT/ACT",
        issue_date=start,
    )
    pr = fit_pca(ch, start, end, 3)
    fr = ch[start].rate(0.25)
    pa = pca_attribution(bond, ch, pr, start, end, fr)
    return pr, pa

@st.cache_data(show_spinner=False)
def compute_ladder(ydf_json: str, start: str, end: str,
                   capital: float, rebal_months: int):
    ydf = pd.read_json(StringIO(ydf_json))
    ch = YieldCurveHistory(ydf)
    fr = ch[start].rate(0.25)
    bt = LadderBacktest(ch, start, end, [2, 5, 7, 10, 30], capital, rebal_months, fr)
    return bt.run()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Plot builders
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _plotly_defaults(fig, h=500):
    fig.update_layout(
        height=h,
        template="plotly_white",
        font=dict(family="Inter, sans-serif", size=12),
        margin=dict(l=50, r=30, t=50, b=50),
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        hovermode="x unified",
    )
    return fig


def plot_yield_curve_3d(ydf: pd.DataFrame):
    """Interactive 3D yield-curve surface."""
    # Subsample for performance
    step = max(1, len(ydf) // 80)
    sub = ydf.iloc[::step]
    date_labels = [str(d.date()) for d in sub.index]
    
    fig = go.Figure(data=[go.Surface(
        z=sub.values,
        x=TENOR_YEARS,
        y=list(range(len(sub))),
        colorscale="Viridis",
        opacity=0.9,
        showscale=True,
        colorbar=dict(title="Yield (%)", len=0.6),
        hovertemplate="Tenor: %{x}Y<br>Date: %{customdata}<br>Yield: %{z:.2f}%<extra></extra>",
        customdata=np.array([[d]*len(TENOR_YEARS) for d in date_labels]),
    )])
    fig.update_layout(
        scene=dict(
            xaxis_title="Tenor (years)",
            yaxis_title="Date",
            zaxis_title="Yield (%)",
            yaxis=dict(
                tickvals=list(range(0, len(sub), max(1, len(sub)//6))),
                ticktext=[date_labels[i] for i in range(0, len(sub), max(1, len(sub)//6))],
            ),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0)),
        ),
        height=550,
        margin=dict(l=0, r=0, t=30, b=0),
        template="plotly_white",
    )
    return fig


def plot_yield_curves_snapshot(ydf: pd.DataFrame, dates: list[str]):
    """Overlay yield curves for selected dates."""
    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    for i, d in enumerate(dates):
        ts = pd.Timestamp(d)
        idx = ydf.index.get_indexer([ts], method="nearest")[0]
        row = ydf.iloc[idx]
        fig.add_trace(go.Scatter(
            x=TENOR_YEARS, y=row.values,
            mode="lines+markers",
            name=str(ydf.index[idx].date()),
            line=dict(width=2.5, color=colors[i % len(colors)]),
            marker=dict(size=6),
        ))
    fig.update_layout(
        xaxis_title="Tenor (years)", yaxis_title="Yield (%)",
        title="Yield Curve Snapshots",
    )
    return _plotly_defaults(fig, 420)


def plot_cumulative_pnl(df: pd.DataFrame):
    """Cumulative PnL with shaded area."""
    cum = df["Actual PnL"].cumsum()
    dates = df["Date"] if "Date" in df.columns else df.index
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=cum.values,
        fill="tozeroy",
        fillcolor="rgba(59,130,246,0.12)",
        line=dict(color=COLORS["primary"], width=2.5),
        name="Cumulative PnL",
    ))
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Cumulative PnL",
        title="Cumulative PnL Over Time",
    )
    return _plotly_defaults(fig, 400)


def plot_waterfall(summary: pd.Series):
    """Attribution waterfall chart."""
    components = ["Carry", "Duration", "Convexity", "Curve Reshape", "Residual"]
    vals = [summary.get(c, 0) for c in components]
    actual = summary.get("Actual PnL", sum(vals))
    
    fig = go.Figure(go.Waterfall(
        name="PnL",
        orientation="v",
        measure=["relative"] * len(components) + ["total"],
        x=components + ["Actual PnL"],
        y=vals + [actual],
        textposition="outside",
        text=[f"{v:+.4f}" for v in vals] + [f"{actual:.4f}"],
        connector=dict(line=dict(color="#aaa", width=1)),
        increasing=dict(marker=dict(color=COLORS["success"])),
        decreasing=dict(marker=dict(color=COLORS["danger"])),
        totals=dict(marker=dict(color=COLORS["primary"])),
    ))
    fig.update_layout(
        title="PnL Attribution Waterfall",
        yaxis_title="PnL",
        showlegend=False,
    )
    return _plotly_defaults(fig, 450)


def plot_market_vs_time(summary: pd.Series):
    """Market vs Time impact grouped bar chart (signed values)."""
    market = summary.get("Market Impact", 0)
    time_i = summary.get("Time Impact", 0)
    residual = summary.get("Residual", 0)
    
    categories = ["Market Impact", "Time Impact", "Residual"]
    values = [market, time_i, residual]
    bar_colors = [COLORS["market"], COLORS["time"], COLORS["residual"]]
    
    fig = go.Figure(go.Bar(
        x=categories, y=values,
        marker_color=bar_colors,
        text=[f"{v:+.4f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="PnL Decomposition (Market vs Time)",
        yaxis_title="PnL",
        showlegend=False,
    )
    return _plotly_defaults(fig, 400)


def plot_cumulative_components(df: pd.DataFrame):
    """Cumulative attribution components."""
    components = ["Carry", "Duration", "Convexity", "Curve Reshape",
                  "Market Impact", "Time Impact", "Actual PnL"]
    avail = [c for c in components if c in df.columns]
    cum = df[avail].cumsum()
    dates = df["Date"] if "Date" in df.columns else df.index
    
    fig = go.Figure()
    comp_colors = {
        "Carry": COLORS["carry"], "Duration": COLORS["duration"],
        "Convexity": COLORS["convexity"], "Curve Reshape": COLORS["reshape"],
        "Market Impact": COLORS["market"], "Time Impact": COLORS["time"],
        "Actual PnL": "#111",
    }
    for c in avail:
        dash = "solid" if c != "Actual PnL" else "dash"
        w = 1.5 if c != "Actual PnL" else 3
        fig.add_trace(go.Scatter(
            x=dates, y=cum[c].values,
            mode="lines",
            name=c,
            line=dict(color=comp_colors.get(c, "#999"), width=w, dash=dash),
        ))
    fig.update_layout(
        title="Cumulative Components",
        xaxis_title="Date", yaxis_title="Cumulative",
    )
    return _plotly_defaults(fig, 420)


def plot_daily_bar(df: pd.DataFrame):
    """Daily PnL bar chart."""
    dates = df["Date"] if "Date" in df.columns else df.index
    colors = [COLORS["success"] if v >= 0 else COLORS["danger"]
              for v in df["Actual PnL"]]
    fig = go.Figure(go.Bar(
        x=dates, y=df["Actual PnL"].values,
        marker_color=colors, name="Daily PnL",
    ))
    fig.update_layout(
        title="Daily PnL", xaxis_title="Date", yaxis_title="PnL",
    )
    return _plotly_defaults(fig, 350)


def plot_pca_variance(pr):
    """PCA variance explained bar + cumulative line."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    labels = [f"PC{i+1}" for i in range(pr.n_components)]
    fig.add_trace(go.Bar(
        x=labels,
        y=pr.explained_variance_ratio * 100,
        name="Variance (%)",
        marker_color=[COLORS["pc1"], COLORS["pc2"], COLORS["pc3"]],
        text=[f"{v:.1f}%" for v in pr.explained_variance_ratio * 100],
        textposition="outside",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=labels,
        y=pr.cumulative_variance * 100,
        name="Cumulative",
        mode="lines+markers",
        line=dict(color="#111", width=2.5),
        marker=dict(size=10, symbol="diamond"),
    ), secondary_y=True)
    fig.update_yaxes(title_text="Individual (%)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative (%)", secondary_y=True,
                     range=[0, 105])
    fig.update_layout(title="PCA Variance Explained")
    return _plotly_defaults(fig, 400)


def plot_pca_loadings(pr):
    """PCA factor loadings heatmap + lines."""
    fig = go.Figure()
    names = ["Level (PC1)", "Slope (PC2)", "Curvature (PC3)"]
    colors_pc = [COLORS["pc1"], COLORS["pc2"], COLORS["pc3"]]
    tenors = list(pr.loadings.columns)
    for i in range(min(3, pr.n_components)):
        fig.add_trace(go.Scatter(
            x=tenors,
            y=pr.loadings.iloc[i].values,
            mode="lines+markers",
            name=names[i] if i < 3 else f"PC{i+1}",
            line=dict(color=colors_pc[i], width=2.5),
            marker=dict(size=8),
        ))
    fig.add_hline(y=0, line_dash="dot", line_color="#aaa")
    fig.update_layout(
        title="Factor Loadings by Tenor",
        xaxis_title="Tenor", yaxis_title="Loading",
    )
    return _plotly_defaults(fig, 400)


def plot_pca_loadings_heatmap(pr):
    """Heatmap version of loadings."""
    names = ["PC1 (Level)", "PC2 (Slope)", "PC3 (Curvature)"]
    fig = go.Figure(go.Heatmap(
        z=pr.loadings.values[:3],
        x=list(pr.loadings.columns),
        y=names[:pr.n_components],
        colorscale="RdBu_r",
        zmid=0,
        text=np.round(pr.loadings.values[:3], 3),
        texttemplate="%{text}",
        textfont=dict(size=11),
    ))
    fig.update_layout(title="Factor Loadings Heatmap", height=280,
                      margin=dict(l=80, r=30, t=50, b=50))
    return fig


def plot_pca_attribution_cum(pa: pd.DataFrame):
    """Cumulative PCA attribution."""
    cols = [c for c in pa.columns if c not in ["Date", "Actual PnL"]]
    cum = pa[cols].cumsum()
    dates = pa["Date"] if "Date" in pa.columns else pa.index
    fig = go.Figure()
    col_colors = {
        "Carry": COLORS["carry"], "Mean PnL": "#6b7280",
        "PC1 PnL": COLORS["pc1"], "PC2 PnL": COLORS["pc2"],
        "PC3 PnL": COLORS["pc3"], "PC Total": "#111",
        "Residual": COLORS["residual"],
    }
    for c in cols:
        fig.add_trace(go.Scatter(
            x=dates, y=cum[c].values,
            mode="lines", name=c,
            line=dict(color=col_colors.get(c, "#999"), width=1.8),
        ))
    fig.update_layout(
        title="Cumulative PCA Attribution",
        xaxis_title="Date", yaxis_title="Cumulative",
    )
    return _plotly_defaults(fig, 420)


def _get_dates(df: pd.DataFrame):
    """Delegate to shared util – kept as thin wrapper for back-compat."""
    return get_dates(df)


def plot_ladder_portfolio(ts: pd.DataFrame):
    """Ladder portfolio value over time."""
    dates = _get_dates(ts)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4],
                        subplot_titles=["Portfolio Value ($)", "Cumulative Return (%)"])
    fig.add_trace(go.Scatter(
        x=dates, y=ts["Portfolio Value"],
        fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
        line=dict(color=COLORS["primary"], width=2),
        name="Value",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=ts["Cumulative Return"] * 100,
        fill="tozeroy",
        fillcolor="rgba(16,185,129,0.08)",
        line=dict(color=COLORS["success"], width=2),
        name="Return (%)",
    ), row=2, col=1)
    fig.update_layout(showlegend=False)
    return _plotly_defaults(fig, 500)


def plot_ladder_attribution(at: pd.DataFrame):
    """Ladder attribution stacked area."""
    cols = ["Income", "Rolldown", "Rate Movement", "Residual"]
    cum = at[cols].cumsum()
    dates = at.index if "Date" not in at.columns else at["Date"]
    
    fig = go.Figure()
    col_colors = {"Income": COLORS["carry"], "Rolldown": COLORS["purple"],
                  "Rate Movement": COLORS["danger"], "Residual": COLORS["gray"]}
    for c in cols:
        fig.add_trace(go.Scatter(
            x=dates, y=cum[c].values,
            mode="lines", name=c,
            line=dict(color=col_colors[c], width=2),
            stackgroup="one" if c != "Residual" else None,
        ))
    fig.update_layout(
        title="Cumulative Ladder Attribution ($)",
        xaxis_title="Date", yaxis_title="Cumulative ($)",
    )
    return _plotly_defaults(fig, 400)


def plot_ladder_waterfall(at: pd.DataFrame):
    """Ladder total attribution waterfall."""
    cols = ["Income", "Rolldown", "Rate Movement", "Residual"]
    vals = [at[c].sum() for c in cols]
    total = sum(vals)
    
    fig = go.Figure(go.Waterfall(
        measure=["relative"] * len(cols) + ["total"],
        x=cols + ["Total PnL"],
        y=vals + [total],
        text=[f"${v:+,.0f}" for v in vals] + [f"${total:,.0f}"],
        textposition="outside",
        connector=dict(line=dict(color="#aaa")),
        increasing=dict(marker=dict(color=COLORS["success"])),
        decreasing=dict(marker=dict(color=COLORS["danger"])),
        totals=dict(marker=dict(color=COLORS["primary"])),
    ))
    fig.update_layout(title="Ladder PnL Waterfall", showlegend=False)
    return _plotly_defaults(fig, 400)


def plot_comparison_bar(trad_summary: pd.Series, pca_summary: pd.Series):
    """Side-by-side grouped bar comparing Traditional vs PCA on unified categories."""
    # Unified categories: Carry, Market/PC Total, Residual, Actual PnL
    categories = ["Carry", "Market/Rate Total", "Residual", "Actual PnL"]
    trad_vals = [
        trad_summary.get("Carry", 0),
        trad_summary.get("Market Impact", 0),
        trad_summary.get("Residual", 0),
        trad_summary.get("Actual PnL", 0),
    ]
    pca_vals = [
        pca_summary.get("Carry", 0),
        pca_summary.get("PC Total", 0),
        pca_summary.get("Residual", 0),
        pca_summary.get("Actual PnL", 0),
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Traditional (Campisi)",
        x=categories, y=trad_vals,
        marker_color=COLORS["primary"], opacity=0.85,
        text=[f"{v:+.4f}" for v in trad_vals], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="PCA-Based",
        x=categories, y=pca_vals,
        marker_color=COLORS["danger"], opacity=0.85,
        text=[f"{v:+.4f}" for v in pca_vals], textposition="outside",
    ))
    fig.update_layout(
        title="Traditional vs PCA Attribution (Unified Categories)",
        barmode="group", yaxis_title="PnL",
    )
    return _plotly_defaults(fig, 420)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main flow
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_current_fp = input_fingerprint(
    start_str, end_str, maturity_years, coupon_pct,
    face_value, freq, run_pca, run_ladder_opt,
    ladder_capital if run_ladder_opt else 0,
    ladder_rebal if run_ladder_opt else 0,
)

# Detect stale results: sidebar changed after last Run
_has_stale = (
    "results" in st.session_state
    and st.session_state.get("_input_fp") != _current_fp
    and not run_btn
)
if _has_stale:
    st.warning(
        "⚠️ Sidebar parameters have changed since the last run. "
        "Click **Run Analysis** to update the results."
    )

if run_btn or "results" in st.session_state:
    if run_btn:
        # Input validation
        if start_date >= end_date:
            st.error("Start date must be before end date.")
            st.stop()
        if (end_date - start_date).days < 30:
            st.warning("Window is very short (< 30 days). Results may be unreliable.")
        
        try:
            with st.spinner("Loading yields from FRED..."):
                ydf = load_yields(start_str, end_str)
        except FileNotFoundError as e:
            st.error(f"FRED API key not found. Set `FRED_API_KEY` env var or create `fred-api-key.txt`. ({e})")
            st.stop()
        except Exception as e:
            st.error(f"Failed to load yield data: {e}. Check your internet connection and API key.")
            st.stop()
        
        if len(ydf) < 10:
            st.error(f"Only {len(ydf)} trading days available. Need at least 10.")
            st.stop()
        
        # Snap dates: start forward, end backward
        start_str = _snap_to_business_day(ydf, start_str, "forward")
        end_str = _snap_to_business_day(ydf, end_str, "backward")
        
        ydf_json = ydf.to_json()
        
        with st.spinner("Computing Core Attribution..."):
            core_df, core_summary, bond_info, coupon_used, fin_rate = compute_core(
                ydf_json, start_str, end_str,
                maturity_years, coupon_pct, face_value, freq)
        
        pca_result = pca_attr = None
        if run_pca:
            with st.spinner("Computing PCA Analysis..."):
                binfo_json = pd.Series(bond_info).to_json()
                pca_result, pca_attr = compute_pca(
                    ydf_json, start_str, end_str,
                    binfo_json, coupon_used, face_value, freq)
        
        ladder_result = None
        if run_ladder_opt:
            with st.spinner("Running Ladder Backtest..."):
                ladder_result = compute_ladder(
                    ydf_json, start_str, end_str,
                    ladder_capital, ladder_rebal)
        
        st.session_state["results"] = {
            "ydf": ydf, "core_df": core_df, "core_summary": core_summary,
            "bond_info": bond_info, "coupon_used": coupon_used,
            "fin_rate": fin_rate,
            "pca_result": pca_result, "pca_attr": pca_attr,
            "ladder_result": ladder_result,
            "start": start_str, "end": end_str,
        }
        st.session_state["_input_fp"] = _current_fp
    
    r = st.session_state["results"]
    ydf = r["ydf"]
    core_df = r["core_df"]
    core_summary = r["core_summary"]
    bond_info = r["bond_info"]
    pca_result = r["pca_result"]
    pca_attr = r["pca_attr"]
    ladder_result = r["ladder_result"]
    
    # ── Overview Metrics ──────────────────────────────────────────────
    section_header("📋 Overview")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    actual = core_summary.get("Actual PnL", 0)
    market = core_summary.get("Market Impact", 0)
    time_i = core_summary.get("Time Impact", 0)
    residual = core_summary.get("Residual", 0)
    res_pct = abs(residual / actual) * 100 if actual else 0
    
    m1.metric("Actual PnL", f"{actual:.4f}", delta=f"{'Gain' if actual >= 0 else 'Loss'}")
    m2.metric("Market Impact", f"{market:.4f}")
    m3.metric("Time Impact", f"{time_i:.4f}")
    m4.metric("Residual", f"{residual:.6f}")
    m5.metric("|Res/Act|", f"{res_pct:.2f}%", delta="✓" if res_pct < 1 else "⚠")
    m6.metric("Trading Days", f"{len(core_df)}")
    
    st.markdown("")
    
    # ── Bond Info ─────────────────────────────────────────────────────
    with st.expander("🏦 Bond Specification & Assumptions", expanded=False):
        bc1, bc2 = st.columns(2)
        with bc1:
            for k, v in bond_info.items():
                st.markdown(f"**{k}:** {v}")
        with bc2:
            st.markdown(f"**Financing Rate (3M):** {r['fin_rate']*100:.2f}%")
            st.markdown(f"**Analysis Window:** {r['start']} → {r['end']}")
            st.markdown(f"**Day Count:** ACT/ACT")
            st.markdown(f"**Compounding:** Continuous")
            info_box("Model uses continuous discounting and cubic-spline interpolated CMT yields as zero-rate proxy.")
    
    # ── Tabs ──────────────────────────────────────────────────────────
    tab_names = ["🔍 Yield Curve", "📊 Core Attribution"]
    if pca_result: tab_names.append("🧮 PCA Analysis")
    if ladder_result: tab_names.append("🪜 Ladder Backtest")
    if pca_result: tab_names.append("⚖️ Comparison")
    
    tabs = st.tabs(tab_names)
    
    # ─── Tab 1: Yield Curve ───────────────────────────────────────────
    with tabs[0]:
        section_header("U.S. Treasury Yield Curve")
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(plot_yield_curve_3d(ydf), use_container_width=True)
        with c2:
            # Pick some dates for snapshot
            snap_dates = [r["start"], r["end"]]
            mid = pd.Timestamp(r["start"]) + (pd.Timestamp(r["end"]) - pd.Timestamp(r["start"])) / 2
            snap_dates.insert(1, mid.strftime("%Y-%m-%d"))
            st.plotly_chart(plot_yield_curves_snapshot(ydf, snap_dates),
                           use_container_width=True)
        
        # Yield heatmap
        section_header("Yield Level Heatmap")
        step = max(1, len(ydf) // 120)
        sub = ydf.iloc[::step]
        fig_hm = go.Figure(go.Heatmap(
            z=sub.values.T,
            x=[str(d.date()) for d in sub.index],
            y=TENOR_LABELS,
            colorscale="YlOrRd",
            colorbar=dict(title="Yield (%)"),
        ))
        fig_hm.update_layout(
            height=350, margin=dict(l=60, r=30, t=30, b=60),
            xaxis=dict(tickangle=-45, nticks=20),
        )
        st.plotly_chart(fig_hm, use_container_width=True)
        
        # Yield changes
        section_header("Daily Yield Changes (bps)")
        changes = ydf.diff() * 100  # to bps
        fig_chg = go.Figure()
        for tenor in ["2Y", "5Y", "10Y", "30Y"]:
            if tenor in changes.columns:
                fig_chg.add_trace(go.Scatter(
                    x=changes.index, y=changes[tenor].values,
                    mode="lines", name=tenor, line=dict(width=1.2),
                ))
        _plotly_defaults(fig_chg, 350)
        fig_chg.update_layout(xaxis_title="Date", yaxis_title="Δ Yield (bps)")
        st.plotly_chart(fig_chg, use_container_width=True)
    
    # ─── Tab 2: Core Attribution ──────────────────────────────────────
    with tabs[1]:
        section_header("Core PnL Attribution (Campisi Framework)")
        
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_cumulative_pnl(core_df), use_container_width=True)
        with c2:
            st.plotly_chart(plot_waterfall(core_summary), use_container_width=True)
        
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(plot_cumulative_components(core_df), use_container_width=True)
        with c4:
            st.plotly_chart(plot_market_vs_time(core_summary), use_container_width=True)
        
        section_header("Daily PnL Distribution")
        c5, c6 = st.columns(2)
        with c5:
            st.plotly_chart(plot_daily_bar(core_df), use_container_width=True)
        with c6:
            # Histogram
            fig_hist = go.Figure(go.Histogram(
                x=core_df["Actual PnL"].values,
                nbinsx=40,
                marker_color=COLORS["primary"],
                opacity=0.8,
            ))
            fig_hist.update_layout(
                title="PnL Distribution",
                xaxis_title="Daily PnL", yaxis_title="Count",
            )
            st.plotly_chart(_plotly_defaults(fig_hist, 350), use_container_width=True)
        
        # Summary table
        section_header("Aggregate Summary")
        summary_df = core_summary.reset_index() if isinstance(core_summary, pd.Series) else core_summary
        if isinstance(core_summary, pd.Series):
            summary_df = pd.DataFrame({
                "Component": core_summary.index,
                "Value": core_summary.values,
            })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        with st.expander("📋 Daily Attribution Table"):
            st.dataframe(core_df.style.format("{:.6f}", subset=core_df.select_dtypes("number").columns),
                         use_container_width=True, height=400)
    
    # ─── Tab 3: PCA Analysis ─────────────────────────────────────────
    if pca_result:
        with tabs[2]:
            section_header("PCA Yield Curve Decomposition")
            
            info_box(f"<b>3 Principal Components</b> explain <b>{pca_result.cumulative_variance[-1]*100:.1f}%</b> of daily yield-curve variation. "
                     "PC1 ≈ Level (parallel shift), PC2 ≈ Slope (twist), PC3 ≈ Curvature (butterfly).")
            
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(plot_pca_variance(pca_result), use_container_width=True)
            with c2:
                st.plotly_chart(plot_pca_loadings(pca_result), use_container_width=True)
            
            section_header("Factor Loadings Heatmap")
            st.plotly_chart(plot_pca_loadings_heatmap(pca_result), use_container_width=True)
            
            if pca_attr is not None and not pca_attr.empty:
                section_header("PCA-Based PnL Attribution")
                c3, c4 = st.columns(2)
                with c3:
                    st.plotly_chart(plot_pca_attribution_cum(pca_attr), use_container_width=True)
                with c4:
                    # PCA waterfall
                    pca_sum = pca_attr.sum()
                    pca_comps = ["Carry", "Mean PnL", "PC1 PnL", "PC2 PnL", "PC3 PnL", "Residual"]
                    pca_vals = [pca_sum.get(c, 0) for c in pca_comps]
                    fig_pca_wf = go.Figure(go.Waterfall(
                        measure=["relative"] * len(pca_comps) + ["total"],
                        x=pca_comps + ["Actual PnL"],
                        y=pca_vals + [pca_sum.get("Actual PnL", sum(pca_vals))],
                        text=[f"{v:+.4f}" for v in pca_vals] + [f"{pca_sum.get('Actual PnL', sum(pca_vals)):.4f}"],
                        textposition="outside",
                        connector=dict(line=dict(color="#aaa")),
                        increasing=dict(marker=dict(color=COLORS["success"])),
                        decreasing=dict(marker=dict(color=COLORS["danger"])),
                        totals=dict(marker=dict(color=COLORS["purple"])),
                    ))
                    fig_pca_wf.update_layout(title="PCA Attribution Waterfall", showlegend=False)
                    st.plotly_chart(_plotly_defaults(fig_pca_wf, 450), use_container_width=True)
                
                # Factor scores
                section_header("Daily Factor Scores")
                fig_scores = go.Figure()
                for c in pca_result.scores.columns:
                    fig_scores.add_trace(go.Scatter(
                        x=pca_result.scores.index, y=pca_result.scores[c].values,
                        mode="lines", name=c, line=dict(width=1.2),
                    ))
                _plotly_defaults(fig_scores, 350)
                fig_scores.update_layout(xaxis_title="Date", yaxis_title="Score")
                st.plotly_chart(fig_scores, use_container_width=True)
                
                with st.expander("📋 PCA Attribution Table"):
                    st.dataframe(pca_attr.style.format("{:.6f}",
                                 subset=pca_attr.select_dtypes("number").columns),
                                 use_container_width=True, height=400)
    
    # ─── Tab 4: Ladder Backtest ───────────────────────────────────────
    if ladder_result:
        tab_idx = 3 if pca_result else 2
        with tabs[tab_idx]:
            section_header("Bond Ladder Backtest")
            
            ts = ladder_result["portfolio_ts"]
            at = ladder_result["attribution"]
            
            # Key metrics
            m1, m2, m3, m4 = st.columns(4)
            fv = ts["Portfolio Value"].iloc[-1]
            init_v = ts["Portfolio Value"].iloc[0]
            total_ret = ts["Cumulative Return"].iloc[-1]
            # Use elapsed calendar years for consistent annualization
            _start_ts = pd.Timestamp(r["start"])
            _end_ts = pd.Timestamp(r["end"])
            _elapsed = max((_end_ts - _start_ts).days / 365.25, 0.01)
            ann_ret = (1 + total_ret) ** (1 / _elapsed) - 1
            
            m1.metric("Final Value", f"${fv:,.0f}",
                      delta=f"${fv - init_v:+,.0f}")
            m2.metric("Total Return", f"{total_ret*100:.2f}%")
            m3.metric("Annualized", f"{ann_ret*100:.2f}%")
            # True peak-to-trough drawdown
            cum_ret = ts['Cumulative Return']
            running_max = cum_ret.cummax()
            drawdown = cum_ret - running_max
            max_dd = drawdown.min()
            m4.metric("Max Drawdown", f"{max_dd*100:.2f}%")
            
            st.plotly_chart(plot_ladder_portfolio(ts), use_container_width=True)
            
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(plot_ladder_attribution(at), use_container_width=True)
            with c2:
                st.plotly_chart(plot_ladder_waterfall(at), use_container_width=True)
            
            # Holdings
            section_header("Current Holdings")
            holdings = ladder_result["holdings"]
            if not holdings.empty:
                st.dataframe(holdings, use_container_width=True, hide_index=True)
            
            with st.expander("📋 Rebalance Log"):
                st.dataframe(ladder_result["rebalance_log"],
                             use_container_width=True, hide_index=True)
    
    # ─── Tab 5: Comparison ────────────────────────────────────────────
    if pca_result:
        tab_idx = (4 if ladder_result else 3) if pca_result else -1
        with tabs[tab_idx]:
            section_header("Traditional vs PCA-Based Attribution")
            
            info_box("Both frameworks decompose the <b>same actual PnL</b>. "
                     "The traditional (Campisi) approach uses economic intuition "
                     "(carry, duration, convexity, reshape); "
                     "the PCA approach uses statistically-derived factors (level, slope, curvature).")
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### Traditional (Campisi)")
                trad = core_summary
                trad_show = pd.DataFrame({
                    "Component": ["Carry", "Duration", "Convexity", "Curve Reshape",
                                  "Rate Residual", "Time Residual", "**Residual**",
                                  "Market Impact", "Time Impact", "**Actual PnL**"],
                    "Value": [trad.get(c, 0) for c in
                              ["Carry", "Duration", "Convexity", "Curve Reshape",
                               "Rate Residual", "Time Residual", "Residual",
                               "Market Impact", "Time Impact", "Actual PnL"]],
                })
                st.dataframe(trad_show, use_container_width=True, hide_index=True)
            with c2:
                st.markdown("### PCA-Based")
                pca_s = pca_attr.sum() if pca_attr is not None else pd.Series()
                pca_show = pd.DataFrame({
                    "Component": ["Carry", "Mean PnL", "PC1 (Level)",
                                  "PC2 (Slope)", "PC3 (Curvature)",
                                  "PC Total", "**Residual**", "**Actual PnL**"],
                    "Value": [pca_s.get(c, 0) for c in
                              ["Carry", "Mean PnL", "PC1 PnL", "PC2 PnL",
                               "PC3 PnL", "PC Total", "Residual", "Actual PnL"]],
                })
                st.dataframe(pca_show, use_container_width=True, hide_index=True)
            
            # Side-by-side comparison chart
            if pca_attr is not None:
                st.plotly_chart(plot_comparison_bar(core_summary, pca_attr.sum()),
                               use_container_width=True)

else:
    # Landing page
    st.markdown("---")
    st.markdown("""
    ### 🚀 Getting Started
    
    1. **Configure** your analysis in the sidebar (dates, bond spec, features)
    2. Click **Run Analysis** to compute all attributions
    3. Explore results across the interactive tabs
    
    ### 📝 System Features
    
    | Feature | Description |
    |---------|-------------|
    | **Core Attribution** | Campisi-framework PnL decomposition into carry, duration, convexity, curve reshape |
    | **PCA Analysis** | Statistical decomposition of yield-curve movements into level, slope, curvature |
    | **Ladder Backtest** | 5-rung Treasury ladder portfolio backtesting with income & rate attribution |
    | **Yield Curve** | 3D surface, snapshots, heatmap, daily changes visualization |
    
    ### 📊 Data Source
    
    U.S. Treasury constant-maturity yields from **FRED** (Federal Reserve Economic Data).  
    11 tenors: 1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y.
    """)
    
    st.info("👈 Configure parameters in the sidebar and click **Run Analysis** to begin.", icon="ℹ️")
