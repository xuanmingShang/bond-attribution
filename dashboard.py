"""
dashboard.py - Streamlit interactive dashboard for Bond PnL Attribution.

Usage:
    python -m streamlit run dashboard.py
"""
from __future__ import annotations

import inspect
import sys
from io import StringIO
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from bond_pnl.attribution import attribution_summary, run_attribution
from bond_pnl.bond import BondSpec
from bond_pnl.ladder import DEFAULT_RUNGS, IMMUNIZED_RUNGS, LadderBacktest
from bond_pnl.pca import fit_pca, pca_attribution
from bond_pnl.utils import snap_to_business_day
from bond_pnl.yield_curve import (
    TENOR_LABELS,
    TENOR_YEARS,
    YieldCurveHistory,
    fetch_yields,
)


st.set_page_config(
    page_title="Bond PnL Attribution System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    .block-container { padding-top: 1rem; }

    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label {
        font-size: 0.85rem !important;
        color: #475569 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.45rem !important;
        font-weight: 700 !important;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }

    .section-header {
        background: #111827;
        color: white;
        padding: 11px 18px;
        border-radius: 8px;
        margin: 14px 0 10px 0;
        font-size: 1.05rem;
        font-weight: 650;
    }

    .info-box {
        background: #eff6ff;
        border-left: 4px solid #2563eb;
        padding: 11px 14px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        font-size: 0.9rem;
    }

    .muted-note {
        color: #64748b;
        font-size: 0.9rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


COLORS = {
    "primary": "#2563eb",
    "success": "#059669",
    "danger": "#dc2626",
    "warning": "#d97706",
    "purple": "#7c3aed",
    "gray": "#64748b",
    "carry": "#059669",
    "duration": "#2563eb",
    "convexity": "#dc2626",
    "reshape": "#d97706",
    "residual": "#64748b",
    "rolldown": "#7c3aed",
    "funding": "#db2777",
    "accrual": "#0f766e",
    "market": "#2563eb",
    "time": "#059669",
    "pc1": "#2563eb",
    "pc2": "#dc2626",
    "pc3": "#d97706",
}

DEFAULT_START = pd.Timestamp("2023-06-01")
DEFAULT_END = pd.Timestamp("2024-06-30")
DEFAULT_LADDER_RUNGS = tuple(DEFAULT_RUNGS)
DEFAULT_IMMUNIZED_RUNGS = tuple(IMMUNIZED_RUNGS)
LADDER_STRATEGIES = {
    "Classic Roll": "classic",
    "Withdrawal": "withdrawal",
    "Immunized": "immunized",
}
FREQ_LABELS = {1: "Annual", 2: "Semi-annual", 4: "Quarterly"}
FREQ_VALUES = {label: value for value, label in FREQ_LABELS.items()}


def section_header(text: str) -> None:
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def info_box(text: str) -> None:
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)


def muted_note(text: str) -> None:
    st.markdown(f'<div class="muted-note">{text}</div>', unsafe_allow_html=True)


def _snap_to_business_day(
    ydf: pd.DataFrame,
    date_str: str,
    direction: str = "forward",
) -> str:
    return snap_to_business_day(ydf, date_str, direction)


@st.cache_data(show_spinner=False)
def load_yields(start: str, end: str) -> pd.DataFrame:
    return fetch_yields(start, end)


@st.cache_data(show_spinner=False)
def compute_core(
    ydf_json: str,
    start: str,
    end: str,
    maturity_years: int,
    coupon: float,
    face: float,
    freq: int,
):
    ydf = pd.read_json(StringIO(ydf_json))
    ch = YieldCurveHistory(ydf)
    sd = pd.Timestamp(start)
    md = sd + pd.DateOffset(years=maturity_years)
    if coupon <= 0:
        coupon = round(ch[start].rate(float(maturity_years)), 4)
    else:
        coupon = coupon / 100.0

    bspec = dict(
        maturity=md.strftime("%Y-%m-%d"),
        face=face,
        coupon=coupon,
        freq=freq,
        day_count="ACT/ACT",
        issue_date=start,
    )
    bond = BondSpec(**bspec)
    financing_rate = ch[start].rate(0.25)
    df = run_attribution(bond, ch, start, end, financing_rate)
    summary = attribution_summary(df)
    return df, summary, bond.summary(), coupon, financing_rate


@st.cache_data(show_spinner=False)
def compute_pca(
    ydf_json: str,
    start: str,
    end: str,
    bond_spec_json: str,
    coupon: float,
    face: float,
    freq: int,
):
    ydf = pd.read_json(StringIO(ydf_json))
    ch = YieldCurveHistory(ydf)
    bond_info = pd.read_json(StringIO(bond_spec_json), typ="series")
    bond = BondSpec(
        maturity=bond_info["Maturity"],
        face=face,
        coupon=coupon,
        freq=freq,
        day_count="ACT/ACT",
        issue_date=start,
    )
    pca_result = fit_pca(ch, start, end, 3)
    financing_rate = ch[start].rate(0.25)
    pca_attr = pca_attribution(bond, ch, pca_result, start, end, financing_rate)
    return pca_result, pca_attr


@st.cache_data(show_spinner=False)
def compute_ladder(
    ydf_json: str,
    start: str,
    end: str,
    capital: float,
    rungs: tuple[int, ...],
    strategy: str,
    withdrawal_amount: float,
    withdrawal_frequency: str,
    first_withdrawal_date: str | None,
    target_mode: str,
    target_duration: float | None,
    liability_date: str | None,
    liability_amount: float | None,
):
    ydf = pd.read_json(StringIO(ydf_json))
    ch = YieldCurveHistory(ydf)
    financing_rate = ch[start].rate(0.25)
    bt = LadderBacktest(
        ch,
        start,
        end,
        list(rungs),
        capital,
        12,
        financing_rate,
        strategy=strategy,
        withdrawal_amount=withdrawal_amount,
        withdrawal_frequency=withdrawal_frequency,
        first_withdrawal_date=first_withdrawal_date,
        target_mode=target_mode,
        target_duration=target_duration,
        liability_date=liability_date,
        liability_amount=liability_amount,
    )
    return bt.run()


def _plotly_defaults(fig: go.Figure, h: int = 500) -> go.Figure:
    fig.update_layout(
        height=h,
        template="plotly_white",
        font=dict(family="Inter, sans-serif", size=12),
        margin=dict(l=50, r=30, t=50, b=50),
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        hovermode="x unified",
    )
    return fig


def _surface_sample(ydf: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    step = max(1, len(ydf) // 80)
    sub = ydf.iloc[::step]
    if len(sub) and sub.index[-1] != ydf.index[-1]:
        sub = pd.concat([sub, ydf.iloc[[-1]]])
    date_labels = [str(d.date()) for d in sub.index]
    return sub, date_labels


def plot_yield_curve_3d(ydf: pd.DataFrame) -> tuple[go.Figure, list[str]]:
    sub, date_labels = _surface_sample(ydf)
    customdata = np.array([[d] * len(TENOR_YEARS) for d in date_labels], dtype=object)

    fig = go.Figure(
        data=[
            go.Surface(
                z=sub.values,
                x=TENOR_YEARS,
                y=list(range(len(sub))),
                customdata=customdata,
                colorscale="Viridis",
                opacity=0.92,
                showscale=True,
                colorbar=dict(title="Yield (%)", len=0.62),
                hovertemplate=(
                    "Tenor: %{x}Y<br>"
                    "Date: %{customdata}<br>"
                    "Yield: %{z:.2f}%<extra></extra>"
                ),
            )
        ]
    )
    tick_step = max(1, len(sub) // 6)
    tick_vals = list(range(0, len(sub), tick_step))
    fig.update_layout(
        title="Yield Curve Surface",
        scene=dict(
            xaxis_title="Tenor (years)",
            yaxis_title="Date",
            zaxis_title="Yield (%)",
            yaxis=dict(tickvals=tick_vals, ticktext=[date_labels[i] for i in tick_vals]),
            camera=dict(eye=dict(x=1.55, y=-1.45, z=1.0)),
        ),
        height=590,
        margin=dict(l=0, r=0, t=45, b=0),
        template="plotly_white",
    )
    return fig, date_labels


def plot_yield_curve_snapshot(ydf: pd.DataFrame, date: str) -> go.Figure:
    idx = ydf.index.get_indexer([pd.Timestamp(date)], method="nearest")[0]
    row = ydf.iloc[idx]
    label = str(ydf.index[idx].date())
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=TENOR_LABELS,
            y=row.values,
            mode="lines+markers",
            name=label,
            line=dict(width=3, color=COLORS["primary"]),
            marker=dict(size=7),
        )
    )
    fig.update_layout(
        title=f"Yield Curve Snapshot: {label}",
        xaxis_title="Tenor",
        yaxis_title="Yield (%)",
        showlegend=False,
    )
    return _plotly_defaults(fig, 420)


def plot_cumulative_pnl(df: pd.DataFrame) -> go.Figure:
    cum = df["Actual PnL"].cumsum()
    dates = df["Date"] if "Date" in df.columns else df.index
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=cum.values,
            fill="tozeroy",
            fillcolor="rgba(37,99,235,0.12)",
            line=dict(color=COLORS["primary"], width=2.5),
            name="Cumulative PnL",
        )
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Cumulative PnL",
        title="Cumulative PnL Over Time",
    )
    return _plotly_defaults(fig, 400)


def plot_waterfall(summary: pd.Series) -> go.Figure:
    components = ["Carry", "Duration", "Convexity", "Curve Reshape", "Residual"]
    vals = [summary.get(c, 0) for c in components]
    actual = summary.get("Actual PnL", sum(vals))

    fig = go.Figure(
        go.Waterfall(
            name="PnL",
            orientation="v",
            measure=["relative"] * len(components) + ["total"],
            x=components + ["Actual PnL"],
            y=vals + [actual],
            textposition="outside",
            text=[f"{v:+.4f}" for v in vals] + [f"{actual:.4f}"],
            connector=dict(line=dict(color="#94a3b8", width=1)),
            increasing=dict(marker=dict(color=COLORS["success"])),
            decreasing=dict(marker=dict(color=COLORS["danger"])),
            totals=dict(marker=dict(color=COLORS["primary"])),
        )
    )
    fig.update_layout(title="PnL Attribution Waterfall", yaxis_title="PnL", showlegend=False)
    return _plotly_defaults(fig, 450)


def plot_market_vs_time(summary: pd.Series) -> go.Figure:
    categories = ["Market Impact", "Time Impact", "Residual"]
    values = [summary.get("Market Impact", 0), summary.get("Time Impact", 0), summary.get("Residual", 0)]
    bar_colors = [COLORS["market"], COLORS["time"], COLORS["residual"]]
    fig = go.Figure(
        go.Bar(
            x=categories,
            y=values,
            marker_color=bar_colors,
            text=[f"{v:+.4f}" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(title="PnL Decomposition (Market vs Time)", yaxis_title="PnL", showlegend=False)
    return _plotly_defaults(fig, 400)


def plot_cumulative_components(df: pd.DataFrame) -> go.Figure:
    components = [
        "Carry",
        "Duration",
        "Convexity",
        "Curve Reshape",
        "Market Impact",
        "Time Impact",
        "Actual PnL",
    ]
    avail = [c for c in components if c in df.columns]
    cum = df[avail].cumsum()
    dates = df["Date"] if "Date" in df.columns else df.index

    fig = go.Figure()
    comp_colors = {
        "Carry": COLORS["carry"],
        "Duration": COLORS["duration"],
        "Convexity": COLORS["convexity"],
        "Curve Reshape": COLORS["reshape"],
        "Market Impact": COLORS["market"],
        "Time Impact": COLORS["time"],
        "Actual PnL": "#111827",
    }
    for component in avail:
        is_actual = component == "Actual PnL"
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=cum[component].values,
                mode="lines",
                name=component,
                line=dict(
                    color=comp_colors.get(component, "#64748b"),
                    width=3 if is_actual else 1.5,
                    dash="solid",
                ),
            )
        )
    fig.update_layout(
        title="Cumulative Components",
        xaxis_title="Date",
        yaxis_title="Cumulative",
    )
    return _plotly_defaults(fig, 420)


def plot_pca_variance(pr) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    labels = [f"PC{i + 1}" for i in range(pr.n_components)]
    fig.add_trace(
        go.Bar(
            x=labels,
            y=pr.explained_variance_ratio * 100,
            name="Variance (%)",
            marker_color=[COLORS["pc1"], COLORS["pc2"], COLORS["pc3"]],
            text=[f"{v:.1f}%" for v in pr.explained_variance_ratio * 100],
            textposition="outside",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=pr.cumulative_variance * 100,
            name="Cumulative",
            mode="lines+markers",
            line=dict(color="#111827", width=2.5),
            marker=dict(size=10, symbol="diamond"),
        ),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="Individual (%)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative (%)", secondary_y=True, range=[0, 105])
    fig.update_layout(title="PCA Variance Explained")
    return _plotly_defaults(fig, 400)


def plot_pca_loadings(pr) -> go.Figure:
    fig = go.Figure()
    names = ["Level (PC1)", "Slope (PC2)", "Curvature (PC3)"]
    colors_pc = [COLORS["pc1"], COLORS["pc2"], COLORS["pc3"]]
    tenors = list(pr.loadings.columns)
    for i in range(min(3, pr.n_components)):
        fig.add_trace(
            go.Scatter(
                x=tenors,
                y=pr.loadings.iloc[i].values,
                mode="lines+markers",
                name=names[i] if i < 3 else f"PC{i + 1}",
                line=dict(color=colors_pc[i], width=2.5),
                marker=dict(size=8),
            )
        )
    fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
    fig.update_layout(title="Factor Loadings by Tenor", xaxis_title="Tenor", yaxis_title="Loading")
    return _plotly_defaults(fig, 400)


def plot_pca_loadings_heatmap(pr) -> go.Figure:
    names = ["PC1 (Level)", "PC2 (Slope)", "PC3 (Curvature)"]
    fig = go.Figure(
        go.Heatmap(
            z=pr.loadings.values[:3],
            x=list(pr.loadings.columns),
            y=names[: pr.n_components],
            colorscale="RdBu_r",
            zmid=0,
            text=np.round(pr.loadings.values[:3], 3),
            texttemplate="%{text}",
            textfont=dict(size=11),
        )
    )
    fig.update_layout(title="Factor Loadings Heatmap", height=280, margin=dict(l=80, r=30, t=50, b=50))
    return fig


def plot_pca_attribution_cum(pa: pd.DataFrame) -> go.Figure:
    components = ["Carry", "PC1 PnL", "PC2 PnL", "PC3 PnL", "Residual", "Actual PnL"]
    avail = [c for c in components if c in pa.columns]
    cum = pa[avail].cumsum()
    dates = pa["Date"] if "Date" in pa.columns else pa.index

    fig = go.Figure()
    colors = {
        "Carry": COLORS["carry"],
        "PC1 PnL": COLORS["pc1"],
        "PC2 PnL": COLORS["pc2"],
        "PC3 PnL": COLORS["pc3"],
        "Residual": COLORS["residual"],
        "Actual PnL": "#111827",
    }
    for component in avail:
        is_actual = component == "Actual PnL"
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=cum[component].values,
                mode="lines",
                name=component,
                line=dict(
                    color=colors.get(component, "#64748b"),
                    width=3 if is_actual else 1.5,
                    dash="dash" if is_actual else "solid",
                ),
            )
        )
    fig.update_layout(title="Cumulative PCA Attribution", xaxis_title="Date", yaxis_title="Cumulative PnL")
    return _plotly_defaults(fig, 420)


def plot_pca_waterfall(pca_attr: pd.DataFrame) -> go.Figure:
    pca_sum = pca_attr.sum()
    components = ["Carry", "PC1 PnL", "PC2 PnL", "PC3 PnL", "Residual"]
    vals = [pca_sum.get(c, 0) for c in components]
    actual = pca_sum.get("Actual PnL", sum(vals))
    fig = go.Figure(
        go.Waterfall(
            measure=["relative"] * len(components) + ["total"],
            x=components + ["Actual PnL"],
            y=vals + [actual],
            text=[f"{v:+.4f}" for v in vals] + [f"{actual:.4f}"],
            textposition="outside",
            connector=dict(line=dict(color="#94a3b8")),
            increasing=dict(marker=dict(color=COLORS["success"])),
            decreasing=dict(marker=dict(color=COLORS["danger"])),
            totals=dict(marker=dict(color=COLORS["purple"])),
        )
    )
    fig.update_layout(title="PCA Attribution Waterfall", showlegend=False)
    return _plotly_defaults(fig, 450)


def plot_pca_scores(pr) -> go.Figure:
    fig = go.Figure()
    for col in pr.scores.columns:
        fig.add_trace(
            go.Scatter(
                x=pr.scores.index,
                y=pr.scores[col].values,
                mode="lines",
                name=col,
                line=dict(width=1.2),
            )
        )
    fig.update_layout(title="Daily Factor Scores", xaxis_title="Date", yaxis_title="Score")
    return _plotly_defaults(fig, 350)


def plot_ladder_portfolio(ts: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=ts.index,
            y=ts["Portfolio Value"],
            mode="lines",
            name="Portfolio Value",
            line=dict(color=COLORS["primary"], width=2.5),
        ),
        secondary_y=False,
    )
    if "Cash Balance" in ts.columns:
        fig.add_trace(
            go.Scatter(
                x=ts.index,
                y=ts["Cash Balance"],
                mode="lines",
                name="Cash Balance",
                line=dict(color=COLORS["warning"], width=1.8, dash="dot"),
            ),
            secondary_y=False,
        )
    fig.add_trace(
        go.Scatter(
            x=ts.index,
            y=ts["Cumulative Return"] * 100,
            mode="lines",
            name="Cumulative Return (%)",
            line=dict(color=COLORS["success"], width=2.5),
        ),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="Portfolio Value ($)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Return (%)", secondary_y=True)
    fig.update_layout(title="Ladder Portfolio Value and Return")
    return _plotly_defaults(fig, 430)


def plot_ladder_cashflows(ts: pd.DataFrame) -> go.Figure:
    columns = [
        ("Coupon Cashflow", COLORS["accrual"]),
        ("Principal Cashflow", COLORS["primary"]),
        ("Withdrawal Paid", COLORS["success"]),
        ("Withdrawal Shortfall", COLORS["danger"]),
    ]
    fig = go.Figure()
    for column, color in columns:
        if column in ts.columns and ts[column].abs().sum() > 0:
            fig.add_trace(
                go.Bar(
                    x=ts.index,
                    y=ts[column],
                    name=column,
                    marker_color=color,
                )
            )
    fig.update_layout(
        title="Ladder Cashflows",
        xaxis_title="Date",
        yaxis_title="Cashflow ($)",
        barmode="group",
    )
    return _plotly_defaults(fig, 360)


def plot_ladder_duration_match(match_ts: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if match_ts.empty:
        fig.update_layout(title="Portfolio Duration")
        return _plotly_defaults(fig, 360)
    fig.add_trace(
        go.Scatter(
            x=match_ts.index,
            y=match_ts["Portfolio Duration"],
            mode="lines",
            name="Portfolio Duration",
            line=dict(color=COLORS["primary"], width=2.2),
        )
    )
    if "Target Duration" in match_ts.columns and match_ts["Target Duration"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=match_ts.index,
                y=match_ts["Target Duration"],
                mode="lines",
                name="Target Duration",
                line=dict(color=COLORS["danger"], width=2, dash="dash"),
            )
        )
    fig.update_layout(title="Duration Profile", xaxis_title="Date", yaxis_title="Duration")
    return _plotly_defaults(fig, 360)


def plot_ladder_attribution(at: pd.DataFrame) -> go.Figure:
    components = ["Income", "Rolldown", "Rate Movement", "Residual"]
    cum = at[components].cumsum()
    fig = go.Figure()
    colors = {
        "Income": COLORS["accrual"],
        "Rolldown": COLORS["rolldown"],
        "Rate Movement": COLORS["duration"],
        "Residual": COLORS["residual"],
    }
    for component in components:
        fig.add_trace(
            go.Scatter(
                x=cum.index,
                y=cum[component],
                mode="lines",
                name=component,
                line=dict(color=colors[component], width=2),
            )
        )
    fig.update_layout(title="Cumulative Ladder Attribution", xaxis_title="Date", yaxis_title="PnL ($)")
    return _plotly_defaults(fig, 420)


def plot_ladder_waterfall(at: pd.DataFrame) -> go.Figure:
    sums = at[["Income", "Rolldown", "Rate Movement", "Residual"]].sum()
    total = at["Total PnL"].sum()
    fig = go.Figure(
        go.Waterfall(
            measure=["relative"] * len(sums) + ["total"],
            x=list(sums.index) + ["Total PnL"],
            y=list(sums.values) + [total],
            text=[f"${v:,.0f}" for v in sums.values] + [f"${total:,.0f}"],
            textposition="outside",
            connector=dict(line=dict(color="#94a3b8")),
            increasing=dict(marker=dict(color=COLORS["success"])),
            decreasing=dict(marker=dict(color=COLORS["danger"])),
            totals=dict(marker=dict(color=COLORS["primary"])),
        )
    )
    fig.update_layout(title="Ladder Attribution Waterfall", yaxis_title="PnL ($)", showlegend=False)
    return _plotly_defaults(fig, 420)


def _supports_plotly_selection() -> bool:
    try:
        return "on_select" in inspect.signature(st.plotly_chart).parameters
    except (TypeError, ValueError):
        return False


def _render_plotly_with_optional_selection(fig: go.Figure, key: str):
    if _supports_plotly_selection():
        return st.plotly_chart(
            fig,
            use_container_width=True,
            key=key,
            on_select="rerun",
            selection_mode="points",
        )
    st.plotly_chart(fig, use_container_width=True)
    return None


def _event_selection_points(event: Any) -> list[dict[str, Any]]:
    if event is None:
        return []
    selection = event.get("selection") if hasattr(event, "get") else getattr(event, "selection", None)
    if selection is None:
        return []
    points = selection.get("points") if hasattr(selection, "get") else getattr(selection, "points", None)
    return list(points or [])


def _coerce_date_label(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and len(value) >= 10:
        return value[:10]
    if isinstance(value, (list, tuple, np.ndarray)) and len(value):
        return _coerce_date_label(value[0])
    return None


def _selected_date_from_surface_event(event: Any, surface_dates: list[str]) -> str | None:
    for point in _event_selection_points(event):
        custom_date = _coerce_date_label(point.get("customdata"))
        if custom_date:
            return custom_date

        y_value = point.get("y")
        if isinstance(y_value, (int, float)) and np.isfinite(y_value):
            idx = int(round(y_value))
            if 0 <= idx < len(surface_dates):
                return surface_dates[idx]
    return None


def _ensure_session_calendar_date(key: str, ydf: pd.DataFrame, preferred: str) -> None:
    if ydf.empty:
        return
    preferred_date = pd.Timestamp(preferred).date()
    min_date = ydf.index.min().date()
    max_date = ydf.index.max().date()
    if preferred_date < min_date or preferred_date > max_date:
        preferred_date = max_date

    current = st.session_state.get(key)
    if current is None:
        st.session_state[key] = preferred_date
        return

    current_date = pd.Timestamp(current).date()
    if current_date < min_date or current_date > max_date:
        st.session_state[key] = preferred_date


def _load_yield_window(start_date, end_date, minimum_days: int, label: str):
    if start_date >= end_date:
        st.error("Start date must be before end date.")
        return None
    if (end_date - start_date).days < 30:
        st.warning("Window is very short (< 30 days). Results may be unreliable.")

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    try:
        with st.spinner(f"Loading yields for {label}..."):
            ydf = load_yields(start_str, end_str)
    except FileNotFoundError as exc:
        st.error(f"FRED API key not found. Set `FRED_API_KEY` or create `fred-api-key.txt`. ({exc})")
        return None
    except Exception as exc:  # pragma: no cover - Streamlit user-facing path
        st.error(f"Failed to load yield data: {exc}")
        return None

    if len(ydf) < minimum_days:
        st.error(f"Only {len(ydf)} trading days available. Need at least {minimum_days}.")
        return None

    start_snap = _snap_to_business_day(ydf, start_str, "forward")
    end_snap = _snap_to_business_day(ydf, end_str, "backward")
    return ydf, start_snap, end_snap


def _format_summary_table(summary: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"Component": summary.index, "Value": summary.values})


def _view_selector(label: str, options: list[str], key: str) -> str:
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[0]
    if hasattr(st, "segmented_control"):
        return st.segmented_control(label, options, key=key)
    return st.radio(label, options, horizontal=True, key=key)


def _display_bond_assumptions(result: dict[str, Any]) -> None:
    with st.expander("Bond Specification and Assumptions", expanded=False):
        left, right = st.columns(2)
        with left:
            for key, value in result["bond_info"].items():
                st.markdown(f"**{key}:** {value}")
        with right:
            st.markdown(f"**Financing Rate (3M):** {result['fin_rate'] * 100:.2f}%")
            st.markdown(f"**Analysis Window:** {result['start']} to {result['end']}")
            st.markdown("**Day Count:** ACT/ACT")
            st.markdown("**Compounding:** Continuous")
            info_box("The model uses cubic-spline interpolated CMT yields as a zero-rate proxy.")


def _render_field_tab() -> None:
    section_header("Field Parameters")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        start_date = st.date_input("Start", DEFAULT_START, key="field_start")
    with c2:
        end_date = st.date_input("End", DEFAULT_END, key="field_end")
    with c3:
        st.write("")
        st.write("")
        load_btn = st.button("Load Field", type="primary", use_container_width=True, key="field_load")

    if load_btn:
        loaded = _load_yield_window(start_date, end_date, minimum_days=10, label="Field")
        if loaded is not None:
            ydf, start_snap, end_snap = loaded
            st.session_state["field_results"] = {"ydf": ydf, "start": start_snap, "end": end_snap}
            st.session_state.pop("field_snapshot_date", None)

    if "field_results" not in st.session_state:
        st.info("Set a date window and click Load Field to inspect the yield-curve surface.")
        return

    result = st.session_state["field_results"]
    ydf = result["ydf"]
    start = result["start"]
    end = result["end"]

    section_header("Field Summary")
    start_row = ydf.loc[pd.Timestamp(start)]
    end_row = ydf.loc[pd.Timestamp(end)]
    ten_year_delta_bps = (end_row["10Y"] - start_row["10Y"]) * 100
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Trading Days", f"{len(ydf)}")
    m2.metric("Start 10Y", f"{start_row['10Y']:.2f}%")
    m3.metric("End 10Y", f"{end_row['10Y']:.2f}%")
    m4.metric("10Y Change", f"{ten_year_delta_bps:+.1f} bps")
    muted_note(f"Loaded window: {start} to {end}. Non-business input dates are snapped to available FRED dates.")

    section_header("Yield Curve Surface and Snapshot")
    left, right = st.columns([1.35, 1])
    with left:
        surface_fig, surface_dates = plot_yield_curve_3d(ydf)
        event = _render_plotly_with_optional_selection(surface_fig, key="field_surface")
        selected_from_surface = _selected_date_from_surface_event(event, surface_dates)
        if selected_from_surface:
            st.session_state["field_snapshot_date"] = pd.Timestamp(selected_from_surface).date()
        if not _supports_plotly_selection():
            muted_note("This Streamlit version does not expose Plotly selection events. Use the date selector.")

    with right:
        _ensure_session_calendar_date("field_snapshot_date", ydf, preferred=end)
        selected_calendar_date = st.date_input(
            "Snapshot date",
            key="field_snapshot_date",
            min_value=ydf.index.min().date(),
            max_value=ydf.index.max().date(),
        )
        selected_idx = ydf.index.get_indexer([pd.Timestamp(selected_calendar_date)], method="nearest")[0]
        selected_date = str(ydf.index[selected_idx].date())
        st.plotly_chart(plot_yield_curve_snapshot(ydf, selected_date), use_container_width=True)
        selected_row = ydf.iloc[selected_idx]
        s1, s2, s3 = st.columns(3)
        s1.metric("2Y", f"{selected_row['2Y']:.2f}%")
        s2.metric("10Y", f"{selected_row['10Y']:.2f}%")
        s3.metric("30Y", f"{selected_row['30Y']:.2f}%")


def _run_attribution_workflow(
    start_date,
    end_date,
    maturity_years: int,
    coupon_pct: float,
    face_value: float,
    freq: int,
    run_pca: bool,
) -> None:
    loaded = _load_yield_window(start_date, end_date, minimum_days=10, label="Attribution")
    if loaded is None:
        return

    ydf, start_snap, end_snap = loaded
    ydf_json = ydf.to_json()
    with st.spinner("Computing core attribution..."):
        core_df, core_summary, bond_info, coupon_used, fin_rate = compute_core(
            ydf_json,
            start_snap,
            end_snap,
            maturity_years,
            coupon_pct,
            face_value,
            freq,
        )

    pca_result = pca_attr = None
    if run_pca:
        with st.spinner("Computing PCA analysis..."):
            bond_info_json = pd.Series(bond_info).to_json()
            pca_result, pca_attr = compute_pca(
                ydf_json,
                start_snap,
                end_snap,
                bond_info_json,
                coupon_used,
                face_value,
                freq,
            )

    st.session_state["attr_results"] = {
        "ydf": ydf,
        "core_df": core_df,
        "core_summary": core_summary,
        "bond_info": bond_info,
        "coupon_used": coupon_used,
        "fin_rate": fin_rate,
        "pca_result": pca_result,
        "pca_attr": pca_attr,
        "start": start_snap,
        "end": end_snap,
    }


def _render_core_view(core_df: pd.DataFrame, core_summary: pd.Series) -> None:
    section_header("Core PnL Attribution")
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

    section_header("Aggregate Summary")
    st.dataframe(_format_summary_table(core_summary), use_container_width=True, hide_index=True)

    with st.expander("Daily Attribution Table"):
        st.dataframe(
            core_df.style.format("{:.6f}", subset=core_df.select_dtypes("number").columns),
            use_container_width=True,
            height=400,
        )


def _render_pca_view(pca_result, pca_attr: pd.DataFrame | None) -> None:
    section_header("PCA Yield Curve Decomposition")
    info_box(
        f"3 principal components explain {pca_result.cumulative_variance[-1] * 100:.1f}% "
        "of daily yield-curve variation. PC1 ~= level, PC2 ~= slope, PC3 ~= curvature."
    )

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
            st.plotly_chart(plot_pca_waterfall(pca_attr), use_container_width=True)

        st.plotly_chart(plot_pca_scores(pca_result), use_container_width=True)

        with st.expander("PCA Attribution Table"):
            st.dataframe(
                pca_attr.style.format("{:.6f}", subset=pca_attr.select_dtypes("number").columns),
                use_container_width=True,
                height=400,
            )


def _render_compare_view(core_summary: pd.Series, pca_attr: pd.DataFrame | None) -> None:
    section_header("Traditional vs PCA-Based Attribution")
    info_box(
        "Both frameworks explain the same actual PnL. Traditional attribution uses economic "
        "buckets; PCA attribution uses statistical yield-curve factors."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Traditional (Campisi)")
        trad_show = pd.DataFrame(
            {
                "Component": [
                    "Carry",
                    "Duration",
                    "Convexity",
                    "Curve Reshape",
                    "Rate Residual",
                    "Time Residual",
                    "Residual",
                    "Market Impact",
                    "Time Impact",
                    "Actual PnL",
                ],
                "Value": [
                    core_summary.get(c, 0)
                    for c in [
                        "Carry",
                        "Duration",
                        "Convexity",
                        "Curve Reshape",
                        "Rate Residual",
                        "Time Residual",
                        "Residual",
                        "Market Impact",
                        "Time Impact",
                        "Actual PnL",
                    ]
                ],
            }
        )
        st.dataframe(trad_show, use_container_width=True, hide_index=True)

    with c2:
        st.markdown("### PCA-Based")
        pca_summary = pca_attr.sum() if pca_attr is not None else pd.Series(dtype=float)
        pca_show = pd.DataFrame(
            {
                "Component": [
                    "Carry",
                    "PC1 (Level)",
                    "PC2 (Slope)",
                    "PC3 (Curvature)",
                    "PC Total",
                    "Residual",
                    "Actual PnL",
                ],
                "Value": [
                    pca_summary.get(c, 0)
                    for c in [
                        "Carry",
                        "PC1 PnL",
                        "PC2 PnL",
                        "PC3 PnL",
                        "PC Total",
                        "Residual",
                        "Actual PnL",
                    ]
                ],
            }
        )
        st.dataframe(pca_show, use_container_width=True, hide_index=True)


def _render_attribution_tab() -> None:
    section_header("Attribution Parameters")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        start_date = st.date_input("Start", DEFAULT_START, key="attr_start")
    with c2:
        end_date = st.date_input("End", DEFAULT_END, key="attr_end")
    with c3:
        maturity_years = st.slider("Maturity (years)", 1, 30, 5, key="attr_maturity")
    with c4:
        coupon_pct = st.number_input(
            "Coupon Rate (%)",
            0.0,
            15.0,
            0.0,
            step=0.25,
            help="0 = auto par coupon",
            key="attr_coupon",
        )

    c5, c6, c7, c8 = st.columns([1, 1, 1, 1])
    with c5:
        face_value = st.number_input("Face Value ($)", 10.0, 10000.0, 100.0, step=10.0, key="attr_face")
    with c6:
        freq_label = st.selectbox(
            "Coupon Frequency",
            ["Semi-annual", "Annual", "Quarterly"],
            index=0,
            key="attr_freq",
        )
        freq = FREQ_VALUES[freq_label]
    with c7:
        run_pca = st.checkbox("Compute PCA", value=True, key="attr_run_pca")
    with c8:
        st.write("")
        st.write("")
        run_btn = st.button("Run Attribution", type="primary", use_container_width=True, key="attr_run")

    if run_btn:
        _run_attribution_workflow(start_date, end_date, maturity_years, coupon_pct, face_value, freq, run_pca)

    if "attr_results" not in st.session_state:
        st.info("Set attribution parameters and click Run Attribution.")
        return

    result = st.session_state["attr_results"]
    core_df = result["core_df"]
    core_summary = result["core_summary"]
    pca_result = result["pca_result"]
    pca_attr = result["pca_attr"]

    section_header("Attribution Summary")
    actual = core_summary.get("Actual PnL", 0)
    market = core_summary.get("Market Impact", 0)
    time_impact = core_summary.get("Time Impact", 0)
    residual = core_summary.get("Residual", 0)
    residual_pct = abs(residual / actual) * 100 if actual else 0
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Actual PnL", f"{actual:.4f}", delta="Gain" if actual >= 0 else "Loss")
    m2.metric("Market Impact", f"{market:.4f}")
    m3.metric("Time Impact", f"{time_impact:.4f}")
    m4.metric("Residual", f"{residual:.6f}")
    m5.metric("|Res / Act|", f"{residual_pct:.2f}%")
    m6.metric("Trading Days", f"{len(core_df)}")
    muted_note(f"Results shown for {result['start']} to {result['end']}.")
    _display_bond_assumptions(result)

    options = ["Core"]
    if pca_result is not None:
        options.extend(["PCA", "Compare"])
    view = _view_selector("Attribution view", options, key="attr_view")
    if view == "Core":
        _render_core_view(core_df, core_summary)
    elif view == "PCA":
        _render_pca_view(pca_result, pca_attr)
    else:
        _render_compare_view(core_summary, pca_attr)


def _run_ladder_workflow(
    start_date,
    end_date,
    capital: float,
    strategy_label: str,
    withdrawal_amount: float,
    withdrawal_frequency: str,
    first_withdrawal_date,
    target_mode_label: str,
    target_duration: float | None,
    liability_date,
    liability_amount: float | None,
) -> None:
    loaded = _load_yield_window(start_date, end_date, minimum_days=10, label="Ladder")
    if loaded is None:
        return

    ydf, start_snap, end_snap = loaded
    strategy = LADDER_STRATEGIES[strategy_label]
    rungs = DEFAULT_IMMUNIZED_RUNGS if strategy == "immunized" else DEFAULT_LADDER_RUNGS
    target_mode = "liability" if target_mode_label == "Liability" else "target_duration"
    first_withdrawal = None
    if strategy == "withdrawal" and first_withdrawal_date is not None:
        first_withdrawal = pd.Timestamp(first_withdrawal_date).strftime("%Y-%m-%d")
    liability_date_str = None
    if strategy == "immunized" and target_mode == "liability" and liability_date is not None:
        liability_date_str = pd.Timestamp(liability_date).strftime("%Y-%m-%d")

    with st.spinner("Running ladder backtest..."):
        ladder_result = compute_ladder(
            ydf.to_json(),
            start_snap,
            end_snap,
            capital,
            rungs,
            strategy,
            float(withdrawal_amount if strategy == "withdrawal" else 0.0),
            withdrawal_frequency,
            first_withdrawal,
            target_mode,
            float(target_duration) if strategy == "immunized" and target_mode == "target_duration" else None,
            liability_date_str,
            float(liability_amount) if strategy == "immunized" and target_mode == "liability" else None,
        )
    st.session_state["ladder_results"] = {
        "ydf": ydf,
        "ladder_result": ladder_result,
        "start": start_snap,
        "end": end_snap,
        "capital": capital,
        "strategy_label": strategy_label,
        "strategy": strategy,
        "rungs": rungs,
    }


def _render_ladder_tab() -> None:
    section_header("Ladder Parameters")
    strategy_label = _view_selector("Strategy", list(LADDER_STRATEGIES.keys()), key="ladder_strategy")

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        start_date = st.date_input("Start", DEFAULT_START, key="ladder_start")
    with c2:
        end_date = st.date_input("End", DEFAULT_END, key="ladder_end")
    with c3:
        capital = st.number_input("Capital ($)", 100_000, 100_000_000, 1_000_000, step=100_000, key="ladder_capital")

    withdrawal_amount = 0.0
    withdrawal_frequency = "Annual"
    first_withdrawal_date = None
    target_mode_label = "Target duration"
    target_duration = 5.0
    liability_date = None
    liability_amount = None

    if strategy_label == "Withdrawal":
        w1, w2, w3 = st.columns([1, 1, 1])
        with w1:
            withdrawal_amount = st.number_input(
                "Withdrawal amount ($)",
                0,
                10_000_000,
                25_000,
                step=5_000,
                key="ladder_withdrawal_amount",
            )
        with w2:
            withdrawal_frequency = st.selectbox(
                "Withdrawal frequency",
                ["Monthly", "Quarterly", "Semiannual", "Annual"],
                index=3,
                key="ladder_withdrawal_frequency",
            )
        with w3:
            first_withdrawal_date = st.date_input(
                "First withdrawal",
                start_date,
                key="ladder_first_withdrawal",
            )
    elif strategy_label == "Immunized":
        i1, i2, i3 = st.columns([1, 1, 1])
        with i1:
            target_mode_label = st.selectbox(
                "Target mode",
                ["Target duration", "Liability"],
                index=0,
                key="ladder_target_mode",
            )
        if target_mode_label == "Target duration":
            with i2:
                target_duration = st.number_input(
                    "Target duration",
                    0.5,
                    30.0,
                    5.0,
                    step=0.25,
                    key="ladder_target_duration",
                )
            with i3:
                muted_note("Uses 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y candidates.")
        else:
            with i2:
                liability_date = st.date_input(
                    "Liability date",
                    min(end_date, start_date + pd.DateOffset(years=5)),
                    key="ladder_liability_date",
                )
            with i3:
                liability_amount = st.number_input(
                    "Liability amount ($)",
                    100_000,
                    100_000_000,
                    1_000_000,
                    step=100_000,
                    key="ladder_liability_amount",
                )

    r1, r2 = st.columns([2, 1])
    with r1:
        rungs = DEFAULT_IMMUNIZED_RUNGS if strategy_label == "Immunized" else DEFAULT_LADDER_RUNGS
        muted_note("Rungs: " + ", ".join(f"{r}Y" for r in rungs))
    with r2:
        run_btn = st.button("Run Ladder", type="primary", use_container_width=True, key="ladder_run")

    if run_btn:
        _run_ladder_workflow(
            start_date,
            end_date,
            capital,
            strategy_label,
            withdrawal_amount,
            withdrawal_frequency,
            first_withdrawal_date,
            target_mode_label,
            target_duration,
            liability_date,
            liability_amount,
        )

    if "ladder_results" not in st.session_state:
        st.info("Set ladder parameters and click Run Ladder.")
        return

    result = st.session_state["ladder_results"]
    ladder_result = result["ladder_result"]
    ts = ladder_result["portfolio_ts"]
    attribution = ladder_result["attribution"]
    summary = ladder_result.get("summary", {})
    strategy = result.get("strategy", "classic")

    section_header("Ladder Summary")
    final_value = ts["Portfolio Value"].iloc[-1]
    initial_value = ts["Portfolio Value"].iloc[0]
    total_return = ts["Cumulative Return"].iloc[-1]
    elapsed_years = max((pd.Timestamp(result["end"]) - pd.Timestamp(result["start"])).days / 365.25, 0.01)
    annualized_return = (1 + total_return) ** (1 / elapsed_years) - 1
    drawdown = ts["Cumulative Return"] - ts["Cumulative Return"].cummax()
    max_drawdown = drawdown.min()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Final Value", f"${final_value:,.0f}", delta=f"${final_value - initial_value:+,.0f}")
    m2.metric("Total Return", f"{total_return * 100:.2f}%")
    m3.metric("Annualized", f"{annualized_return * 100:.2f}%")
    m4.metric("Max Drawdown", f"{max_drawdown * 100:.2f}%")
    muted_note(f"Results shown for {result['start']} to {result['end']}.")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Cash Balance", f"${summary.get('Cash Balance', ts.get('Cash Balance', pd.Series([0])).iloc[-1]):,.0f}")
    s2.metric("Bond Market Value", f"${summary.get('Bond Market Value', ts.get('Bond Market Value', pd.Series([0])).iloc[-1]):,.0f}")
    s3.metric("Portfolio Duration", f"{summary.get('Portfolio Duration', 0):.2f}")
    if strategy == "withdrawal":
        s4.metric("Shortfall", f"${summary.get('Total Shortfall', 0):,.0f}")
    elif strategy == "immunized":
        match_ts = ladder_result.get("match_ts", pd.DataFrame())
        gap = match_ts["Duration Gap"].dropna().iloc[-1] if not match_ts.empty and match_ts["Duration Gap"].notna().any() else 0.0
        s4.metric("Duration Gap", f"{gap:.2f}")
    else:
        s4.metric("Total Par", f"${summary.get('Total Par', 0):,.0f}")

    st.plotly_chart(plot_ladder_portfolio(ts), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        if any(col in ts.columns and ts[col].abs().sum() > 0 for col in ["Coupon Cashflow", "Principal Cashflow", "Withdrawal Paid", "Withdrawal Shortfall"]):
            st.plotly_chart(plot_ladder_cashflows(ts), use_container_width=True)
        else:
            st.plotly_chart(plot_ladder_attribution(attribution), use_container_width=True)
    with c2:
        match_ts = ladder_result.get("match_ts", pd.DataFrame())
        if not match_ts.empty:
            st.plotly_chart(plot_ladder_duration_match(match_ts), use_container_width=True)
        else:
            st.plotly_chart(plot_ladder_waterfall(attribution), use_container_width=True)

    st.plotly_chart(plot_ladder_attribution(attribution), use_container_width=True)

    section_header("Current Holdings")
    holdings = ladder_result["holdings"]
    if holdings.empty:
        st.info("No active holdings at the end of the selected period.")
    else:
        st.dataframe(holdings, use_container_width=True, hide_index=True)

    with st.expander("Activity / Roll Log"):
        st.dataframe(ladder_result.get("trade_log", ladder_result["rebalance_log"]), use_container_width=True, hide_index=True)

    cashflows = ladder_result.get("cashflows", pd.DataFrame())
    if not cashflows.empty:
        with st.expander("Coupon and Principal Cashflows"):
            st.dataframe(cashflows, use_container_width=True, hide_index=True)

    withdrawals = ladder_result.get("withdrawals", pd.DataFrame())
    if not withdrawals.empty:
        with st.expander("Withdrawal Coverage"):
            st.dataframe(withdrawals, use_container_width=True, hide_index=True)

    rung_analytics = ladder_result.get("rung_analytics", pd.DataFrame())
    if strategy == "immunized" and not rung_analytics.empty:
        with st.expander("Immunized Rung Analytics", expanded=True):
            st.dataframe(rung_analytics, use_container_width=True, hide_index=True)
        target_log = ladder_result.get("target_log", pd.DataFrame())
        if not target_log.empty:
            with st.expander("Target Match Log"):
                st.dataframe(target_log, use_container_width=True, hide_index=True)


st.markdown(
    """
# Bond PnL Attribution System
U.S. Treasury fixed-rate bond analytics: yield-curve field, PnL attribution, and ladder backtest.
"""
)

field_tab, attribution_tab, ladder_tab = st.tabs(["Field", "Attribution", "Ladder"])

with field_tab:
    _render_field_tab()

with attribution_tab:
    _render_attribution_tab()

with ladder_tab:
    _render_ladder_tab()
