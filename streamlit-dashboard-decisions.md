# Streamlit Dashboard Redesign Decisions

## Context

This document records the confirmed design decisions for modifying the Streamlit dashboard in this project.

Source inputs:

- `todo.md`
- Existing `dashboard.py`
- User decisions from the grill-me session

The target is to make the Streamlit app simpler, more user-facing, and better suited for interactive analysis.

## Confirmed Decisions

### 1. Top-Level Information Architecture

Use three top-level tabs only:

- `Field`
- `Attribution`
- `Ladder`

Remove the previous top-level tabs:

- `Overview`
- `Yield Curve`
- `Core Attribution`
- `PCA Analysis`
- `Ladder Backtest`
- `Comparison`

The previous `Core Attribution`, `PCA Analysis`, and `Comparison` content should be merged under the single `Attribution` tab.

### 2. Sidebar

Remove the sidebar.

Parameters should be placed directly inside the relevant tab. This keeps each workflow self-contained and avoids global controls whose effects are unclear.

### 3. Default Behavior

Do not auto-run a default case.

Inputs may have sensible prefilled values, but the app must not compute or display a default analysis until the user clicks the relevant run/load button.

### 4. Date Handling

Do not build cross-tab date synchronization.

Each tab should expose its own prefilled `Start` and `End` date inputs. Users manually modify dates in each tab and run that tab's computation.

Earlier ideas about `Field` syncing dates into other tabs are rejected to keep the implementation simple.

### 5. Overview Metrics

Remove the standalone `Overview` section.

Move metrics into the relevant tab:

- `Field`: field/date/yield-curve summary metrics where useful.
- `Attribution`: `Actual PnL`, `Market Impact`, `Time Impact`, `Residual`, `|Residual / Actual|`, and trading days.
- `Ladder`: `Final Value`, `Total Return`, `Annualized Return`, and `Max Drawdown`.

### 6. Attribution Internal Structure

Inside the `Attribution` tab, use an internal segmented view:

- `Core`
- `PCA`
- `Compare`

Prefer `st.segmented_control` if available. If the installed Streamlit version does not support it, use `st.radio(..., horizontal=True)` as the fallback.

### 7. Field Tab Interaction

The `Field` tab should center on the 3D yield-curve surface.

The desired interaction:

- User clicks a point on the 3D surface.
- The click identifies the selected date.
- The snapshot chart displays the full yield-curve cross-section for that date, from `1M` through `30Y`.

The selected tenor from the clicked point does not drive the snapshot. The snapshot is date-based only.

Provide a manual selected-date control as a fallback in case Plotly surface click/selection is unavailable or unstable in the local Streamlit environment.

### 8. Field Charts

Keep:

- 3D yield-curve surface.
- Dynamic snapshot chart for the selected date.

Remove:

- Yield level heatmap.
- Daily yield changes chart.

### 9. Core Attribution Charts

Keep:

- Cumulative PnL.
- Attribution waterfall.
- Cumulative components.
- Market vs Time chart.
- Aggregate summary table.
- Daily attribution table in an expander.

Remove:

- PnL distribution chart.

### 10. PCA Charts

Keep all current PCA views:

- PCA variance explained.
- PCA loadings line chart.
- PCA loadings heatmap.
- PCA-based PnL attribution.
- PCA attribution waterfall.
- Daily factor scores.
- PCA attribution table in an expander.

### 11. Compare View

Keep:

- Traditional attribution table.
- PCA-based attribution table.

Remove:

- Comparison bar chart.

### 12. Ladder Tab

Keep all current ladder views:

- Key ladder metrics.
- Portfolio value / cumulative return chart.
- Ladder attribution chart.
- Ladder waterfall.
- Holdings table.
- Rebalance log in an expander.

### 13. Streamlit Version And Plotly Selection

The local environment was observed to have:

- `streamlit 1.28.1`
- `plotly 5.9.0`

`streamlit 1.28.1` does not expose a stable `st.plotly_chart(..., on_select=...)` signature.

It is acceptable to update the project requirement for Streamlit to a newer version that supports Plotly chart selection events, but the implementation must still include a manual selected-date fallback.

### 14. Preserve Local Anaconda Streamlit Entry Fix

Do not modify:

```text
d:\Application\anaconda3\Scripts\streamlit-script.py
```

The user specifically noted that the bug fix in that file must be preserved.

That file currently imports:

```python
from streamlit.web.cli import main
```

Project changes should not overwrite this environment-level script.

### 15. Run Command

Update project documentation and user-facing run guidance to prefer:

```powershell
python -m streamlit run dashboard.py
```

instead of:

```powershell
streamlit run dashboard.py
```

This avoids relying on the Anaconda-generated `streamlit-script.py` entry script.

## Implementation Direction

The expected implementation should:

- Refactor `dashboard.py` from sidebar-first layout to tab-local parameter panels.
- Preserve existing calculation functions where practical.
- Keep cache behavior where useful.
- Avoid complex cross-tab state synchronization.
- Keep the UX simple: prefilled controls, explicit run buttons, no implicit default analysis.
- Use the repo's existing plotting and computation helpers where possible.
- Add or adjust tests only where behavior changes are testable without requiring Streamlit browser automation.
- Run the existing test suite after changes.
