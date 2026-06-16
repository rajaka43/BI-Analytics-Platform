"""
app.py
─────────────────────────────────────────────────────────────────────────────
Real-Time Business Intelligence Dashboard — Streamlit frontend.

Run:  streamlit run app.py
      (data_generator.py must be running in a separate terminal)
"""

import sqlite3
import time
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from predictor import predict_next_hour

# ── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BI Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
      /* ── Global ── */
      @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

      html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
        background-color: #0d1117;
        color: #e6edf3;
      }

      /* ── Sidebar ── */
      section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #21262d;
      }

      /* ── KPI cards ── */
      .kpi-card {
        background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        position: relative;
        overflow: hidden;
      }
      .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #58a6ff, #3fb950);
        border-radius: 12px 12px 0 0;
      }
      .kpi-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #8b949e;
        margin-bottom: 6px;
        font-family: 'IBM Plex Mono', monospace;
      }
      .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #58a6ff;
        line-height: 1.1;
        font-family: 'IBM Plex Mono', monospace;
      }
      .kpi-delta {
        font-size: 0.78rem;
        color: #3fb950;
        margin-top: 4px;
      }

      /* ── Section headers ── */
      .section-header {
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: #8b949e;
        border-bottom: 1px solid #21262d;
        padding-bottom: 8px;
        margin: 24px 0 16px 0;
        font-family: 'IBM Plex Mono', monospace;
      }

      /* ── Live badge ── */
      .live-badge {
        display: inline-block;
        background: #238636;
        color: #3fb950;
        border: 1px solid #3fb950;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.7rem;
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.1em;
        animation: pulse 2s infinite;
      }
      @keyframes pulse {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.5; }
      }

      /* ── Plotly chart containers ── */
      .js-plotly-plot .plotly { background: transparent !important; }

      /* ── Hide Streamlit chrome ── */
      #MainMenu, footer, header { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constants ────────────────────────────────────────────────────────────────

DB_PATH        = "business_data.db"
REFRESH_SECS   = 4
CHART_BG       = "rgba(0,0,0,0)"
PAPER_BG       = "rgba(0,0,0,0)"
GRID_COLOR     = "#21262d"
TEXT_COLOR     = "#8b949e"
ACCENT_BLUE    = "#58a6ff"
ACCENT_GREEN   = "#3fb950"
ACCENT_ORANGE  = "#d29922"
ACCENT_PURPLE  = "#bc8cff"
ACCENT_RED     = "#f85149"

REGION_COLORS = {
    "North America":     ACCENT_BLUE,
    "Europe":            ACCENT_GREEN,
    "Asia Pacific":      ACCENT_PURPLE,
    "Latin America":     ACCENT_ORANGE,
    "Middle East & Africa": ACCENT_RED,
}

CATEGORY_COLORS = [
    "#58a6ff", "#3fb950", "#bc8cff", "#d29922",
    "#f85149", "#79c0ff", "#56d364", "#e3b341",
]

# ── Data helpers ─────────────────────────────────────────────────────────────

@st.cache_resource
def get_connection():
    """Persist a single SQLite connection across Streamlit reruns."""
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def load_transactions(window_minutes: int = 60) -> pd.DataFrame:
    """Fetch transactions from the last `window_minutes` minutes."""
    conn = get_connection()
    cutoff = (datetime.utcnow() - timedelta(minutes=window_minutes)).isoformat(
        sep=" ", timespec="seconds"
    )
    df = pd.read_sql_query(
        """
        SELECT timestamp, order_id, customer_region, product_category,
               sale_amount, quantity
        FROM   transactions
        WHERE  timestamp >= ?
        ORDER  BY timestamp
        """,
        conn,
        params=(cutoff,),
        parse_dates=["timestamp"],
    )
    return df


def load_all_transactions() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM transactions ORDER BY timestamp",
        conn,
        parse_dates=["timestamp"],
    )
    return df


# ── Chart helpers ─────────────────────────────────────────────────────────────

LAYOUT_BASE = dict(
    paper_bgcolor=PAPER_BG,
    plot_bgcolor=CHART_BG,
    font=dict(family="IBM Plex Mono, monospace", color=TEXT_COLOR, size=11),
    margin=dict(l=10, r=10, t=36, b=10),
    legend=dict(
        bgcolor="rgba(22,27,34,0.8)",
        bordercolor="#30363d",
        borderwidth=1,
    ),
    xaxis=dict(
        gridcolor=GRID_COLOR, zeroline=False, showline=False, tickcolor=TEXT_COLOR
    ),
    yaxis=dict(
        gridcolor=GRID_COLOR, zeroline=False, showline=False, tickcolor=TEXT_COLOR
    ),
)


def apply_layout(fig: go.Figure, title: str = "", height: int = 320) -> go.Figure:
    layout = {**LAYOUT_BASE, "height": height}
    if title:
        layout["title"] = dict(text=title, font=dict(size=13, color="#e6edf3"), x=0.0, xanchor="left")
    fig.update_layout(**layout)
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Controls")
    window = st.slider("Time window (minutes)", 5, 240, 60, step=5)
    show_forecast = st.toggle("Show Predictive Forecast", value=True)
    forecast_confidence = st.toggle("Show Confidence Band", value=True)
    st.divider()
    st.markdown("### 📡 Data Feed")
    st.markdown('<span class="live-badge">● LIVE</span>', unsafe_allow_html=True)
    refresh_ph = st.empty()
    st.divider()
    st.markdown(
        "<small style='color:#8b949e'>Data refreshes every "
        f"{REFRESH_SECS} s.<br>UTC timestamps.</small>",
        unsafe_allow_html=True,
    )

# ── Page header ───────────────────────────────────────────────────────────────

st.markdown(
    """
    <div style="display:flex; align-items:center; gap:14px; margin-bottom:4px;">
      <span style="font-size:2rem;">📊</span>
      <div>
        <div style="font-size:1.55rem; font-weight:700; color:#e6edf3; line-height:1.1;">
          Real-Time BI &amp; Analytics Platform
        </div>
        <div style="font-size:0.8rem; color:#8b949e; font-family:'IBM Plex Mono',monospace; margin-top:2px;">
          Live transaction intelligence · Predictive trend forecasting
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Main auto-refresh loop ────────────────────────────────────────────────────

placeholder = st.empty()

while True:
    df = load_transactions(window_minutes=window)
    now_str = datetime.utcnow().strftime("%H:%M:%S UTC")
    refresh_ph.markdown(f"<small style='color:#8b949e'>Last refresh: {now_str}</small>", unsafe_allow_html=True)

    with placeholder.container():

        # ── Guard: no data yet ────────────────────────────────────────────
        if df.empty:
            st.warning(
                "⏳  No data yet. Make sure `data_generator.py` is running "
                "and has produced at least one transaction."
            )
            time.sleep(REFRESH_SECS)
            continue

        # ── KPI computations ──────────────────────────────────────────────
        total_revenue = df["sale_amount"].sum()
        total_orders  = len(df)
        avg_order_val = df["sale_amount"].mean()
        total_qty     = df["quantity"].sum()

        # Compare to previous half-window for delta arrows
        half = window // 2
        cutoff_mid = datetime.utcnow() - timedelta(minutes=half)
        df_recent = df[df["timestamp"] >= cutoff_mid]
        df_older  = df[df["timestamp"] <  cutoff_mid]

        def pct_delta(new, old):
            if old == 0:
                return None
            return (new - old) / old * 100

        rev_delta = pct_delta(df_recent["sale_amount"].sum(), df_older["sale_amount"].sum())
        ord_delta = pct_delta(len(df_recent), len(df_older))

        # ── KPI row ───────────────────────────────────────────────────────
        st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)

        def delta_html(val):
            if val is None:
                return ""
            icon = "▲" if val >= 0 else "▼"
            color = "#3fb950" if val >= 0 else "#f85149"
            return f'<div class="kpi-delta" style="color:{color}">{icon} {abs(val):.1f}% vs prev period</div>'

        with k1:
            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-label">Total Revenue</div>'
                f'<div class="kpi-value">${total_revenue:,.0f}</div>'
                f'{delta_html(rev_delta)}'
                f'</div>',
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-label">Total Orders</div>'
                f'<div class="kpi-value">{total_orders:,}</div>'
                f'{delta_html(ord_delta)}'
                f'</div>',
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-label">Avg Order Value</div>'
                f'<div class="kpi-value">${avg_order_val:,.2f}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with k4:
            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-label">Units Sold</div>'
                f'<div class="kpi-value">{total_qty:,}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Row 1: Sales trend + Region bar ───────────────────────────────
        st.markdown('<div class="section-header">Live Sales Activity</div>', unsafe_allow_html=True)
        col_trend, col_region = st.columns([3, 2])

        with col_trend:
            # Resample to 1-minute buckets
            ts_df = (
                df.set_index("timestamp")["sale_amount"]
                .resample("1min")
                .sum()
                .reset_index()
            )
            ts_df.columns = ["time", "revenue"]

            fig_trend = go.Figure()
            fig_trend.add_trace(
                go.Scatter(
                    x=ts_df["time"],
                    y=ts_df["revenue"],
                    mode="lines",
                    name="Revenue / min",
                    line=dict(color=ACCENT_BLUE, width=2.5),
                    fill="tozeroy",
                    fillcolor="rgba(88,166,255,0.08)",
                )
            )
            apply_layout(fig_trend, "Revenue per Minute", height=310)
            fig_trend.update_xaxes(tickformat="%H:%M")
            fig_trend.update_yaxes(tickprefix="$", tickformat=",.0f")
            st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})

        with col_region:
            region_df = (
                df.groupby("customer_region")["sale_amount"]
                .sum()
                .reset_index()
                .sort_values("sale_amount", ascending=True)
            )
            colors_r = [
                REGION_COLORS.get(r, ACCENT_BLUE) for r in region_df["customer_region"]
            ]
            fig_region = go.Figure(
                go.Bar(
                    x=region_df["sale_amount"],
                    y=region_df["customer_region"],
                    orientation="h",
                    marker_color=colors_r,
                    text=region_df["sale_amount"].apply(lambda v: f"${v:,.0f}"),
                    textposition="outside",
                    textfont=dict(size=10),
                )
            )
            apply_layout(fig_region, "Revenue by Region", height=310)
            fig_region.update_xaxes(tickprefix="$", tickformat=",.0f")
            st.plotly_chart(fig_region, use_container_width=True, config={"displayModeBar": False})

        # ── Row 2: Category breakdown + Order volume heatmap ──────────────
        col_cat, col_scatter = st.columns([2, 3])

        with col_cat:
            cat_df = (
                df.groupby("product_category")["sale_amount"]
                .sum()
                .reset_index()
                .sort_values("sale_amount", ascending=False)
            )
            fig_cat = go.Figure(
                go.Bar(
                    x=cat_df["product_category"],
                    y=cat_df["sale_amount"],
                    marker_color=CATEGORY_COLORS[: len(cat_df)],
                    text=cat_df["sale_amount"].apply(lambda v: f"${v:,.0f}"),
                    textposition="outside",
                    textfont=dict(size=10),
                )
            )
            apply_layout(fig_cat, "Revenue by Category", height=300)
            fig_cat.update_xaxes(tickangle=-30)
            fig_cat.update_yaxes(tickprefix="$", tickformat=",.0f")
            st.plotly_chart(fig_cat, use_container_width=True, config={"displayModeBar": False})

        with col_scatter:
            fig_scatter = go.Figure()
            for i, (region, grp) in enumerate(df.groupby("customer_region")):
                fig_scatter.add_trace(
                    go.Scatter(
                        x=grp["timestamp"],
                        y=grp["sale_amount"],
                        mode="markers",
                        name=region,
                        marker=dict(
                            color=REGION_COLORS.get(region, ACCENT_BLUE),
                            size=6,
                            opacity=0.7,
                            line=dict(width=0),
                        ),
                    )
                )
            apply_layout(fig_scatter, "Transaction Scatter (by Region)", height=300)
            fig_scatter.update_xaxes(tickformat="%H:%M")
            fig_scatter.update_yaxes(tickprefix="$", tickformat=",.0f")
            st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})

        # ── Predictive Analytics section ──────────────────────────────────
        if show_forecast:
            st.markdown('<div class="section-header">Predictive Analytics — 60-Minute Forecast</div>', unsafe_allow_html=True)

            with st.spinner("Running forecast model …"):
                try:
                    forecast_df = predict_next_hour(DB_PATH)
                    hist_ts = (
                        load_all_transactions()
                        .set_index("timestamp")["sale_amount"]
                        .resample("1min")
                        .sum()
                        .reset_index()
                    )
                    hist_ts.columns = ["time", "revenue"]

                    fig_fc = go.Figure()

                    # Historical actuals
                    fig_fc.add_trace(
                        go.Scatter(
                            x=hist_ts["time"],
                            y=hist_ts["revenue"],
                            mode="lines",
                            name="Actual",
                            line=dict(color=ACCENT_BLUE, width=2),
                        )
                    )

                    # Confidence band
                    if forecast_confidence:
                        fig_fc.add_trace(
                            go.Scatter(
                                x=list(forecast_df["timestamp"]) + list(reversed(forecast_df["timestamp"])),
                                y=list(forecast_df["upper"]) + list(reversed(forecast_df["lower"])),
                                fill="toself",
                                fillcolor="rgba(188,140,255,0.12)",
                                line=dict(color="rgba(0,0,0,0)"),
                                name="Confidence Band",
                                showlegend=True,
                            )
                        )

                    # Forecast line
                    fig_fc.add_trace(
                        go.Scatter(
                            x=forecast_df["timestamp"],
                            y=forecast_df["predicted"],
                            mode="lines",
                            name="Forecast",
                            line=dict(color=ACCENT_PURPLE, width=2.5, dash="dot"),
                        )
                    )

                    # "Now" vertical line
                    now_line = dict(
                        type="line",
                        x0=datetime.utcnow(),
                        x1=datetime.utcnow(),
                        y0=0,
                        y1=1,
                        yref="paper",
                        line=dict(color=ACCENT_GREEN, width=1.5, dash="dash"),
                    )
                    apply_layout(fig_fc, "Actual Revenue vs 60-Min Forecast", height=360)
                    fig_fc.update_layout(shapes=[now_line])
                    fig_fc.update_xaxes(tickformat="%H:%M")
                    fig_fc.update_yaxes(tickprefix="$", tickformat=",.0f")
                    st.plotly_chart(fig_fc, use_container_width=True, config={"displayModeBar": False})

                    # Forecast summary table
                    with st.expander("📋 Forecast table (next 60 min)"):
                        display_df = forecast_df.copy()
                        display_df["timestamp"] = display_df["timestamp"].dt.strftime("%H:%M UTC")
                        display_df.columns = ["Time", "Predicted ($)", "Lower ($)", "Upper ($)"]
                        st.dataframe(display_df, use_container_width=True, hide_index=True)

                except Exception as e:
                    st.info(f"ℹ️  Forecast not available yet: {e}")

        # ── Recent transactions table ─────────────────────────────────────
        st.markdown('<div class="section-header">Recent Transactions (last 20)</div>', unsafe_allow_html=True)
        recent = df.tail(20).sort_values("timestamp", ascending=False).copy()
        recent["timestamp"] = recent["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        recent["sale_amount"] = recent["sale_amount"].apply(lambda v: f"${v:,.2f}")
        recent.columns = ["Timestamp", "Order ID", "Region", "Category", "Sale Amount", "Quantity"]
        st.dataframe(recent, use_container_width=True, hide_index=True)

    # ── Sleep then rerun ──────────────────────────────────────────────────
    time.sleep(REFRESH_SECS)
    st.rerun()
