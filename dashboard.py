import json
import os
import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh

DB_FILE    = "drivesense.db"
STATE_FILE = "drivesense_state.json"

st.set_page_config(
    page_title="DriveSense Dashboard",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 DriveSense — Driver Monitoring System")

tab1, tab2 = st.tabs(["📡 Live Monitor", "📊 Weekly Report"])


# ── TAB 1: LIVE MONITOR ───────────────────────────────────────────────────────
with tab1:
    # Auto-refresh every 1 second
    st_autorefresh(interval=1000, key="live_refresh")

    st.subheader("Real-Time Driver Status")

    # ── Read live state written by main.py ────────────────────────────────────
    if not os.path.exists(STATE_FILE):
        st.info("⏳ Waiting for data — start `main.py` to begin monitoring.")
        st.stop()

    try:
        with open(STATE_FILE) as f:
            state = json.load(f)
    except Exception:
        st.warning("State file is being written — retrying...")
        st.stop()

    alert = state.get("alert_level", "NONE")

    # ── Alert banner ──────────────────────────────────────────────────────────
    if alert == "MULTIPLE":
        st.error("🚨 MULTIPLE ALERTS — Severe risk detected!")
    elif alert == "CRITICAL":
        st.error("🚨 CRITICAL ALERT — Immediate danger detected!")
    elif alert == "SOFT":
        st.warning("⚠️ SOFT ALERT — Mild fatigue or distraction detected.")
    else:
        st.success("✅ Driver is alert and focused.")

    # ── Metric cards ──────────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Alert Level", alert)
    col2.metric("EAR",         state.get("ear",       "—"))
    col3.metric("MAR",         state.get("mar",       "—"))
    col4.metric("Yaw Dev °",   state.get("yaw_dev",   "—"))
    col5.metric("Pitch Dev °", state.get("pitch_dev", "—"))

    # ── Signal flags ──────────────────────────────────────────────────────────
    st.markdown("**Active signals:**")
    flag_cols = st.columns(4)
    flags = {
        "😴 Drowsy":      state.get("drowsy",  False),
        "🥱 Yawning":     state.get("yawning", False),
        "↩️ Head Turned": state.get("head",    False),
        "😤 Emotion":     state.get("emotion_flag", False),
    }
    for col, (label, active) in zip(flag_cols, flags.items()):
        if active:
            col.error(label)
        else:
            col.success(label)

    st.caption(
        f"Emotion: **{state.get('emotion', '—')}** &nbsp;|&nbsp; "
        f"Last update: {state.get('timestamp', '—')}"
    )

    # ── Live history chart (last 100 DB rows) ─────────────────────────────────
    st.markdown("---")
    st.markdown("**Recent alert history**")
    try:
        conn    = sqlite3.connect(DB_FILE)
        df_hist = pd.read_sql_query(
            "SELECT timestamp, ear, mar, alert_level FROM events "
            "ORDER BY timestamp DESC LIMIT 100",
            conn,
        )
        conn.close()
    except Exception:
        df_hist = pd.DataFrame()

    if not df_hist.empty:
        df_hist = df_hist.iloc[::-1].reset_index(drop=True)

        # Map alert level strings to a numeric severity for the chart
        severity_map = {"NONE": 0, "SOFT": 1, "CRITICAL": 2, "MULTIPLE": 3}
        df_hist["severity"] = df_hist["alert_level"].map(severity_map).fillna(0)

        fig = px.line(
            df_hist, x="timestamp", y="severity",
            title="Alert Severity — Last 100 Logged Events",
            labels={"severity": "Severity (0=None … 3=Multiple)", "timestamp": "Time"},
            color_discrete_sequence=["#FF4B4B"],
        )
        fig.update_yaxes(tickvals=[0, 1, 2, 3],
                         ticktext=["None", "Soft", "Critical", "Multiple"])
        fig.add_hline(y=2, line_dash="dash", line_color="red",   annotation_text="Critical")
        fig.add_hline(y=1, line_dash="dash", line_color="orange", annotation_text="Soft")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No logged events yet.")


# ── TAB 2: WEEKLY REPORT ──────────────────────────────────────────────────────
with tab2:
    st.subheader("Weekly Driver Report")

    if st.button("🔄 Refresh report"):
        st.rerun()

    try:
        conn = sqlite3.connect(DB_FILE)
        df_raw = pd.read_sql_query(
            """
            SELECT
                DATE(timestamp)           AS date,
                COUNT(*)                  AS total_events,
                AVG(ear)                  AS avg_ear,
                AVG(mar)                  AS avg_mar,
                SUM(CASE WHEN alert_level IN ('CRITICAL','MULTIPLE') THEN 1 ELSE 0 END)
                                          AS critical_alerts,
                SUM(CASE WHEN alert_level = 'SOFT' THEN 1 ELSE 0 END)
                                          AS soft_alerts
            FROM events
            WHERE timestamp >= DATE('now', '-7 days')
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
            """,
            conn,
        )
        conn.close()
    except Exception as e:
        st.error(f"Could not read database: {e}")
        st.stop()

    if df_raw.empty:
        st.info("No data yet for the weekly report.")
    else:
        df_raw["avg_ear"] = df_raw["avg_ear"].round(3)
        df_raw["avg_mar"] = df_raw["avg_mar"].round(3)

        # ── Summary cards ─────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Events",    int(df_raw["total_events"].sum()))
        c2.metric("Critical Alerts", int(df_raw["critical_alerts"].sum()))
        c3.metric("Soft Alerts",     int(df_raw["soft_alerts"].sum()))
        c4.metric("Days Tracked",    len(df_raw))

        # ── Daily critical alert bar chart ────────────────────────────────────
        fig2 = px.bar(
            df_raw, x="date", y="critical_alerts",
            title="Critical Alerts per Day (Last 7 Days)",
            color="critical_alerts",
            color_continuous_scale="RdYlGn_r",
            labels={"critical_alerts": "Critical Alerts", "date": "Date"},
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ── EAR trend ─────────────────────────────────────────────────────────
        fig3 = px.line(
            df_raw.iloc[::-1], x="date", y="avg_ear",
            title="Average EAR per Day (lower = more drowsiness)",
            color_discrete_sequence=["#636EFA"],
            markers=True,
        )
        st.plotly_chart(fig3, use_container_width=True)

        # ── Full table ────────────────────────────────────────────────────────
        st.markdown("**Full weekly data**")
        st.dataframe(
            df_raw.rename(columns={
                "date":            "Date",
                "total_events":    "Total Events",
                "avg_ear":         "Avg EAR",
                "avg_mar":         "Avg MAR",
                "critical_alerts": "Critical Alerts",
                "soft_alerts":     "Soft Alerts",
            }),
            use_container_width=True,
        )