"""Pipeline Sentinel — Streamlit Monitoring Dashboard."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sentinel.models import database as db

st.set_page_config(
    page_title="Pipeline Sentinel",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Pipeline Sentinel")
st.caption("AI-Powered Data Pipeline Health Monitor")


# --- Initialize ---
db.init_db()


# --- Sidebar ---
st.sidebar.header("Filters")
pipelines = db.list_pipelines()
pipeline_names = {p["id"]: p["name"] for p in pipelines}
selected = st.sidebar.selectbox(
    "Pipeline",
    ["All"] + list(pipeline_names.keys()),
    format_func=lambda x: "All Pipelines" if x == "All" else pipeline_names.get(x, x)
)


# --- Health Summary ---
st.header("Pipeline Health Overview")
health = db.get_pipeline_health_summary()

if health:
    cols = st.columns(4)
    total = len(health)
    healthy = sum(1 for h in health if h["failure_count"] == 0 and h["anomaly_count"] == 0)
    warning = sum(1 for h in health if h["anomaly_count"] > 0 and h["failure_count"] == 0)
    failing = sum(1 for h in health if h["failure_count"] > 0)

    cols[0].metric("Total Pipelines", total)
    cols[1].metric("Healthy", healthy, delta=None)
    cols[2].metric("Warning", warning, delta=None)
    cols[3].metric("Failing", failing, delta=None)

    # Health table
    health_df = pd.DataFrame(health)
    health_df["status"] = health_df.apply(
        lambda r: "🔴 Failing" if r["failure_count"] > 0
        else "🟡 Warning" if r["anomaly_count"] > 0
        else "🟢 Healthy",
        axis=1
    )
    health_df["avg_latency_ms"] = health_df["avg_latency_ms"].round(0)

    st.dataframe(
        health_df[["name", "status", "success_count", "failure_count",
                    "warning_count", "anomaly_count", "avg_latency_ms", "last_run"]],
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No pipeline data yet. Run the simulator first: `python -m simulations.pipeline_simulator`")


# --- Anomaly Feed ---
st.header("Recent Anomalies")

anomaly_filter = selected if selected != "All" else None
anomalies = db.get_anomalies(pipeline_id=anomaly_filter, limit=20)

if anomalies:
    for a in anomalies:
        severity_icon = {"low": "🔵", "medium": "🟡", "high": "🟠", "critical": "🔴"}.get(a["severity"], "⚪")
        pipeline_name = pipeline_names.get(a["pipeline_id"], a["pipeline_id"])

        with st.expander(
            f"{severity_icon} [{a['severity'].upper()}] {pipeline_name} — {a['metric_name']}",
            expanded=a["severity"] in ("high", "critical")
        ):
            col1, col2, col3 = st.columns(3)
            col1.metric("Expected", f"{a['expected_value']:.2f}" if a["expected_value"] else "N/A")
            col2.metric("Actual", f"{a['actual_value']:.2f}" if a["actual_value"] else "N/A")
            col3.metric("Deviation", f"{a['deviation_score']:.3f}" if a["deviation_score"] else "N/A")

            st.text(f"Type: {a['anomaly_type']} | Detected: {a['detected_at']}")
            st.text(f"Resolved: {'Yes' if a['resolved'] else 'No'}")

            if a["root_cause"]:
                st.markdown(f"**Root Cause:** {a['root_cause']}")
            if a["remediation"]:
                st.markdown(f"**Remediation:** {a['remediation']}")
else:
    st.success("No anomalies detected!")


# --- Pipeline Detail View ---
if selected != "All":
    st.header(f"Pipeline: {pipeline_names.get(selected, selected)}")

    runs = db.get_recent_runs(selected, 100)

    if runs:
        runs_df = pd.DataFrame(runs)

        # Row count trend
        if "row_count" in runs_df.columns:
            fig_rows = px.line(
                runs_df, x="started_at", y="row_count",
                title="Row Count Over Time",
                color_discrete_sequence=["#4F46E5"]
            )
            fig_rows.update_layout(height=300)
            st.plotly_chart(fig_rows, use_container_width=True)

        # Latency trend
        col_a, col_b = st.columns(2)

        if "latency_ms" in runs_df.columns:
            fig_latency = px.line(
                runs_df, x="started_at", y="latency_ms",
                title="Latency (ms)",
                color_discrete_sequence=["#F59E0B"]
            )
            fig_latency.update_layout(height=250)
            col_a.plotly_chart(fig_latency, use_container_width=True)

        if "null_percentage" in runs_df.columns:
            fig_nulls = px.line(
                runs_df, x="started_at", y="null_percentage",
                title="Null Percentage",
                color_discrete_sequence=["#EF4444"]
            )
            fig_nulls.update_layout(height=250)
            col_b.plotly_chart(fig_nulls, use_container_width=True)

        # Status distribution
        status_counts = runs_df["status"].value_counts()
        fig_status = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Run Status Distribution",
            color_discrete_map={
                "success": "#22C55E", "failure": "#EF4444",
                "warning": "#F59E0B", "running": "#3B82F6"
            }
        )
        fig_status.update_layout(height=300)
        st.plotly_chart(fig_status, use_container_width=True)


# --- Lineage Graph ---
st.header("Pipeline Lineage")

edges = db.get_all_edges()
if edges:
    import networkx as nx

    G = nx.DiGraph()
    for e in edges:
        G.add_edge(e["source_pipeline"], e["target_pipeline"])

    # Simple text-based lineage display
    st.text("Dependency Graph:")
    for src, tgt in G.edges():
        src_name = pipeline_names.get(src, src)
        tgt_name = pipeline_names.get(tgt, tgt)
        st.text(f"  {src_name} → {tgt_name}")
else:
    st.info("No lineage data available.")


# --- Footer ---
st.divider()
st.caption("Pipeline Sentinel v0.1.0 | Refresh page for latest data")
