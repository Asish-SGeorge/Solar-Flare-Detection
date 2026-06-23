import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import os

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Solar Flare Forecasting",
    page_icon="☀️",
    layout="wide"
)

st.title("☀️ Solar Flare Forecasting using Aditya-L1")
st.markdown(
"""
Forecasting Solar Flares using combined
**SoLEXS (Soft X-rays)** and
**HEL1OS (Hard X-rays)** observations from Aditya-L1 ISRO Mission.
"""
)

# ISRO mission control theme
st.markdown("""
<style>
.stApp{background-color:#050816;color:white;}
[data-testid="stMetricValue"]{font-size:34px;color:#00E5FF;}
.stButton>button{background-color:#003366;color:white}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():
    pred = pd.read_csv("Prediction_Test_Period.csv")
    warn = pd.read_csv("Warning_Events.csv")
    flare = pd.read_csv("Unified_Flare_Catalog.csv")

    pred["Time"] = pd.to_datetime(pred["Time"])
    warn["Warning_Time"] = pd.to_datetime(warn["Warning_Time"])
    flare["SoLEXS_Time"] = pd.to_datetime(flare["SoLEXS_Time"])

    start = pred["Time"].min()
    end = pred["Time"].max()

    flare = flare[
        (flare["SoLEXS_Time"] >= start) &
        (flare["SoLEXS_Time"] <= end)
    ]

    return pred, warn, flare

df, warnings, flares = load_data()

# === Merge lead time data (if available) ===
try:
    if os.path.exists("Lead_Time_Analysis.csv"):
        lead = pd.read_csv("Lead_Time_Analysis.csv")
        # normalize column names
        lead_cols = {c: c.strip() for c in lead.columns}
        lead.rename(columns=lead_cols, inplace=True)
        if "Warning_Time" in lead.columns:
            lead["Warning_Time"] = pd.to_datetime(lead["Warning_Time"])            
            # detect any lead-time-like column automatically (e.g., Lead_Time, Lead_Minutes, Lead Time)
            lead_col = None
            for c in lead.columns:
                if c.lower() == "warning_time":
                    continue
                if "lead" in c.lower():
                    lead_col = c
                    break

            # prepare merge if we found a lead column
            if lead_col is not None:
                # create numeric Lead_Time (minutes)
                lead["Lead_Time"] = pd.to_numeric(lead[lead_col], errors="coerce")
                # if numeric conversion gave NaN, try timedelta parsing (e.g., '0 days 00:14:04')
                if lead["Lead_Time"].isna().all():
                    try:
                        td = pd.to_timedelta(lead[lead_col])
                        lead["Lead_Time"] = td.dt.total_seconds() / 60.0
                    except Exception:
                        # as last resort, if there's a 'Lead_Minutes' column, use it
                        if "Lead_Minutes" in lead.columns:
                            lead["Lead_Time"] = pd.to_numeric(lead["Lead_Minutes"], errors="coerce")

                # include Flare_Time if present
                if "Flare_Time" in lead.columns:
                    lead["Flare_Time"] = pd.to_datetime(lead["Flare_Time"])

                # merge and coerce
                lead_merge_cols = [c for c in ["Warning_Time", "Lead_Time", "Flare_Time"] if c in lead.columns]
                if len(lead_merge_cols) > 1:
                    warnings = warnings.merge(lead[lead_merge_cols], on="Warning_Time", how="left")
                    if "Lead_Time" in warnings.columns:
                        warnings["Lead_Time"] = pd.to_numeric(warnings["Lead_Time"], errors="coerce")
except Exception:
    pass

# === Load validation/metrics (TP/FP/FN) as dataframes if available ===
tp_df = pd.DataFrame()
fp_df = pd.DataFrame()
fn_df = pd.DataFrame()
try:
    if os.path.exists("True_Positive_Warnings.csv"):
        tp_df = pd.read_csv("True_Positive_Warnings.csv")
        if "Warning_Time" in tp_df.columns:
            tp_df["Warning_Time"] = pd.to_datetime(tp_df["Warning_Time"])
    if os.path.exists("False_Positive_Warnings.csv"):
        fp_df = pd.read_csv("False_Positive_Warnings.csv")
        if "Warning_Time" in fp_df.columns:
            fp_df["Warning_Time"] = pd.to_datetime(fp_df["Warning_Time"])
    if os.path.exists("Missed_Flares.csv"):
        fn_df = pd.read_csv("Missed_Flares.csv")
        # Missed_Flares may have Flare_Time; attempt to parse any time-like columns
        for c in ["Flare_Time", "Warning_Time"]:
            if c in fn_df.columns:
                fn_df[c] = pd.to_datetime(fn_df[c])
except Exception:
    tp_df = fp_df = fn_df = pd.DataFrame()

# If validation files are not present, stop and notify
if tp_df.empty and fp_df.empty and fn_df.empty:
    st.error("Validation result files not found. Please add True_Positive_Warnings.csv, False_Positive_Warnings.csv, and Missed_Flares.csv to the workspace.")
    st.stop()

# Compute canonical metrics from validation results (global)
tp = len(tp_df)
fp = len(fp_df)
fn = len(fn_df)
precision_from_validation = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall_from_validation = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1_from_validation = 2 * precision_from_validation * recall_from_validation / (precision_from_validation + recall_from_validation) if (precision_from_validation + recall_from_validation) > 0 else 0.0

# ==========================================================
# SIDEBAR - INTERACTIVE CONTROLS
# ==========================================================

# Sidebar: mission info and fixed operational parameters (validated)
st.sidebar.header("Mission Information")
THRESHOLD = 0.60
PERSISTENCE = 60  # seconds
COOLDOWN = 30  # minutes
WARNING_WINDOW = 30  # minutes

st.sidebar.markdown(f"""
**Mission:** Aditya-L1  
**Payloads:** SoLEXS + HEL1OS  
**Model:** XGBoost  
**Features:** 63  
**Threshold:** {THRESHOLD:.2f}  
**Persistence:** {PERSISTENCE} sec  
**Cooldown:** {COOLDOWN} min  
**Warning Window:** {WARNING_WINDOW} min
""")

# Date range filter (kept for exploration)
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(df["Time"].min().date(), df["Time"].max().date()),
    min_value=df["Time"].min().date(),
    max_value=df["Time"].max().date(),
    key="date_range"
)

if len(date_range) == 2:
    start_date, end_date = date_range
    mask = (df["Time"].dt.date >= start_date) & (df["Time"].dt.date <= end_date)
    df_filtered = df[mask].reset_index(drop=True)
else:
    df_filtered = df.copy()

# Filter warnings and flares by date (used in KPIs)
warnings_filtered = warnings[
    (warnings["Warning_Time"].dt.date >= date_range[0]) &
    (warnings["Warning_Time"].dt.date <= date_range[1])
] if len(date_range) == 2 else warnings

flares_filtered = flares[
    (flares["SoLEXS_Time"].dt.date >= date_range[0]) &
    (flares["SoLEXS_Time"].dt.date <= date_range[1])
] if len(date_range) == 2 else flares

# Use validated threshold constant everywhere
threshold = THRESHOLD
# Always show elements in production-like dashboard
show_warnings = True
show_flares = True
show_threshold_line = True
selected_charts = ["SoLEXS", "HEL1OS", "Forecast Probability"]

# ==========================================================
# DYNAMIC METRICS (Based on threshold)
# ==========================================================

st.markdown("### 📊 Forecast KPIs")

# Calculate warnings relative to threshold
warnings_above_threshold = df_filtered[df_filtered["Probability"] >= threshold]

# Use validation-derived metrics (tp/fp/fn) if available
detected_flares = tp
false_positives = fp
missed_flares = fn

# Canonical metrics (from validation)
precision = precision_from_validation
recall = recall_from_validation
f1 = f1_from_validation

# Latest forecast probability and status badge
current_prob = df_filtered["Probability"].iloc[-1] if len(df_filtered) > 0 else 0.0

col_a, col_b, col_c, col_d = st.columns([2, 1, 1, 1])

with col_a:
    st.metric("Forecast Probability", f"{current_prob:.2f}")
    if current_prob >= threshold:
        st.error("🔴 HIGH SOLAR FLARE RISK")
    else:
        st.success("🟢 LOW SOLAR FLARE RISK")
    # Gauge chart
    try:
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=float(current_prob),
                number={'valueformat':'.2f'},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': 'red'},
                    'steps': [
                        {'range': [0, 0.6], 'color': 'green'},
                        {'range': [0.6, 0.8], 'color': 'orange'},
                        {'range': [0.8, 1], 'color': 'red'},
                    ],
                }
            )
        )
        fig_gauge.update_layout(template='plotly_dark', height=200, margin={'t':10,'b':10,'l':10,'r':10}, paper_bgcolor='#050816', plot_bgcolor='#050816')
        st.plotly_chart(fig_gauge, width="stretch")
    except Exception:
        pass

with col_b:
    st.markdown("**Warning Events**")
    st.metric("Count", f"{len(warnings_filtered)}")

with col_c:
    st.markdown("**Detected Flares**")
    total_observed = tp + fn if (tp + fn) > 0 else 0
    st.metric("Detected", f"{tp} / {total_observed}")

with col_d:
    st.markdown("**Average Lead Time**")
    if "Lead_Time" in warnings.columns:
        avg_lead_series = pd.to_numeric(warnings["Lead_Time"], errors="coerce").dropna()
        avg_lead = avg_lead_series.mean() if len(avg_lead_series) > 0 else None
    else:
        avg_lead = None
    st.metric("Avg", f"{avg_lead:.2f} min" if avg_lead is not None else "N/A")

# Secondary small KPIs
col1, col2 = st.columns(2)
col1.metric("Precision", f"{precision:.3f}")
col2.metric("Recall", f"{recall:.3f}")

# Mission information card
st.markdown("**Mission**")
mis_col1, mis_col2 = st.columns([2, 3])
with mis_col1:
    st.info("""
**Aditya-L1**  
**Payloads:** SoLEXS, HEL1OS  
**Model:** XGBoost  
**Features:** 63
""")
with mis_col2:
    st.markdown("""
**Operational Settings**  
- Threshold: 0.60  
- Persistence: 60 sec  
- Cooldown: 30 min  
- Warning Window: 30 min
""")

# spacer
st.markdown("---")

# ==========================================================
# INTERACTIVE CHARTS (Conditional Display)
# ==========================================================

# SOLEXS Chart
if "SoLEXS" in selected_charts:
    fig1 = go.Figure()

    fig1.add_trace(
        go.Scatter(
            x=df_filtered["Time"],
            y=df_filtered["SoLEXS_Count"],
            mode="lines",
            line=dict(color="blue", width=2),
            name="SoLEXS",
            hovertemplate="<b>Time:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>"
        )
    )

    if show_warnings:
        for t in warnings_filtered["Warning_Time"]:
            fig1.add_vline(
                x=t,
                line_dash="dash",
                line_color="orange",
                opacity=0.6
            )

    if show_flares:
        for t in flares_filtered["SoLEXS_Time"]:
            fig1.add_vline(
                x=t,
                line_color="red",
                opacity=0.9
            )

    fig1.update_layout(
        title="SoLEXS Light Curve (Soft X-rays)",
        height=400,
        hovermode="x unified",
        xaxis_title="Time",
        yaxis_title="Count",
        template="plotly_dark",
        paper_bgcolor="#050816",
        plot_bgcolor="#050816"
    )

    st.plotly_chart(fig1, width="stretch")

# HEL1OS Chart
if "HEL1OS" in selected_charts:
    fig2 = go.Figure()

    fig2.add_trace(
        go.Scatter(
            x=df_filtered["Time"],
            y=df_filtered["HEL1OS_Count"],
            mode="lines",
            line=dict(color="green", width=2),
            name="HEL1OS",
            hovertemplate="<b>Time:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>"
        )
    )

    if show_warnings:
        for t in warnings_filtered["Warning_Time"]:
            fig2.add_vline(
                x=t,
                line_dash="dash",
                line_color="orange",
                opacity=0.6
            )

    if show_flares:
        for t in flares_filtered["SoLEXS_Time"]:
            fig2.add_vline(
                x=t,
                line_color="red",
                opacity=0.9
            )

    fig2.update_layout(
        title="HEL1OS Light Curve (Hard X-rays)",
        height=400,
        hovermode="x unified",
        xaxis_title="Time",
        yaxis_title="Count",
        template="plotly_dark",
        paper_bgcolor="#050816",
        plot_bgcolor="#050816"
    )

    st.plotly_chart(fig2, width="stretch")

# Forecast Probability Chart
if "Forecast Probability" in selected_charts:
    fig3 = go.Figure()

    fig3.add_trace(
        go.Scatter(
            x=df_filtered["Time"],
            y=df_filtered["Probability"],
            mode="lines",
            line=dict(color="red", width=2),
            name="Probability",
            fill="tozeroy",
            fillcolor="rgba(255,0,0,0.1)",
            hovertemplate="<b>Time:</b> %{x}<br><b>Probability:</b> %{y:.3f}<extra></extra>"
        )
    )

    if show_threshold_line:
        fig3.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="black",
            line_width=2,
            annotation_text=f"Threshold: {threshold:.2f}",
            annotation_position="right"
        )
        # shade forecast warning region
        fig3.add_hrect(
            y0=threshold,
            y1=1,
            fillcolor="orange",
            opacity=0.12,
            line_width=0
        )

    if show_warnings:
        for t in warnings_filtered["Warning_Time"]:
            fig3.add_vline(
                x=t,
                line_dash="dash",
                line_color="orange",
                opacity=0.6
            )

    if show_flares:
        for t in flares_filtered["SoLEXS_Time"]:
            fig3.add_vline(
                x=t,
                line_color="red",
                opacity=0.9
            )

    # Highlight matched warning -> flare intervals (using Lead_Time or Flare_Time)
    try:
        for _, row in warnings_filtered.iterrows():
            wt = row.get("Warning_Time")
            if pd.isna(wt):
                continue
            ft = None
            if "Flare_Time" in row.index and pd.notna(row.get("Flare_Time", None)):
                ft = row.get("Flare_Time")
            elif "Lead_Time" in row.index and pd.notna(row.get("Lead_Time", None)):
                try:
                    lead_min = float(row.get("Lead_Time"))
                    ft = wt + pd.to_timedelta(lead_min, unit='m')
                except Exception:
                    ft = None
            if ft is not None and ft > wt:
                fig3.add_vrect(x0=wt, x1=ft, fillcolor='green', opacity=0.06, line_width=0)
                # annotate mid-point with lead time if available
                try:
                    lead_minutes = (ft - wt) / pd.Timedelta(minutes=1)
                    ann_x = wt + (ft - wt) / 2
                    ann_y = min(0.95, threshold + 0.08)
                    fig3.add_annotation(x=ann_x, y=ann_y, text=f"{lead_minutes:.1f} min", showarrow=False, font=dict(color='white'))
                except Exception:
                    pass
    except Exception:
        pass

    fig3.update_layout(
        title="Forecast Probability with Dynamic Threshold",
        yaxis=dict(range=[0, 1]),
        height=400,
        hovermode="x unified",
        xaxis_title="Time",
        yaxis_title="Probability",
        template="plotly_dark",
        paper_bgcolor="#050816",
        plot_bgcolor="#050816"
    )

    st.plotly_chart(fig3, width="stretch")

# ==========================================================
# TABBED DATA EXPLORER
# ==========================================================

tab1, tab2, tab3, tab4 = st.tabs(["📋 Warning Events", "🔴 Flare Events", "📈 Detailed Analysis", "📥 Downloads"])

with tab1:
    st.subheader("Warning Events")
    
    if len(warnings_filtered) > 0:
        # Search/filter functionality
        col_search, col_count = st.columns([3, 1])
        with col_search:
            search_term = st.text_input("🔍 Search warnings", placeholder="Filter by any column...")
        with col_count:
            st.metric("Total Warnings", len(warnings_filtered))
        
        # Display filtered data
        if search_term:
            mask = warnings_filtered.astype(str).apply(lambda x: x.str.contains(search_term, case=False)).any(axis=1)
            display_warnings = warnings_filtered[mask]
        else:
            display_warnings = warnings_filtered
        
        st.dataframe(
            display_warnings,
            width="stretch",
            height=400
        )
        
        # Show simplified validated statistics
        if len(display_warnings) > 0:
            col1, col2, col3 = st.columns(3)
            # Total warnings in selected range
            total_warnings = len(display_warnings)
            # True/False positive counts within the selected date range (match by Warning_Time)
            tp_count = 0
            fp_count = 0
            try:
                if not tp_df.empty and 'Warning_Time' in tp_df.columns:
                    tp_count = len(display_warnings.merge(tp_df[['Warning_Time']], on='Warning_Time', how='inner'))
                if not fp_df.empty and 'Warning_Time' in fp_df.columns:
                    fp_count = len(display_warnings.merge(fp_df[['Warning_Time']], on='Warning_Time', how='inner'))
            except Exception:
                tp_count = tp
                fp_count = fp

            col1.metric("Total Warning Events", f"{total_warnings}")
            col2.metric("True Positive Warnings", f"{tp_count}")
            col3.metric("False Positive Warnings", f"{fp_count}")
    else:
        st.info("No warnings in selected date range")

with tab2:
    st.subheader("Observed Flare Events")
    
    if len(flares_filtered) > 0:
        # Search/filter functionality
        col_search, col_count = st.columns([3, 1])
        with col_search:
            search_term = st.text_input("🔍 Search flares", placeholder="Filter by any column...")
        with col_count:
            st.metric("Total Flares", len(flares_filtered))
        
        # Display filtered data
        if search_term:
            mask = flares_filtered.astype(str).apply(lambda x: x.str.contains(search_term, case=False)).any(axis=1)
            display_flares = flares_filtered[mask]
        else:
            display_flares = flares_filtered
        
        st.dataframe(
            display_flares,
            width="stretch",
            height=400
        )
        
        # Show statistics
        if len(display_flares) > 0:
            col1, col2, col3 = st.columns(3)
            if 'Class' in display_flares.columns:
                class_dist = display_flares['Class'].value_counts()
                col1.metric("Flare Class (Most Common)", class_dist.index[0] if len(class_dist) > 0 else "N/A")
            if 'SoLEXS_Time' in display_flares.columns:
                col2.metric("Date Range", f"{display_flares['SoLEXS_Time'].min().date()} to {display_flares['SoLEXS_Time'].max().date()}")
            col3.metric("Flares Shown", len(display_flares))
    else:
        st.info("No flares in selected date range")

with tab3:
    st.subheader("Detailed Analysis")
    
    analysis_col1, analysis_col2 = st.columns(2)
    
    with analysis_col1:
        st.markdown("#### Prediction Summary")
        st.info(f"""
        **Threshold:** {threshold:.2f}  
        **True Positives:** {detected_flares}  
        **False Positives:** {false_positives}  
        **False Negatives (Missed):** {missed_flares}  
        """)
    
    with analysis_col2:
        st.markdown("#### Performance Breakdown")
        # Performance metrics from validation
        performance_data = {
            "Metric": ["Precision", "Recall", "F1 Score"],
            "Value": [
                precision,
                recall,
                f1
            ]
        }
        perf_df = pd.DataFrame(performance_data)
        st.bar_chart(perf_df.set_index("Metric"))
        # Confusion matrix (TP/FP/FN/TN if available)
        tn = max(0, len(df_filtered) - (tp + fp + fn))
        cm_z = [[tn, fp], [fn, tp]]
        cm = go.Figure(data=go.Heatmap(z=cm_z, x=["Pred No","Pred Yes"], y=["Actual No","Actual Yes"], colorscale="Viridis"))
        cm.update_layout(title="Confusion Matrix", template='plotly_dark', paper_bgcolor='#050816', plot_bgcolor='#050816', height=300)
        st.plotly_chart(cm, width="stretch")
    
    st.markdown("---")
    
    # Expandable sections for more details
    with st.expander("📊 Show Probability Distribution"):
        prob_col1, prob_col2 = st.columns(2)
        with prob_col1:
            st.metric("Max Probability", f"{df_filtered['Probability'].max():.3f}")
        with prob_col2:
            st.metric("Mean Probability", f"{df_filtered['Probability'].mean():.3f}")
        
        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Histogram(
                x=df_filtered["Probability"],
                nbinsx=50,
                name="Probability Distribution",
                marker_color="blue"
            )
        )
        fig_hist.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="red",
            opacity=0.8
        )
        fig_hist.update_layout(
            title="Distribution of Forecast Probabilities",
            xaxis_title="Probability",
            yaxis_title="Frequency",
            height=350,
            template="plotly_dark",
            paper_bgcolor="#050816",
            plot_bgcolor="#050816"
        )
        st.plotly_chart(fig_hist, width="stretch")
    
    with st.expander("⏱️ Show Lead Time Analysis"):
        if "Lead_Time" in warnings_filtered.columns and len(warnings_filtered) > 0:
            lead_times = pd.to_numeric(warnings_filtered["Lead_Time"], errors="coerce").dropna()
            if len(lead_times) > 0:
                col1, col2, col3 = st.columns(3)
                col1.metric("Average Lead Time", f"{lead_times.mean():.2f} min")
                col2.metric("Max Lead Time", f"{lead_times.max():.2f} min")
                col3.metric("Min Lead Time", f"{lead_times.min():.2f} min")
                
                fig_lead = go.Figure()
                fig_lead.add_trace(
                    go.Box(y=lead_times, name="Lead Time", marker_color="green")
                )
                fig_lead.update_layout(
                    title="Lead Time Distribution",
                    yaxis_title="Minutes",
                    height=300,
                    template="plotly_dark",
                    paper_bgcolor="#050816",
                    plot_bgcolor="#050816"
                )
                st.plotly_chart(fig_lead, width="stretch")
        else:
            st.info("Lead time data not available")

with tab4:
    st.subheader("Downloads")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            "📥 Download Filtered Predictions",
            df_filtered.to_csv(index=False),
            f"Predictions_{date_range[0]}_to_{date_range[1]}.csv",
            "text/csv",
            key="download_predictions"
        )
    
    with col2:
        st.download_button(
            "📥 Download Filtered Warnings",
            warnings_filtered.to_csv(index=False),
            f"Warnings_{date_range[0]}_to_{date_range[1]}.csv",
            "text/csv",
            key="download_warnings"
        )
    
    with col3:
        st.download_button(
            "📥 Download Filtered Flares",
            flares_filtered.to_csv(index=False),
            f"Flares_{date_range[0]}_to_{date_range[1]}.csv",
            "text/csv",
            key="download_flares"
        )
    
    st.markdown("---")
    
    # Summary statistics export
    if st.button("📊 Generate Summary Report"):
        summary_report = f"""
# Solar Flare Forecasting Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Filter Settings
- Date Range: {date_range[0]} to {date_range[1]}
- Probability Threshold: {threshold:.3f}

## Performance Metrics
- Precision: {precision:.3f}
- Recall: {recall:.3f}
- F1 Score: {f1:.3f}
- True Positives: {detected_flares}
- False Positives: {false_positives}
- False Negatives: {missed_flares}

## Data Summary
- Total Predictions: {len(df_filtered)}
- Warnings Generated: {len(warnings_filtered)}
- Observed Flares: {len(flares_filtered)}

---
*Generated by Solar Flare Forecasting System*
        """
        st.download_button(
            "📄 Download Summary Report",
            summary_report,
            f"Report_{date_range[0]}_to_{date_range[1]}.md",
            "text/markdown",
            key="download_report"
        )

# ==========================================================
# FOOTER
# ==========================================================

st.divider()
st.markdown("""
---
### 📖 About This Dashboard
- **SoLEXS:** Soft X-ray measurements for solar activity monitoring
- **HEL1OS:** Hard X-ray observations for flare detection
- **Interactive Features:** Use the sidebar to customize threshold, date range, and chart display
- **Dynamic Metrics:** All metrics update based on your selected threshold and date range
- **Data Export:** Download predictions, warnings, and analysis reports

**Project:** Aditya-L1 Solar Flare Forecasting System | ISRO Mission
""")
# Footer model performance (from validation)
avg_lead_global = None
max_lead_global = None
if "Lead_Time" in warnings.columns:
    lead_vals = pd.to_numeric(warnings["Lead_Time"], errors="coerce").dropna()
    if len(lead_vals) > 0:
        avg_lead_global = lead_vals.mean()
        max_lead_global = lead_vals.max()

st.markdown("""
**Model Performance (validation)**
""")
st.markdown(f"- Model: XGBoost  \n- Features: 63  \n- Precision: {precision:.3f}  \n- Recall: {recall:.3f}  \n- F1 Score: {f1:.3f}  \n- Average Lead Time: {avg_lead_global:.2f} min" if avg_lead_global is not None else f"- Model: XGBoost  \n- Features: 63  \n- Precision: {precision:.3f}  \n- Recall: {recall:.3f}  \n- F1 Score: {f1:.3f}")

st.success("✅ Forecasting Pipeline Completed Successfully - Ready for Interactive Analysis")