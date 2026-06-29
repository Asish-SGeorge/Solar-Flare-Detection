import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import numpy as np
import os
import json
import zipfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, roc_auc_score,
)

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Solar Flare Forecasting | Aditya-L1",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# PATHS  ← single source of truth, works on Streamlit Cloud
# ==========================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
OUT_DIR    = SCRIPT_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# ==========================================================
# AUTO-EXTRACT Prediction_Test_Period.zip → .csv
# ==========================================================

_zip_path = SCRIPT_DIR / "Prediction_Test_Period.zip"
_csv_path = SCRIPT_DIR / "Prediction_Test_Period.csv"

if _zip_path.exists() and not _csv_path.exists():
    with zipfile.ZipFile(_zip_path, "r") as _zf:
        _zf.extract("Prediction_Test_Period.csv", SCRIPT_DIR)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Rajdhani:wght@300;400;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #020b18 !important;
    color: #e0f0ff !important;
    font-family: 'Rajdhani', sans-serif !important;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020b18 0%, #061a30 100%) !important;
    border-right: 1px solid #0a3a5c !important;
}
[data-testid="stHeader"] { background: transparent !important; }

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        radial-gradient(1px 1px at 10% 20%, rgba(255,255,255,0.6) 0%, transparent 100%),
        radial-gradient(1px 1px at 30% 60%, rgba(255,255,255,0.4) 0%, transparent 100%),
        radial-gradient(1px 1px at 50% 10%, rgba(255,255,255,0.5) 0%, transparent 100%),
        radial-gradient(1px 1px at 70% 80%, rgba(255,255,255,0.3) 0%, transparent 100%),
        radial-gradient(1px 1px at 90% 40%, rgba(255,255,255,0.6) 0%, transparent 100%),
        radial-gradient(1px 1px at 20% 85%, rgba(255,255,255,0.4) 0%, transparent 100%),
        radial-gradient(1px 1px at 60% 50%, rgba(255,255,255,0.3) 0%, transparent 100%),
        radial-gradient(1px 1px at 80% 15%, rgba(255,255,255,0.5) 0%, transparent 100%),
        radial-gradient(circle at 50% 22%, rgba(2,11,24,0.30) 0%, rgba(2,11,24,0.85) 55%, rgba(2,11,24,0.97) 100%),
        url('https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0171.jpg');
    background-repeat: repeat, repeat, repeat, repeat, repeat, repeat, repeat, repeat, no-repeat, no-repeat;
    background-size: auto, auto, auto, auto, auto, auto, auto, auto, cover, cover;
    background-position: 0 0, 0 0, 0 0, 0 0, 0 0, 0 0, 0 0, 0 0, center top, center top;
    background-attachment: fixed;
    pointer-events: none;
    z-index: 0;
    animation: twinkle 4s infinite alternate;
}
@keyframes twinkle { 0% { opacity: 0.75; } 100% { opacity: 1.0; } }

.hero-title {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 2.4rem; font-weight: 900; text-align: center;
    background: linear-gradient(90deg, #00c6ff, #ffd166, #ff6b35, #c77dff, #00c6ff);
    background-size: 300% auto;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: shimmer 5s linear infinite;
    letter-spacing: 2px; margin-bottom: 0.2rem;
}
@keyframes shimmer { 0% { background-position: 0% center; } 100% { background-position: 300% center; } }
.hero-sub {
    text-align: center; color: #5eafd6; font-size: 1rem;
    letter-spacing: 3px; text-transform: uppercase; margin-bottom: 1.5rem;
}
.metric-card {
    background: linear-gradient(135deg, rgba(6,26,48,0.88) 0%, rgba(10,42,72,0.88) 100%);
    border: 1px solid #0d4f7a; border-radius: 12px; padding: 1.2rem 1.5rem;
    position: relative; overflow: hidden;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.metric-card:hover { transform: translateY(-4px); box-shadow: 0 8px 32px rgba(0,198,255,0.25); }
.metric-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #00c6ff, #0080ff);
    animation: scanline 2s ease-in-out infinite;
}
@keyframes scanline { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
.metric-label { font-size: 0.75rem; color: #5eafd6; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0.4rem; }
.metric-value { font-family: 'Space Grotesk', sans-serif; font-size: 2rem; font-weight: 700; color: #00e5ff; }
.metric-sub { font-size: 0.75rem; color: #3a7a9c; margin-top: 0.2rem; }
.metric-card.c-orange { border-color: #7a4a1d; }
.metric-card.c-orange::before { background: linear-gradient(90deg, #ff6b35, #ffb37a); }
.metric-card.c-orange .metric-value { color: #ff6b35; }
.metric-card.c-orange .metric-label { color: #d99a6c; }
.metric-card.c-orange:hover { box-shadow: 0 8px 32px rgba(255,107,53,0.30); }
.metric-card.c-green { border-color: #1d7a52; }
.metric-card.c-green::before { background: linear-gradient(90deg, #00e676, #5dffb0); }
.metric-card.c-green .metric-value { color: #00e676; }
.metric-card.c-green .metric-label { color: #6fc99a; }
.metric-card.c-green:hover { box-shadow: 0 8px 32px rgba(0,230,118,0.30); }
.metric-card.c-purple { border-color: #5a2d7a; }
.metric-card.c-purple::before { background: linear-gradient(90deg, #c77dff, #e0c3ff); }
.metric-card.c-purple .metric-value { color: #c77dff; }
.metric-card.c-purple .metric-label { color: #b08fd1; }
.metric-card.c-purple:hover { box-shadow: 0 8px 32px rgba(199,125,255,0.30); }
.metric-card.c-gold { border-color: #8a6a1d; }
.metric-card.c-gold::before { background: linear-gradient(90deg, #ffd166, #ffe8a8); }
.metric-card.c-gold .metric-value { color: #ffd166; }
.metric-card.c-gold .metric-label { color: #d1b46c; }
.metric-card.c-gold:hover { box-shadow: 0 8px 32px rgba(255,209,102,0.30); }
.alert-high {
    background: linear-gradient(135deg, rgba(255,60,0,0.20), rgba(255,100,0,0.08));
    border: 1px solid #ff3c00; border-radius: 10px; padding: 0.8rem 1.2rem;
    color: #ff6b35; font-family: 'Space Grotesk', sans-serif; font-size: 0.85rem;
    letter-spacing: 2px; animation: pulse-red 1.5s infinite;
}
.alert-low {
    background: linear-gradient(135deg, rgba(0,200,100,0.16), rgba(0,150,80,0.08));
    border: 1px solid #00c864; border-radius: 10px; padding: 0.8rem 1.2rem;
    color: #00e676; font-family: 'Space Grotesk', sans-serif; font-size: 0.85rem; letter-spacing: 2px;
}
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 8px rgba(255,60,0,0.4); }
    50%       { box-shadow: 0 0 24px rgba(255,60,0,0.8); }
}
.section-header {
    font-family: 'Space Grotesk', sans-serif; font-size: 0.9rem; color: #00c6ff;
    text-transform: uppercase; letter-spacing: 3px; border-left: 3px solid #00c6ff;
    padding-left: 12px; margin: 1.5rem 0 1rem 0;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSlider p { color: #7fb8d8 !important; font-family: 'Rajdhani', sans-serif !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-family: 'Space Grotesk', sans-serif !important; color: #00c6ff !important; font-size: 0.85rem !important;
}
.sidebar-badge {
    background: linear-gradient(135deg, #0a3a5c, #061a30);
    border: 1px solid #0d4f7a; border-radius: 8px; padding: 1rem; margin: 0.5rem 0;
}
.sidebar-badge p { margin: 0.2rem 0; color: #7fb8d8; font-size: 0.85rem; }
.sidebar-badge span { color: #00e5ff; font-weight: 600; }
.sidebar-badge.c-orange { background: linear-gradient(135deg, #3a1f0a, #1f0f05); border-color: #7a4a1d; }
.sidebar-badge.c-orange span { color: #ffb37a; }
.sidebar-badge.c-green { background: linear-gradient(135deg, #0a3a28, #051f16); border-color: #1d7a52; }
.sidebar-badge.c-green span { color: #5dffb0; }
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #061a30 !important; border-radius: 10px !important; padding: 4px !important; gap: 4px !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important; color: #5eafd6 !important;
    font-family: 'Rajdhani', sans-serif !important; font-size: 0.9rem !important;
    letter-spacing: 1px !important; border-radius: 8px !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: linear-gradient(135deg, #0d4f7a, #4a1f6e) !important; color: #00e5ff !important;
}
[data-testid="stDataFrame"] { border: 1px solid #0d4f7a !important; border-radius: 10px !important; }
.stDownloadButton > button {
    background: linear-gradient(135deg, #0a3a5c, #061a30) !important;
    border: 1px solid #0d4f7a !important; color: #00e5ff !important;
    font-family: 'Rajdhani', sans-serif !important; letter-spacing: 1px !important;
    border-radius: 8px !important; transition: all 0.3s ease !important; width: 100% !important;
}
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #0d4f7a, #4a1f6e) !important;
    box-shadow: 0 4px 20px rgba(0,198,255,0.3) !important; transform: translateY(-2px) !important;
}
[data-testid="stPlotlyChart"] {
    border: 1px solid #0a3a5c; border-radius: 12px; overflow: hidden; background: rgba(3,15,30,0.88);
}
hr { border-color: #0a3a5c !important; }
[data-testid="stExpander"] {
    background: rgba(6,26,48,0.88) !important; border: 1px solid #0a3a5c !important; border-radius: 10px !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', sans-serif !important; color: #00e5ff !important; font-size: 1.8rem !important;
}
[data-testid="stMetricLabel"] {
    color: #5eafd6 !important; font-size: 0.75rem !important;
    text-transform: uppercase !important; letter-spacing: 1px !important;
}
[data-testid="stAlert"] { background: rgba(6,26,48,0.9) !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# CONSTANTS
# ==========================================================

THRESHOLD        = 0.60
PERSISTENCE      = 60
COOLDOWN         = 30
WARNING_WINDOW   = 30
FLARE_WINDOW_MIN = 15

# ==========================================================
# MODEL EVALUATION ENGINE
# ==========================================================

def run_model_evaluation():
    pred_path  = SCRIPT_DIR / "Prediction_Test_Period.csv"
    flare_path = SCRIPT_DIR / "Unified_Flare_Catalog.csv"
    if not pred_path.exists() or not flare_path.exists():
        return False

    pred  = pd.read_csv(pred_path)
    flare = pd.read_csv(flare_path)
    pred["Time"]         = pd.to_datetime(pred["Time"])
    flare["SoLEXS_Time"] = pd.to_datetime(flare["SoLEXS_Time"])
    pred = pred.sort_values("Time").reset_index(drop=True)

    start, end = pred["Time"].min(), pred["Time"].max()
    flare = flare[(flare["SoLEXS_Time"] >= start) & (flare["SoLEXS_Time"] <= end)]

    if "Label" in pred.columns:
        y_true = pred["Label"].astype(int).values
    else:
        pred["y_true"] = 0
        window = pd.Timedelta(minutes=FLARE_WINDOW_MIN)
        for t in flare["SoLEXS_Time"]:
            mask = (pred["Time"] >= t - window) & (pred["Time"] <= t + window)
            pred.loc[mask, "y_true"] = 1
        y_true = pred["y_true"].values

    y_prob = pred["Probability"].values
    y_pred = (y_prob >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    accuracy_val  = accuracy_score(y_true, y_pred)
    precision_val = precision_score(y_true, y_pred, zero_division=0)
    recall_val    = recall_score(y_true, y_pred, zero_division=0)
    f1_val        = f1_score(y_true, y_pred, zero_division=0)
    fpr, tpr, _   = roc_curve(y_true, y_prob)
    roc_auc       = roc_auc_score(y_true, y_prob)

    results = {
        "threshold": THRESHOLD,
        "flare_window_minutes": FLARE_WINDOW_MIN,
        "rows_evaluated": int(len(pred)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "accuracy":  float(accuracy_val),
        "precision": float(precision_val),
        "recall":    float(recall_val),
        "f1":        float(f1_val),
        "roc_auc":   float(roc_auc),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
    }
    with open(SCRIPT_DIR / "Model_Eval_Results.json", "w") as f:
        json.dump(results, f, indent=2)

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.grid": True, "grid.alpha": 0.3, "font.size": 12,
        "axes.titleweight": "bold", "axes.titlesize": 18,
    })

    def savefig(fig, name):
        fig.savefig(str(OUT_DIR / name), dpi=150, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(pred["Time"], pred["Probability"], color="#FFC107", lw=1, label="Predicted Probability")
    ax.axhline(THRESHOLD, color="red", ls="--", lw=2, label=f"Threshold = {THRESHOLD:.2f}")
    ax.set_title("Forecast Probability Over Time")
    ax.set_xlabel("Time (UTC)"); ax.set_ylabel("Probability"); ax.set_ylim(0, 1.02); ax.legend()
    savefig(fig, "01_Forecast_Probability.png")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(pred["Time"], pred["SoLEXS_Count"], color="#3F6FE0", lw=0.8)
    ax.set_title("SoLEXS Soft X-ray Light Curve"); ax.set_xlabel("Time (UTC)"); ax.set_ylabel("Counts")
    savefig(fig, "02_SoLEXS_LightCurve.png")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.fill_between(pred["Time"], pred["HEL1OS_Count"], color="#0F8C0F")
    ax.set_title("HEL1OS Hard X-ray Light Curve"); ax.set_xlabel("Time (UTC)"); ax.set_ylabel("Counts")
    savefig(fig, "03_HEL1OS_LightCurve.png")

    fig, ax = plt.subplots(figsize=(10, 8))
    names  = ["Accuracy", "Precision", "Recall", "F1 Score"]
    vals   = [accuracy_val, precision_val, recall_val, f1_val]
    colors = ["#4C7CF0", "#FFA500", "#0A7D1E", "#8E24AA"]
    bars = ax.bar(names, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v+0.02, f"{v:.3f}", ha="center", fontweight="bold", fontsize=13)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Score"); ax.set_title("Model Performance Metrics")
    savefig(fig, "04_Model_Performance.png")

    labels   = ["Precision", "Accuracy", "Recall", "F1 Score"]
    vals_r   = [precision_val, accuracy_val, recall_val, f1_val]
    angles   = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    vals_c   = vals_r + vals_r[:1]
    angles_c = angles + angles[:1]
    fig = plt.figure(figsize=(8, 8))
    ax  = fig.add_subplot(111, polar=True)
    ax.plot(angles_c, vals_c, color="#8E24AA", lw=2.5)
    ax.fill(angles_c, vals_c, color="#8E24AA", alpha=0.25)
    ax.set_xticks(angles); ax.set_xticklabels(labels); ax.set_ylim(0, 1)
    ax.set_title("Performance Radar Chart", pad=30)
    savefig(fig, "05_Performance_Radar.png")

    cm_arr = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cm_arr, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_xticklabels(["No Flare","Flare"])
    ax.set_yticks([0,1]); ax.set_yticklabels(["No Flare","Flare"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
    thresh = cm_arr.max()/2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm_arr[i,j]}", ha="center", va="center", fontsize=14,
                    color="white" if cm_arr[i,j] > thresh else "black")
    savefig(fig, "06_Confusion_Matrix.png")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.hist(pred["Probability"], bins=50, color="#FFC107", edgecolor="black")
    ax.axvline(THRESHOLD, color="red", ls="--", lw=2, label="Threshold")
    ax.set_title("Distribution of Forecast Probabilities")
    ax.set_xlabel("Forecast Probability"); ax.set_ylabel("Frequency"); ax.legend()
    savefig(fig, "07_Probability_Distribution.png")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, color="#FF8C00", lw=3, label=f"ROC AUC = {roc_auc:.3f}")
    ax.plot([0,1],[0,1], color="black", ls="--", lw=1.5)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve"); ax.legend(loc="upper left")
    savefig(fig, "08_ROC_Curve.png")

    return True


@st.cache_resource
def ensure_model_eval():
    json_path = SCRIPT_DIR / "Model_Eval_Results.json"
    pred_path = SCRIPT_DIR / "Prediction_Test_Period.csv"
    needs_run = not json_path.exists()
    if not needs_run and pred_path.exists():
        needs_run = pred_path.stat().st_mtime > json_path.stat().st_mtime
    if needs_run:
        return run_model_evaluation()
    return True

ensure_model_eval()

# ==========================================================
# HERO HEADER
# ==========================================================

st.markdown('<div class="hero-title">☀ SOLAR FLARE FORECASTING</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Aditya-L1 · SoLEXS · HEL1OS · ISRO Mission</div>', unsafe_allow_html=True)
now_str = datetime.utcnow().strftime("🕐  %Y-%m-%d  %H:%M:%S  UTC")
st.markdown(f"<p style='text-align:center;color:#3a7a9c;font-size:0.8rem;letter-spacing:2px;'>{now_str}</p>", unsafe_allow_html=True)
st.divider()

# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():
    pred  = pd.read_csv(SCRIPT_DIR / "Prediction_Test_Period.csv")
    warn  = pd.read_csv(SCRIPT_DIR / "Warning_Events.csv")
    flare = pd.read_csv(SCRIPT_DIR / "Unified_Flare_Catalog.csv")
    pred["Time"]         = pd.to_datetime(pred["Time"])
    warn["Warning_Time"] = pd.to_datetime(warn["Warning_Time"])
    flare["SoLEXS_Time"] = pd.to_datetime(flare["SoLEXS_Time"])
    start = pred["Time"].min()
    end   = pred["Time"].max()
    flare = flare[(flare["SoLEXS_Time"] >= start) & (flare["SoLEXS_Time"] <= end)]
    return pred, warn, flare

df, warnings, flares = load_data()

# Lead time — compute from True_Positive_Warnings.csv (has Warning_Time + Flare_Time)
try:
    tp_path = SCRIPT_DIR / "True_Positive_Warnings.csv"
    if tp_path.exists():
        tp_df = pd.read_csv(tp_path)
        tp_df["Warning_Time"] = pd.to_datetime(tp_df["Warning_Time"], errors="coerce")
        if "Flare_Time" in tp_df.columns:
            tp_df["Flare_Time"] = pd.to_datetime(tp_df["Flare_Time"], errors="coerce")
            tp_df["Lead_Time"] = (tp_df["Flare_Time"] - tp_df["Warning_Time"]).dt.total_seconds() / 60
        warnings["Warning_Time"] = pd.to_datetime(warnings["Warning_Time"], errors="coerce")
        merge_cols = [c for c in ["Warning_Time", "Lead_Time", "Flare_Time"] if c in tp_df.columns]
        if len(merge_cols) > 1:
            warnings = warnings.merge(tp_df[merge_cols], on="Warning_Time", how="left")
            warnings["Lead_Time"] = pd.to_numeric(warnings["Lead_Time"], errors="coerce")
except Exception:
    pass

# ==========================================================
# MODEL-LEVEL EVALUATION
# ==========================================================

model_eval = None
eval_path  = SCRIPT_DIR / "Model_Eval_Results.json"
if eval_path.exists():
    try:
        with open(eval_path) as f:
            model_eval = json.load(f)
    except Exception:
        model_eval = None

if model_eval is not None:
    accuracy_val  = model_eval["accuracy"]
    precision_val = model_eval["precision"]
    recall_val    = model_eval["recall"]
    f1_val        = model_eval["f1"]
    roc_auc_val   = model_eval["roc_auc"]
    tn = model_eval["confusion_matrix"]["tn"]
    fp = model_eval["confusion_matrix"]["fp"]
    fn = model_eval["confusion_matrix"]["fn"]
    tp = model_eval["confusion_matrix"]["tp"]
    roc_fpr = model_eval["roc_curve"]["fpr"]
    roc_tpr = model_eval["roc_curve"]["tpr"]
else:
    accuracy_val = precision_val = recall_val = f1_val = roc_auc_val = None
    tn = fp = fn = tp = None
    roc_fpr = roc_tpr = None

param_df = pd.DataFrame()
try:
    ppath = SCRIPT_DIR / "Parameter_Search_Results.csv"
    if ppath.exists():
        param_df = pd.read_csv(ppath)
except Exception:
    pass

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.markdown("## ⚡ MISSION CONTROL")
st.sidebar.markdown(f"""
<div class="sidebar-badge">
<p>🛰️ Mission: <span>Aditya-L1</span></p>
<p>📡 Payloads: <span>SoLEXS + HEL1OS</span></p>
<p>🤖 Model: <span>XGBoost</span></p>
<p>🔢 Features: <span>63</span></p>
</div>""", unsafe_allow_html=True)

st.sidebar.markdown(f"""
<div class="sidebar-badge c-orange">
<p>🎯 Threshold: <span>{THRESHOLD:.2f}</span></p>
<p>⏱ Persistence: <span>{PERSISTENCE} sec</span></p>
<p>❄️ Cooldown: <span>{COOLDOWN} min</span></p>
<p>🪟 Warning Window: <span>{WARNING_WINDOW} min</span></p>
</div>""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("## 📅 DATE FILTER")
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(df["Time"].min().date(), df["Time"].max().date()),
    min_value=df["Time"].min().date(),
    max_value=df["Time"].max().date()
)

if len(date_range) == 2:
    s, e = date_range
    df_filtered       = df[(df["Time"].dt.date >= s) & (df["Time"].dt.date <= e)].reset_index(drop=True)
    warnings_filtered = warnings[(warnings["Warning_Time"].dt.date >= s) & (warnings["Warning_Time"].dt.date <= e)]
    flares_filtered   = flares[(flares["SoLEXS_Time"].dt.date >= s) & (flares["SoLEXS_Time"].dt.date <= e)]
else:
    df_filtered       = df.copy()
    warnings_filtered = warnings.copy()
    flares_filtered   = flares.copy()

st.sidebar.markdown("---")
st.sidebar.markdown("## 📊 MODEL PERFORMANCE")
st.sidebar.caption("Row-level classifier · full test set · not affected by date filter")
if model_eval is not None:
    st.sidebar.markdown(f"""
    <div class="sidebar-badge c-green">
    <p>🎯 Accuracy: <span>{accuracy_val:.3f}</span></p>
    <p>✅ Precision: <span>{precision_val:.3f}</span></p>
    <p>🔁 Recall: <span>{recall_val:.3f}</span></p>
    <p>⚖️ F1 Score: <span>{f1_val:.3f}</span></p>
    <p>📈 ROC AUC: <span>{roc_auc_val:.3f}</span></p>
    </div>""", unsafe_allow_html=True)
else:
    st.sidebar.warning("Model_Eval_Results.json not found — ensure Prediction_Test_Period.zip exists.")

st.sidebar.markdown("---")
st.sidebar.markdown("## 🔄 RE-RUN EVALUATION")
if st.sidebar.button("♻️ Regenerate Model_Eval_Results.json + PNGs"):
    with st.spinner("Running model evaluation…"):
        ok = run_model_evaluation()
        ensure_model_eval.clear()
    if ok:
        st.sidebar.success("Done! Reload the page to see updated metrics.")
    else:
        st.sidebar.error("Missing Prediction_Test_Period.csv or Unified_Flare_Catalog.csv.")

# ==========================================================
# STATUS BANNER
# ==========================================================

current_prob = float(df_filtered["Probability"].iloc[-1]) if len(df_filtered) > 0 else 0.0
if current_prob >= THRESHOLD:
    st.markdown(f'<div class="alert-high">🔴 &nbsp; HIGH SOLAR FLARE RISK DETECTED &nbsp;|&nbsp; PROBABILITY: {current_prob:.2f} &nbsp;|&nbsp; THRESHOLD EXCEEDED</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="alert-low">🟢 &nbsp; NOMINAL SOLAR CONDITIONS &nbsp;|&nbsp; PROBABILITY: {current_prob:.2f} &nbsp;|&nbsp; BELOW THRESHOLD</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ==========================================================
# KPI CARDS
# ==========================================================

st.markdown('<div class="section-header" style="color:#00c6ff;border-left-color:#00c6ff;">FORECAST KPIs</div>', unsafe_allow_html=True)

avg_lead = None
if "Lead_Time" in warnings.columns:
    lt = pd.to_numeric(warnings["Lead_Time"], errors="coerce").dropna()
    avg_lead = lt.mean() if len(lt) > 0 else None

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card c-orange">
        <div class="metric-label">Current Probability</div>
        <div class="metric-value">{current_prob:.2f}</div>
        <div class="metric-sub">XGBoost output</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card c-gold">
        <div class="metric-label">Warning Events</div>
        <div class="metric-value">{len(warnings_filtered)}</div>
        <div class="metric-sub">in selected range</div>
    </div>""", unsafe_allow_html=True)
with c3:
    lead_str = f"{avg_lead:.1f} min" if avg_lead is not None else "N/A"
    st.markdown(f"""<div class="metric-card c-purple">
        <div class="metric-label">Avg Lead Time</div>
        <div class="metric-value" style="font-size:1.5rem">{lead_str}</div>
        <div class="metric-sub">early warning</div>
    </div>""", unsafe_allow_html=True)
with c4:
    f1_display = f"{f1_val:.3f}" if model_eval is not None else "N/A"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Model F1 Score</div>
        <div class="metric-value">{f1_display}</div>
        <div class="metric-sub">row-level classifier</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================================
# GAUGE + RADAR
# ==========================================================

gauge_col, perf_col = st.columns([1, 1])

with gauge_col:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_prob,
        number={"valueformat": ".3f", "font": {"color": "#00e5ff", "family": "Space Grotesk", "size": 36}},
        delta={"reference": THRESHOLD, "valueformat": ".2f",
               "increasing": {"color": "#ff3c00"}, "decreasing": {"color": "#00e676"}},
        gauge={
            "axis": {"range": [0, 1], "tickcolor": "#1a4a6a", "tickfont": {"color": "#5eafd6"}},
            "bar": {"color": "#ff6b35" if current_prob >= THRESHOLD else "#00c6ff", "thickness": 0.25},
            "bgcolor": "rgba(3,15,30,0.6)", "bordercolor": "#0a3a5c",
            "steps": [
                {"range": [0.0, 0.3], "color": "rgba(0,200,100,0.16)"},
                {"range": [0.3, 0.6], "color": "rgba(255,209,102,0.14)"},
                {"range": [0.6, 0.8], "color": "rgba(255,120,0,0.16)"},
                {"range": [0.8, 1.0], "color": "rgba(255,60,0,0.20)"},
            ],
            "threshold": {"line": {"color": "#ffffff", "width": 2}, "thickness": 0.8, "value": THRESHOLD}
        },
        title={"text": "FLARE PROBABILITY GAUGE", "font": {"color": "#5eafd6", "family": "Space Grotesk", "size": 11}}
    ))
    fig_gauge.update_layout(height=320, margin=dict(t=60, b=20, l=30, r=30),
        paper_bgcolor="rgba(3,15,30,0.0)", plot_bgcolor="rgba(3,15,30,0.0)", font_color="#e0f0ff")
    st.plotly_chart(fig_gauge, use_container_width=True)

with perf_col:
    if model_eval is not None:
        radar_labels = ["Precision", "Accuracy", "Recall", "F1 Score"]
        radar_vals   = [precision_val, accuracy_val, recall_val, f1_val]
        fig_radar = go.Figure(go.Scatterpolar(
            r=radar_vals + [radar_vals[0]], theta=radar_labels + [radar_labels[0]],
            fill="toself", fillcolor="rgba(199,125,255,0.16)",
            line=dict(color="#c77dff", width=2), marker=dict(color="#e0c3ff", size=6)
        ))
        fig_radar.update_layout(
            polar=dict(bgcolor="rgba(3,15,30,0.6)",
                radialaxis=dict(visible=True, range=[0,1], color="#1a4a6a", tickfont=dict(color="#3a7a9c", size=9)),
                angularaxis=dict(color="#3a7a9c", tickfont=dict(color="#7fb8d8", size=11))),
            showlegend=False, height=320, margin=dict(t=60, b=20, l=40, r=40),
            paper_bgcolor="rgba(3,15,30,0.0)",
            title=dict(text="MODEL PERFORMANCE RADAR", font=dict(color="#5eafd6", family="Space Grotesk", size=11), x=0.5)
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info("Model evaluation JSON not found — run evaluation first.")

st.divider()

# ==========================================================
# LIGHT CURVES
# ==========================================================

CHART_LAYOUT = dict(
    height=380, hovermode="x unified",
    paper_bgcolor="rgba(3,15,30,0.0)", plot_bgcolor="rgba(3,15,30,0.0)",
    font=dict(color="#7fb8d8", family="Rajdhani"),
    xaxis=dict(title="Time (UTC)", color="#3a7a9c", gridcolor="#0a2a48", showgrid=True, zerolinecolor="#0a3a5c"),
    yaxis=dict(title="Count", color="#3a7a9c", gridcolor="#0a2a48", showgrid=True, zerolinecolor="#0a3a5c"),
    legend=dict(bgcolor="rgba(3,15,30,0.8)", bordercolor="#0a3a5c", borderwidth=1),
    margin=dict(t=50, b=40, l=60, r=40)
)

def add_markers(fig, warnings_f, flares_f):
    for t in warnings_f["Warning_Time"]:
        fig.add_vline(x=t, line_dash="dash", line_color="#ffaa00", opacity=0.5, line_width=1)
    for t in flares_f["SoLEXS_Time"]:
        fig.add_vline(x=t, line_color="#ff3c00", opacity=0.8, line_width=1.5)
    return fig

st.markdown('<div class="section-header" style="color:#ff6b35;border-left-color:#ff6b35;">REAL-TIME LIGHT CURVES</div>', unsafe_allow_html=True)

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df_filtered["Time"], y=df_filtered["SoLEXS_Count"], mode="lines", name="SoLEXS",
    line=dict(color="#00c6ff", width=1.5), fill="tozeroy", fillcolor="rgba(0,198,255,0.06)",
    hovertemplate="<b>%{x}</b><br>Count: %{y}<extra>SoLEXS</extra>"))
fig1 = add_markers(fig1, warnings_filtered, flares_filtered)
fig1.update_layout(**CHART_LAYOUT, title=dict(text="SoLEXS — Soft X-ray Light Curve",
    font=dict(color="#00c6ff", family="Space Grotesk", size=12), x=0.01))
st.plotly_chart(fig1, use_container_width=True)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df_filtered["Time"], y=df_filtered["HEL1OS_Count"], mode="lines", name="HEL1OS",
    line=dict(color="#00e676", width=1.5), fill="tozeroy", fillcolor="rgba(0,230,118,0.06)",
    hovertemplate="<b>%{x}</b><br>Count: %{y}<extra>HEL1OS</extra>"))
fig2 = add_markers(fig2, warnings_filtered, flares_filtered)
fig2.update_layout(**CHART_LAYOUT, title=dict(text="HEL1OS — Hard X-ray Light Curve",
    font=dict(color="#00e676", family="Space Grotesk", size=12), x=0.01))
st.plotly_chart(fig2, use_container_width=True)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=df_filtered["Time"], y=df_filtered["Probability"], mode="lines", name="Probability",
    line=dict(color="#ffd166", width=2), fill="tozeroy", fillcolor="rgba(255,209,102,0.10)",
    hovertemplate="<b>%{x}</b><br>Probability: %{y:.3f}<extra>Forecast</extra>"))
fig3.add_hline(y=THRESHOLD, line_dash="dash", line_color="#ffffff", line_width=1.5,
    annotation_text=f"Threshold {THRESHOLD:.2f}", annotation_font_color="#ffffff", annotation_position="right")
fig3.add_hrect(y0=THRESHOLD, y1=1, fillcolor="rgba(255,107,53,0.10)", line_width=0)
try:
    for _, row in warnings_filtered.iterrows():
        wt = row.get("Warning_Time")
        if pd.isna(wt): continue
        ft = None
        if "Flare_Time" in row.index and pd.notna(row.get("Flare_Time")):
            ft = row["Flare_Time"]
        elif "Lead_Time" in row.index and pd.notna(row.get("Lead_Time")):
            ft = wt + pd.to_timedelta(float(row["Lead_Time"]), unit="m")
        if ft is not None and ft > wt:
            fig3.add_vrect(x0=wt, x1=ft, fillcolor="rgba(0,230,118,0.07)", line_width=0)
            lead_min = (ft - wt) / pd.Timedelta(minutes=1)
            fig3.add_annotation(x=wt+(ft-wt)/2, y=min(0.95, THRESHOLD+0.1),
                text=f"{lead_min:.1f}m", showarrow=False, font=dict(color="#00e676", size=10))
except Exception:
    pass
fig3.update_layout(**{**CHART_LAYOUT, "yaxis": dict(range=[0,1], title="Probability", color="#3a7a9c", gridcolor="#0a2a48", showgrid=True)},
    title=dict(text="Forecast Probability + Dynamic Threshold", font=dict(color="#ffd166", family="Space Grotesk", size=12), x=0.01))
st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ==========================================================
# PARAMETER SEARCH
# ==========================================================

if not param_df.empty:
    st.markdown('<div class="section-header" style="color:#c77dff;border-left-color:#c77dff;">PARAMETER SEARCH RESULTS</div>', unsafe_allow_html=True)
    p1, p2 = st.columns(2)
    with p1:
        st.dataframe(param_df, use_container_width=True, height=300)
    with p2:
        numeric_cols = param_df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            x_col = numeric_cols[0]; y_col = numeric_cols[1]
            fig_p = px.scatter(param_df, x=x_col, y=y_col,
                color=numeric_cols[2] if len(numeric_cols) > 2 else None,
                color_continuous_scale=["#0a1a30","#00c6ff","#ffd166"], template="plotly_dark")
            fig_p.update_layout(paper_bgcolor="rgba(3,15,30,0.0)", plot_bgcolor="rgba(3,15,30,0.0)",
                height=300, margin=dict(t=30,b=30,l=40,r=40),
                title=dict(text="Parameter Search Space", font=dict(color="#5eafd6", family="Space Grotesk", size=11)))
            st.plotly_chart(fig_p, use_container_width=True)
    st.divider()

# ==========================================================
# DATA EXPLORER TABS
# ==========================================================

st.markdown('<div class="section-header" style="color:#00e676;border-left-color:#00e676;">DATA EXPLORER</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "⚠️  Warning Events", "🔴  Flare Catalog", "📈  Analysis",
    "🔭  Probability Distribution", "🧪  ROC Curve", "📥  Downloads"
])

with tab1:
    col_s, col_n = st.columns([3,1])
    with col_s:
        srch = st.text_input("🔍 Search warnings", placeholder="Filter by any column...", key="warn_search")
    with col_n:
        st.metric("Total Warnings", len(warnings_filtered))
    disp = warnings_filtered[warnings_filtered.astype(str).apply(lambda x: x.str.contains(srch, case=False, regex=False)).any(axis=1)] if srch else warnings_filtered
    st.dataframe(disp, use_container_width=True, height=400)

with tab2:
    col_s2, col_n2 = st.columns([3,1])
    with col_s2:
        srch2 = st.text_input("🔍 Search flares", placeholder="Filter by any column...", key="flare_search")
    with col_n2:
        st.metric("Total Flares", len(flares_filtered))
    disp2 = flares_filtered[flares_filtered.astype(str).apply(lambda x: x.str.contains(srch2, case=False, regex=False)).any(axis=1)] if srch2 else flares_filtered
    st.dataframe(disp2, use_container_width=True, height=400)
    if len(disp2) > 0:
        fc1, fc2, fc3 = st.columns(3)
        if "Class" in disp2.columns:
            top_class = disp2["Class"].value_counts().index[0]
            fc1.metric("Most Common Class", top_class)
            class_counts = disp2["Class"].value_counts().reset_index()
            class_counts.columns = ["Class","Count"]
            fig_cls = px.bar(class_counts, x="Class", y="Count",
                color="Count", color_continuous_scale=["#0a1a30","#00c6ff","#ffd166"], template="plotly_dark")
            fig_cls.update_layout(paper_bgcolor="rgba(3,15,30,0.0)", plot_bgcolor="rgba(3,15,30,0.0)",
                height=250, showlegend=False, margin=dict(t=20,b=20,l=40,r=20))
            fc2.plotly_chart(fig_cls, use_container_width=True)
        fc3.metric("Flares Shown", len(disp2))

with tab3:
    a1, a2 = st.columns(2)
    with a1:
        st.markdown('<div class="section-header" style="font-size:0.75rem;color:#ffd166;border-left-color:#ffd166;">LEAD TIME ANALYSIS</div>', unsafe_allow_html=True)
        if "Lead_Time" in warnings_filtered.columns:
            lt_vals = pd.to_numeric(warnings_filtered["Lead_Time"], errors="coerce").dropna()
            if len(lt_vals) > 0:
                st.markdown(f"""
<div class="metric-card c-gold">
<div class="metric-label">Lead Time Summary</div>
<p style="color:#7fb8d8;margin:0.3rem 0">Mean: <span style="color:#ffd166">{lt_vals.mean():.1f} min</span></p>
<p style="color:#7fb8d8;margin:0.3rem 0">Min: <span style="color:#ffd166">{lt_vals.min():.1f} min</span></p>
<p style="color:#7fb8d8;margin:0.3rem 0">Max: <span style="color:#ffd166">{lt_vals.max():.1f} min</span></p>
</div>""", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                fig_lt = go.Figure(go.Box(y=lt_vals, name="Lead Time",
                    marker_color="#c77dff", line_color="#a85ce0", fillcolor="rgba(199,125,255,0.12)"))
                fig_lt.update_layout(
                    title=dict(text="Lead Time Distribution", font=dict(color="#5eafd6", family="Space Grotesk", size=11)),
                    height=280, paper_bgcolor="rgba(3,15,30,0.0)", plot_bgcolor="rgba(3,15,30,0.0)",
                    yaxis=dict(title="Minutes", color="#3a7a9c", gridcolor="#0a2a48"),
                    margin=dict(t=40,b=20,l=40,r=20))
                st.plotly_chart(fig_lt, use_container_width=True)
            else:
                st.info("No lead time data available in selected range.")
        else:
            st.info("Lead_Time_Analysis.csv not found or no lead time column detected.")
    with a2:
        st.markdown('<div class="section-header" style="font-size:0.75rem;color:#00c6ff;border-left-color:#00c6ff;">MODEL CLASSIFICATION METRICS (ROW-LEVEL)</div>', unsafe_allow_html=True)
        if model_eval is not None:
            fig_bar = go.Figure(go.Bar(
                x=["Accuracy","Precision","Recall","F1 Score"],
                y=[accuracy_val, precision_val, recall_val, f1_val],
                marker=dict(color=["#4c7cf0","#00c6ff","#ffd166","#c77dff"], line=dict(color="#0a3a5c", width=1)),
                text=[f"{v:.3f}" for v in [accuracy_val, precision_val, recall_val, f1_val]],
                textposition="outside", textfont=dict(color="#00e5ff", family="Space Grotesk")
            ))
            fig_bar.update_layout(height=280, paper_bgcolor="rgba(3,15,30,0.0)", plot_bgcolor="rgba(3,15,30,0.0)",
                yaxis=dict(range=[0,1.1], gridcolor="#0a2a48", color="#3a7a9c"),
                xaxis=dict(color="#3a7a9c"), margin=dict(t=20,b=20,l=40,r=20), font=dict(color="#7fb8d8"))
            st.plotly_chart(fig_bar, use_container_width=True)

            fig_cm = go.Figure(go.Heatmap(
                z=[[tn, fp],[fn, tp]],
                x=["Pred: No Flare","Pred: Flare"], y=["Actual: No Flare","Actual: Flare"],
                colorscale=[[0,"#0a1a30"],[1,"#00c6ff"]],
                text=[[str(tn),str(fp)],[str(fn),str(tp)]],
                texttemplate="%{text}", textfont=dict(color="white", family="Space Grotesk", size=16),
                showscale=False
            ))
            fig_cm.update_layout(
                title=dict(text="Confusion Matrix (full test set)", font=dict(color="#5eafd6", family="Space Grotesk", size=11)),
                height=280, paper_bgcolor="rgba(3,15,30,0.0)", plot_bgcolor="rgba(3,15,30,0.0)",
                xaxis=dict(color="#3a7a9c"), yaxis=dict(color="#3a7a9c"), margin=dict(t=40,b=20,l=80,r=20))
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.warning("Model evaluation data unavailable. Ensure Prediction_Test_Period.zip exists and reload.")

with tab4:
    d1, d2 = st.columns(2)
    with d1:
        st.metric("Max Probability", f"{df_filtered['Probability'].max():.3f}")
        st.metric("Mean Probability", f"{df_filtered['Probability'].mean():.3f}")
    with d2:
        above = (df_filtered["Probability"] >= THRESHOLD).sum()
        pct = above / len(df_filtered) * 100 if len(df_filtered) > 0 else 0
        st.metric("Time Above Threshold", f"{above} rows")
        st.metric("% Time at Risk", f"{pct:.1f}%")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=df_filtered["Probability"], nbinsx=50,
        marker=dict(color="#00c6ff", opacity=0.7, line=dict(color="#0a3a5c", width=0.5)), name="Probability"))
    fig_hist.add_vline(x=THRESHOLD, line_dash="dash", line_color="#ff6b35", line_width=2,
        annotation_text=f"Threshold {THRESHOLD}", annotation_font_color="#ff6b35")
    fig_hist.update_layout(
        title=dict(text="Distribution of Forecast Probabilities", font=dict(color="#5eafd6", family="Space Grotesk", size=11)),
        height=350, paper_bgcolor="rgba(3,15,30,0.0)", plot_bgcolor="rgba(3,15,30,0.0)",
        xaxis=dict(title="Probability", color="#3a7a9c", gridcolor="#0a2a48"),
        yaxis=dict(title="Frequency", color="#3a7a9c", gridcolor="#0a2a48"),
        margin=dict(t=50,b=40,l=60,r=40), font=dict(color="#7fb8d8"))
    st.plotly_chart(fig_hist, use_container_width=True)

with tab5:
    if model_eval is not None and roc_fpr is not None:
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=roc_fpr, y=roc_tpr, mode="lines",
            name=f"ROC AUC = {roc_auc_val:.3f}", line=dict(color="#ff8c00", width=3)))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random classifier",
            line=dict(color="#7fb8d8", width=1.5, dash="dash")))
        fig_roc.update_layout(
            title=dict(text="ROC Curve — Row-Level Classifier", font=dict(color="#5eafd6", family="Space Grotesk", size=12)),
            height=450, paper_bgcolor="rgba(3,15,30,0.0)", plot_bgcolor="rgba(3,15,30,0.0)",
            xaxis=dict(title="False Positive Rate", color="#3a7a9c", gridcolor="#0a2a48", range=[0,1]),
            yaxis=dict(title="True Positive Rate", color="#3a7a9c", gridcolor="#0a2a48", range=[0,1]),
            legend=dict(bgcolor="rgba(3,15,30,0.8)", bordercolor="#0a3a5c", borderwidth=1),
            font=dict(color="#7fb8d8"))
        st.plotly_chart(fig_roc, use_container_width=True)
        st.caption("Computed once on the full row-level test set — not affected by the sidebar date filter.")
    else:
        st.warning("ROC data unavailable. Ensure Prediction_Test_Period.zip exists and reload.")

with tab6:
    dc1, dc2, dc3 = st.columns(3)
    date_label = f"{date_range[0]}_to_{date_range[1]}" if len(date_range)==2 else "all"
    with dc1:
        st.download_button("📥 Filtered Predictions",
            df_filtered.to_csv(index=False), f"Predictions_{date_label}.csv", "text/csv", key="dl_pred")
    with dc2:
        st.download_button("📥 Filtered Warnings",
            warnings_filtered.to_csv(index=False), f"Warnings_{date_label}.csv", "text/csv", key="dl_warn")
    with dc3:
        st.download_button("📥 Filtered Flares",
            flares_filtered.to_csv(index=False), f"Flares_{date_label}.csv", "text/csv", key="dl_flare")
    st.markdown("---")
    if st.button("📄 Generate Summary Report ↗"):
        avg_lead_g = None
        if "Lead_Time" in warnings.columns:
            lt_g = pd.to_numeric(warnings["Lead_Time"], errors="coerce").dropna()
            avg_lead_g = lt_g.mean() if len(lt_g) > 0 else None
        model_block = (
            f"- Accuracy:  {accuracy_val:.3f}\n"
            f"- Precision: {precision_val:.3f}\n"
            f"- Recall:    {recall_val:.3f}\n"
            f"- F1 Score:  {f1_val:.3f}\n"
            f"- ROC AUC:   {roc_auc_val:.3f}\n"
            f"- Confusion Matrix: TN={tn}  FP={fp}  FN={fn}  TP={tp}"
            if model_eval is not None else
            "- Not available. Ensure Prediction_Test_Period.zip exists and reload."
        )
        report = f"""# Solar Flare Forecasting Report
Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}

## Filter Settings
- Date Range: {date_label}
- Probability Threshold: {THRESHOLD:.3f}
- Persistence: {PERSISTENCE} sec | Cooldown: {COOLDOWN} min | Warning Window: {WARNING_WINDOW} min

## Model
- Algorithm: XGBoost
- Features: 63
- Payloads: SoLEXS (Soft X-ray) + HEL1OS (Hard X-ray)
- Mission: Aditya-L1, ISRO

## Model Classification Performance (row-level, full test set)
{model_block}

## Data Summary
- Total Predictions: {len(df_filtered)}
- Warnings Generated: {len(warnings_filtered)}
- Observed Flares: {len(flares_filtered)}
{f'- Average Lead Time: {avg_lead_g:.2f} min' if avg_lead_g else ''}

---
*Solar Flare Forecasting System — BAH 2026 | PS15*
"""
        st.download_button("📄 Download Report", report, f"Report_{date_label}.md", "text/markdown", key="dl_report")

# ==========================================================
# FOOTER
# ==========================================================

st.divider()
st.markdown("""
<div style="text-align:center;padding:1rem 0;">
    <p style="font-family:'Space Grotesk',sans-serif;font-size:0.75rem;color:#1a4a6a;letter-spacing:3px;">
        ADITYA-L1 · SOLAR FLARE FORECASTING SYSTEM · BAH 2026 · PS15 · ISRO
    </p>
    <p style="font-size:0.75rem;color:#0d3a5c;">
        SoLEXS (Soft X-ray Spectrometer) &nbsp;·&nbsp; HEL1OS (High Energy L1 Orbiting X-ray Spectrometer) &nbsp;·&nbsp; XGBoost · 63 Features
    </p>
</div>
""", unsafe_allow_html=True)
