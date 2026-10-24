"""Wi-Fi Radar - real-time dashboard.   Run:  streamlit run dashboard.py"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import CLASSES, LABELS
from engine import Engine
from model import load_or_train
from simulator import SimSource
from sources import SerialSource

st.set_page_config(page_title="Wi-Fi Radar", page_icon="📡", layout="wide")
COLORS = {"empty": "#2e7d32", "sitting": "#1565c0", "walking": "#ef6c00", "fall": "#c62828"}
ICONS = {"empty": "🟢", "sitting": "🪑", "walking": "🚶", "fall": "🚨"}


@st.cache_resource
def get_engine():
    model, meta = load_or_train()
    eng = Engine(model, meta)
    eng.start(SimSource())
    return eng


eng = get_engine()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("📡 Wi-Fi Radar")
    st.caption("Camera-free presence & activity sensing from Wi-Fi CSI")
    kind = st.radio("CSI source", ["Simulator", "ESP32 (serial)"],
                    index=0 if getattr(eng.source, "kind", "sim") == "sim" else 1)
    if kind.startswith("ESP32"):
        port = st.text_input("Serial port", "COM5", help="e.g. COM5, /dev/ttyUSB0, /dev/cu.usbserial-0001")
        baud = st.number_input("Baud rate", value=921600, step=1)
    if st.button("Apply source", width="stretch"):
        eng.start(SimSource() if kind == "Simulator" else SerialSource(port, int(baud)))
        st.rerun()

    if getattr(eng.source, "kind", "") == "sim":
        st.subheader("Simulator controls")
        eng.source.set_mode(st.selectbox("Scenario", ["auto", "empty", "sitting", "walking"],
                                         help="auto cycles empty > sitting > walking > fall"))
        if st.button("Trigger simulated fall", width="stretch"):
            eng.source.trigger_fall()

    st.subheader("Alerting")
    eng.fall_thr = st.slider("Fall probability threshold", 0.3, 0.95, eng.fall_thr, 0.05)
    st.caption("Optional: set env var WEBHOOK_URL to push alerts to Slack/Discord/ntfy.")
    st.subheader("Model")
    st.caption(f"Trained on: {eng.meta.get('trained_on', 'n/a')}  \nHold-out accuracy: "
               f"{eng.meta.get('accuracy', 0):.1%}")

st.title("Wi-Fi Radar")
if getattr(eng.source, "kind", "") == "sim":
    st.warning("**Simulator mode** - all signals are synthetic CSI-like data, not real people. "
               "Connect an ESP32 to sense a real room.")


@st.fragment(run_every=0.5)
def live():
    s = eng.snapshot()
    if s["error"]:
        st.error(f"Source error: {s['error']}")
        return
    if s["label"] is None:
        st.info(f"Collecting signal... (needs ~4 s of data, packet rate {s['rate']:.0f}/s)")
        return

    # new-alert toast
    if s["n_alerts"] > st.session_state.get("seen_alerts", 0):
        st.toast("🚨 Possible fall detected!", icon="🚨")
    st.session_state["seen_alerts"] = s["n_alerts"]

    lab = s["label"]
    conf = float(s["probs"][CLASSES.index(lab)])
    text = "POSSIBLE FALL DETECTED" if lab == "fall" else LABELS[lab]
    st.markdown(
        f"<div style='padding:18px 24px;border-radius:14px;background:{COLORS[lab]};color:white;"
        f"font-size:30px;font-weight:700'>{ICONS[lab]} {text}"
        f"<span style='font-size:16px;font-weight:400;margin-left:14px'>"
        f"{'' if lab == 'fall' else f'confidence {conf:.0%}'}</span></div>", unsafe_allow_html=True)
    st.write("")

    c1, c2, c3 = st.columns(3)
    c1.metric("Packet rate", f"{s['rate']:.0f} /s")
    c2.metric("Fall alerts (session)", s["n_alerts"])
    c3.metric("Fall probability", f"{s['probs'][CLASSES.index('fall')]:.0%}")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Signal activity")
        if s["heat"] is not None:
            x, z = s["heat"]
            fig = go.Figure(go.Heatmap(z=z.T, x=x, colorscale="Viridis", zmin=-0.12, zmax=0.12,
                                       colorbar=dict(title="Δ amp")))
            fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0),
                              xaxis_title="seconds ago", yaxis_title="subcarrier")
            st.plotly_chart(fig, width="stretch", key="heat")
        if s["energy"]:
            t = np.array([e[0] for e in s["energy"]])
            fig = go.Figure(go.Scatter(x=t - t[-1], y=[e[1] for e in s["energy"]], fill="tozeroy",
                                       line=dict(color="#ef6c00")))
            fig.update_layout(height=200, margin=dict(l=0, r=0, t=10, b=0),
                              xaxis_title="seconds ago", yaxis_title="motion energy")
            st.plotly_chart(fig, width="stretch", key="energy")
    with right:
        st.subheader("Activity probabilities")
        fig = go.Figure(go.Bar(x=s["probs"], y=[LABELS[c] for c in CLASSES], orientation="h",
                               marker_color=[COLORS[c] for c in CLASSES]))
        fig.update_layout(height=230, margin=dict(l=0, r=0, t=10, b=0), xaxis=dict(range=[0, 1]),
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch", key="probs")
        st.subheader("Alert log")
        if s["alerts"]:
            df = pd.DataFrame(s["alerts"])
            df["prob"] = (df["prob"] * 100).round(0).astype(int).astype(str) + "%"
            st.dataframe(df.rename(columns={"time": "Time", "prob": "Confidence", "type": "Event"}),
                         hide_index=True, width="stretch")
        else:
            st.caption("No alerts yet.")


live()

with st.expander("How it works"):
    st.markdown(
        "1. **ESP32** pings the Wi-Fi router ~50x/s and prints the Channel State Information (CSI) of each reply.\n"
        "2. **Python** removes gain jitter and static multipath, resamples to 50 Hz and cuts 4 s windows.\n"
        "3. **Features**: energy, PCA coherence, breathing/gait band power, spectral entropy, and a 0.5 s energy "
        "envelope (fall = big burst then quiet; walking = sustained energy).\n"
        "4. **Random Forest** predicts empty / sitting / walking / fall-like; a fall alert needs 2 of the last 3 "
        "windows above the threshold, with a 20 s cooldown.\n\n"
        "No camera, no audio, no identifying data - only radio-signal statistics.")
with st.expander("Model evaluation"):
    cm = eng.meta.get("confusion")
    if cm:
        st.dataframe(pd.DataFrame(cm, index=[f"true: {c}" for c in CLASSES],
                                  columns=[f"pred: {c}" for c in CLASSES]))
    imp = eng.meta.get("importances")
    if imp:
        st.bar_chart(pd.Series(imp).sort_values())
