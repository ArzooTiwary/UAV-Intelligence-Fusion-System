import streamlit as st
import plotly.graph_objects as go
import numpy as np
import sys, os
from datetime import datetime
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code'))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from module4_fusion import fuse
from module5_report import generate_report
from module2_ekf import UAVKalmanFilter

st.set_page_config(
    page_title="UAV-IFS | Command Interface",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Inter:wght@300;400;500;600;700&display=swap');

/* ── KILL ALL STREAMLIT CHROME ── */
#MainMenu, footer, header,
[data-testid="stDecoration"],
[data-testid="stToolbar"],
[data-testid="stStatusWidget"],
div[data-baseweb="notification"] { display:none !important; }

/* ── BASE ── */
html, body { margin:0; padding:0; }
[data-testid="stAppViewContainer"] {
    background:#07090f !important;
    font-family:'Inter',sans-serif;
}
[data-testid="stAppViewBlockContainer"] {
    padding-top: 16px !important;
    max-width: 100% !important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background:#0b0e16 !important;
    border-right:1px solid #1a1f2e;
}
[data-testid="stSidebar"] section { padding-top:24px; }
[data-testid="stSidebar"] * { color:#6e7a91 !important; }
[data-testid="stSidebar"] h4 { color:#3d85c8 !important; font-size:11px !important; letter-spacing:3px; }
[data-testid="stSidebar"] label { color:#8892a4 !important; font-size:11px !important; }
[data-testid="stSidebar"] input { background:#0f1320 !important; border:1px solid #1a1f2e !important; color:#c9d1d9 !important; }
[data-testid="stSidebar"] .stSlider > div > div > div { background:#1f6feb !important; }

/* ── HEADER ── */
.top-header {
    background: linear-gradient(180deg,#0d1321 0%,#090c16 100%);
    border:1px solid #1a1f2e;
    border-top:2px solid #1f6feb;
    border-radius:8px;
    padding:16px 28px;
    margin-bottom:16px;
    display:flex;
    justify-content:space-between;
    align-items:center;
}
.header-left {}
.header-title {
    font-family:'Share Tech Mono',monospace;
    font-size:20px;
    color:#58a6ff;
    letter-spacing:4px;
    line-height:1;
}
.header-sub {
    font-family:'Share Tech Mono',monospace;
    font-size:9px;
    color:#3fb950;
    letter-spacing:3px;
    margin-top:6px;
}
.header-right {
    font-family:'Share Tech Mono',monospace;
    font-size:10px;
    color:#3d4451;
    text-align:right;
    line-height:2;
}
.header-right b { color:#6e7a91; }

/* ── MODULE STATUS BAR ── */
.mod-bar { display:flex; gap:8px; margin-bottom:16px; }
.mod-pill {
    flex:1; text-align:center;
    font-family:'Share Tech Mono',monospace;
    font-size:9px; letter-spacing:2px;
    padding:9px 8px;
    border-radius:4px;
    border:1px solid #1a1f2e;
    background:#0b0e16;
    color:#2d333b;
}
.mod-ok   { color:#3fb950; border-color:#1a4731; background:#0a1410; }
.mod-warn { color:#e3b341; border-color:#4d3b10; background:#120e01; }
.mod-crit { color:#f85149; border-color:#4d1515; background:#130707; }

/* ── THREAT BANNER ── */
.t-high {
    font-family:'Share Tech Mono',monospace;
    font-size:32px; letter-spacing:6px;
    color:#f85149; text-align:center;
    background:linear-gradient(180deg,#160404,#0d0202);
    border:1px solid #3d0c0c; border-left:3px solid #f85149;
    border-radius:6px; padding:20px 16px 16px;
    line-height:1;
}
.t-med {
    font-family:'Share Tech Mono',monospace;
    font-size:32px; letter-spacing:6px;
    color:#e3b341; text-align:center;
    background:linear-gradient(180deg,#141002,#0d0c01);
    border:1px solid #3d3210; border-left:3px solid #e3b341;
    border-radius:6px; padding:20px 16px 16px;
    line-height:1;
}
.t-low {
    font-family:'Share Tech Mono',monospace;
    font-size:32px; letter-spacing:6px;
    color:#3fb950; text-align:center;
    background:linear-gradient(180deg,#021208,#010d05);
    border:1px solid #0e3d1c; border-left:3px solid #3fb950;
    border-radius:6px; padding:20px 16px 16px;
    line-height:1;
}
.t-sub {
    font-size:10px; letter-spacing:3px;
    opacity:0.5; margin-top:8px;
    font-family:'Share Tech Mono',monospace;
}

/* ── CARDS ── */
.card {
    background:#0b0e16;
    border:1px solid #1a1f2e;
    border-radius:6px;
    padding:16px 18px;
    margin-bottom:14px;
}
.card-title {
    font-family:'Share Tech Mono',monospace;
    font-size:9px; color:#2d4a6e;
    letter-spacing:3px; text-transform:uppercase;
    border-bottom:1px solid #1a1f2e;
    padding-bottom:8px; margin-bottom:14px;
}

/* ── SCORE ── */
.score-big {
    font-family:'Share Tech Mono',monospace;
    font-size:52px; font-weight:700;
    line-height:1; text-align:center;
}
.score-range {
    font-family:'Share Tech Mono',monospace;
    font-size:10px; color:#3d4451;
    text-align:center; margin-top:6px; letter-spacing:2px;
}

/* ── FACTOR PILLS ── */
.fp-h { display:inline-block; background:#160404; border:1px solid #f85149;
        border-radius:3px; padding:4px 12px; font-size:10px; color:#f85149;
        margin:3px; font-family:'Share Tech Mono',monospace; letter-spacing:2px; }
.fp-m { display:inline-block; background:#141002; border:1px solid #e3b341;
        border-radius:3px; padding:4px 12px; font-size:10px; color:#e3b341;
        margin:3px; font-family:'Share Tech Mono',monospace; letter-spacing:2px; }
.fp-l { display:inline-block; background:#021208; border:1px solid #3fb950;
        border-radius:3px; padding:4px 12px; font-size:10px; color:#3fb950;
        margin:3px; font-family:'Share Tech Mono',monospace; letter-spacing:2px; }

/* ── INTEL REPORT ── */
.report-box {
    background:#08090f;
    border:1px solid #1a1f2e;
    border-left:3px solid #1f6feb;
    border-radius:4px;
    padding:16px 20px;
    font-size:13px; line-height:1.9;
    color:#8892a4;
}
.report-footer {
    font-family:'Share Tech Mono',monospace;
    font-size:9px; color:#1a1f2e;
    margin-top:10px; letter-spacing:1px;
}

/* ── DS TABLE ── */
.ds-source { font-family:'Share Tech Mono',monospace;
             font-size:9px; color:#2d4a6e;
             letter-spacing:3px; margin:12px 0 6px; }
.ds-row { display:flex; justify-content:space-between;
          padding:5px 0; border-bottom:1px solid #0f1320;
          font-size:11px; font-family:'Share Tech Mono',monospace; }
.ds-row:last-child { border-bottom:none; }
.ds-key { color:#3d4451; letter-spacing:1px; }
.ds-t { color:#f85149; } .ds-b { color:#3fb950; } .ds-u { color:#e3b341; }

/* ── PROGRESS BAR ── */
[data-testid="stProgress"] > div > div {
    border-radius:2px !important;
}
</style>
""", unsafe_allow_html=True)


# ── SIDEBAR ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("#### MISSION INPUT")
    st.divider()
    st.markdown("**LOCATION**")
    location = st.text_input("loc", value="Sector 4", label_visibility="collapsed")
    st.markdown("**VISION — MODULE 1**")

    uploaded_image = st.file_uploader(
        "Upload aerial image (optional)",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible"
    )

    # If image uploaded, run real YOLO detection
    vision_from_image = None
    if uploaded_image is not None:
        import tempfile
        import os
        from ultralytics import YOLO

        VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}
        PEOPLE_CLASSES  = {"person"}
        BOAT_CLASSES    = {"boat"}

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_image.read())
            tmp_path = tmp.name

        try:
            yolo_model = YOLO("yolov8n.pt")
            results    = yolo_model.predict(source=tmp_path,
                                            conf=0.4, verbose=False)
            result     = results[0]
            counts     = {"vehicles": 0, "people": 0, "boats": 0}

            if result.boxes is not None:
                for cls_id in result.boxes.cls.cpu().numpy().astype(int):
                    name = result.names[cls_id]
                    if name in VEHICLE_CLASSES:
                        counts["vehicles"] += 1
                    elif name in PEOPLE_CLASSES:
                        counts["people"]   += 1
                    elif name in BOAT_CLASSES:
                        counts["boats"]    += 1

            vision_from_image = counts
            st.image(tmp_path, caption="Uploaded image", use_container_width=True)
            st.success(f"Detected: {counts['vehicles']} vehicles, "
                       f"{counts['people']} persons, "
                       f"{counts['boats']} boats")
        except Exception as e:
            st.warning(f"Detection failed: {e}")
        finally:
            os.unlink(tmp_path)

    # Use real detection if available, otherwise use sliders
    if vision_from_image is not None:
        vehicle_count = vision_from_image["vehicles"]
        person_count  = vision_from_image["people"]
        st.markdown("*Counts from image detection — adjust manually if needed*")
    else:
        vehicle_count = st.slider("Vehicles detected", 0, 15, 3)
        person_count  = st.slider("Persons detected",  0, 15, 1)

    loitering = st.checkbox("Loitering detected", value=False)
    st.markdown("**CYBER — MODULE 3**")
    spoof_confidence = st.slider("GPS spoofing confidence",   0.0, 1.0, 0.05, step=0.01)
    st.markdown("**CONTEXT**")
    in_restricted_zone = st.checkbox("Restricted zone",       value=False)
    is_nighttime       = st.checkbox("Nighttime",             value=False)
    st.divider()
    st.caption("UAV-IFS v1.0  |  Non-weaponized platform")


# ── COMPUTE ───────────────────────────────────────────────────────
fusion_result = fuse(
    vehicle_count=vehicle_count, person_count=person_count,
    loitering=loitering, spoof_confidence=spoof_confidence,
    in_restricted_zone=in_restricted_zone, is_nighttime=is_nighttime
)
report  = generate_report(fusion_result, vehicle_count=vehicle_count,
                          person_count=person_count, location=location)
level   = fusion_result["threat_level"]
score   = fusion_result["threat_score"]
lo, hi  = fusion_result["confidence_range"]
factors = fusion_result["contributing_factors"]
ev      = fusion_result["evidence_breakdown"]

COLORS = {"HIGH":"#f85149","MEDIUM":"#e3b341","LOW":"#3fb950"}
tc = COLORS[level]
now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")


# ── TOP HEADER ───────────────────────────────────────────────────
st.markdown(f"""
<div class="top-header">
  <div class="header-left">
    <div class="header-title">UAV INTELLIGENCE FUSION SYSTEM</div>
    <div class="header-sub">
      SECURE SITUATIONAL AWARENESS PLATFORM &nbsp;//&nbsp;
      AI-POWERED MULTI-MODAL THREAT ASSESSMENT
    </div>
  </div>
  <div class="header-right">
    {now}<br>
    SECTOR: <b>{location.upper()}</b><br>
    SYSTEM STATUS: <b style="color:#3fb950">ACTIVE</b>
  </div>
</div>
""", unsafe_allow_html=True)


# ── MODULE STATUS BAR ────────────────────────────────────────────
m3c = "mod-warn" if spoof_confidence > 0.5 else "mod-ok"
m4c = {"HIGH":"mod-crit","MEDIUM":"mod-warn","LOW":"mod-ok"}[level]
st.markdown(f"""
<div class="mod-bar">
  <div class="mod-pill mod-ok">MODULE 1 &nbsp;|&nbsp; VISION &nbsp;—&nbsp; ONLINE</div>
  <div class="mod-pill mod-ok">MODULE 2 &nbsp;|&nbsp; EKF STATE EST &nbsp;—&nbsp; ONLINE</div>
  <div class="mod-pill {m3c}">MODULE 3 &nbsp;|&nbsp; CYBER ANOMALY &nbsp;—&nbsp; {'ALERT' if spoof_confidence>0.5 else 'ONLINE'}</div>
  <div class="mod-pill {m4c}">MODULE 4 &nbsp;|&nbsp; FUSION ENGINE &nbsp;—&nbsp; {level}</div>
</div>
""", unsafe_allow_html=True)


# ── ROW 1: Gauge | Evidence bars | Threat + Factors ──────────────
c1, c2, c3 = st.columns([1, 1.5, 1], gap="medium")

with c1:
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font":{"size":40,"color":tc,"family":"Share Tech Mono"},"suffix":""},
        gauge={
            "axis":{"range":[0,1],"tickwidth":1,"tickcolor":"#1a1f2e",
                    "tickfont":{"color":"#2d333b","size":9,"family":"Share Tech Mono"},
                    "nticks":6},
            "bar":{"color":tc,"thickness":0.2},
            "bgcolor":"#0b0e16","borderwidth":0,
            "steps":[
                {"range":[0.00,0.35],"color":"#021208"},
                {"range":[0.35,0.65],"color":"#141002"},
                {"range":[0.65,1.00],"color":"#160404"},
            ],
            "threshold":{"line":{"color":tc,"width":3},"thickness":0.75,"value":score}
        },
        title={"text":"THREAT SCORE",
               "font":{"size":9,"color":"#2d4a6e","family":"Share Tech Mono"}}
    ))
    fig_g.update_layout(
        paper_bgcolor="#0b0e16", plot_bgcolor="#0b0e16",
        height=230, margin=dict(l=20,r=20,t=30,b=0),
        font={"color":"#c9d1d9"}
    )
    st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar":False})
    st.markdown(f'<div class="score-range">CONFIDENCE &nbsp; {lo} – {hi}</div>', unsafe_allow_html=True)

with c2:
    cats = ["VISION","CYBER","CONTEXT"]
    t_v  = [ev["vision"]["threat"], ev["cyber"]["threat"], ev["context"]["threat"]]
    b_v  = [ev["vision"]["benign"], ev["cyber"]["benign"], ev["context"]["benign"]]
    u_v  = [ev["vision"]["uncertain"],ev["cyber"]["uncertain"],ev["context"]["uncertain"]]

    fig_e = go.Figure()
    for vals, name, color in [
        (t_v,"THREAT","#f85149"),
        (b_v,"BENIGN","#3fb950"),
        (u_v,"UNCERTAIN","#e3b341"),
    ]:
        fig_e.add_trace(go.Bar(
            name=name, y=cats, x=vals, orientation="h",
            marker_color=color, marker_line_width=0,
            hovertemplate=f"<b>{name}</b>: %{{x:.3f}}<extra></extra>"
        ))

    fig_e.update_layout(
        barmode="stack",
        paper_bgcolor="#0b0e16", plot_bgcolor="#08090f",
        height=230, margin=dict(l=10,r=10,t=30,b=10),
        xaxis=dict(range=[0,1], gridcolor="#0f1320", zeroline=False,
                   tickfont={"color":"#2d333b","size":9,"family":"Share Tech Mono"},
                   title=dict(text="BELIEF MASS",
                              font={"color":"#2d4a6e","size":9,"family":"Share Tech Mono"})),
        yaxis=dict(tickfont={"color":"#6e7a91","size":10,"family":"Share Tech Mono"},
                   gridcolor="#0f1320"),
        legend=dict(orientation="h",y=1.12,x=0,
                    font={"color":"#6e7a91","size":9,"family":"Share Tech Mono"},
                    bgcolor="rgba(0,0,0,0)"),
        title=dict(text="DEMPSTER-SHAFER EVIDENCE FUSION",
                   font={"size":9,"color":"#2d4a6e","family":"Share Tech Mono"},x=0),
        bargap=0.25,
    )
    st.plotly_chart(fig_e, use_container_width=True, config={"displayModeBar":False})

with c3:
    t_css = {"HIGH":"t-high","MEDIUM":"t-med","LOW":"t-low"}[level]
    st.markdown(f"""
    <div class="{t_css}">
        {level}
        <div class="t-sub">THREAT LEVEL</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="card-title">Active Indicators</div>', unsafe_allow_html=True)

    FACTOR_STYLE = {
        "gps_spoofing_likely":   "fp-h",
        "loitering_detected":    "fp-h",
        "restricted_zone":       "fp-m",
        "nighttime_activity":    "fp-m",
        "multi_vehicle_cluster": "fp-m",
    }
    if "no_significant_indicators" in factors:
        st.markdown('<span class="fp-l">NO INDICATORS</span>', unsafe_allow_html=True)
    else:
        for f in factors:
            cls = FACTOR_STYLE.get(f, "fp-l")
            st.markdown(f'<span class="{cls}">{f.replace("_"," ").upper()}</span>',
                        unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ── ROW 2: EKF RESIDUAL TIMELINE ─────────────────────────────────
st.markdown("""
<div class="card-title" style="margin-top:4px">
EKF INNOVATION RESIDUAL TIMELINE &nbsp;—&nbsp;
TELEMETRY ANOMALY DETECTION &nbsp;|&nbsp;
REAL EKF RUNNING ON SYNTHETIC GPS STREAM
</div>
""", unsafe_allow_html=True)

@st.cache_data
def ekf_timeline(spoof_conf):
    ekf = UAVKalmanFilter(dt=0.5)
    n, residuals, labels = 120, [], []
    lat, lon, alt = 28.6139, 77.2090, 400.0
    rng = np.random.default_rng(42)
    for i in range(n):
        if i >= 80 and spoof_conf > 0.3:
            lat += rng.normal(0.00015, 0.00005)
            lon += rng.normal(0.00015, 0.00005)
            alt += rng.normal(2.0, 0.5)
            labels.append("ANOMALY")
        else:
            lat += rng.normal(0.000008, 0.000003)
            lon += rng.normal(0.000008, 0.000003)
            alt += rng.normal(0, 0.4)
            labels.append("NORMAL")
        r = ekf.step(lat, lon, alt, 18.0, 90.0)
        residuals.append(r["innovation_residual"])
    return list(range(n)), residuals, labels

xs, resids, labels = ekf_timeline(spoof_confidence)

fig_ekf = go.Figure()
nx = [x for x,l in zip(xs,labels) if l=="NORMAL"]
ny = [y for y,l in zip(resids,labels) if l=="NORMAL"]
ax = [x for x,l in zip(xs,labels) if l=="ANOMALY"]
ay = [y for y,l in zip(resids,labels) if l=="ANOMALY"]

fig_ekf.add_trace(go.Scatter(
    x=nx, y=ny, mode="lines", name="NORMAL FLIGHT",
    line=dict(color="#3fb950",width=1.5),
    fill="tozeroy", fillcolor="rgba(63,185,80,0.04)"
))
if ax:
    fig_ekf.add_trace(go.Scatter(
        x=ax, y=ay, mode="lines", name="GPS ANOMALY DETECTED",
        line=dict(color="#f85149",width=2),
        fill="tozeroy", fillcolor="rgba(248,81,73,0.07)"
    ))
    # vertical line at anomaly start
    fig_ekf.add_vline(
        x=80, line_dash="dot", line_color="#f85149",
        line_width=1, opacity=0.4,
        annotation_text="ATTACK START",
        annotation_font={"size":8,"color":"#f85149","family":"Share Tech Mono"},
        annotation_position="top right"
    )

fig_ekf.add_hline(
    y=3.0, line_dash="dot", line_color="#e3b341", line_width=1,
    annotation_text="ALERT THRESHOLD (σ=3)",
    annotation_font={"size":8,"color":"#e3b341","family":"Share Tech Mono"},
    annotation_position="right"
)

fig_ekf.update_layout(
    paper_bgcolor="#0b0e16", plot_bgcolor="#08090f",
    height=190, margin=dict(l=10,r=100,t=8,b=30),
    xaxis=dict(gridcolor="#0f1320", zeroline=False,
               tickfont={"color":"#2d333b","size":9,"family":"Share Tech Mono"},
               title=dict(text="TIMESTEP (0.5s INTERVALS)",
                          font={"color":"#2d4a6e","size":9,"family":"Share Tech Mono"})),
    yaxis=dict(gridcolor="#0f1320", zeroline=False,
               tickfont={"color":"#2d333b","size":9,"family":"Share Tech Mono"},
               title=dict(text="MAHALANOBIS RESIDUAL",
                          font={"color":"#2d4a6e","size":9,"family":"Share Tech Mono"})),
    legend=dict(orientation="h",y=1.08,x=0,
                font={"color":"#6e7a91","size":9,"family":"Share Tech Mono"},
                bgcolor="rgba(0,0,0,0)"),
    hovermode="x unified",
)
st.plotly_chart(fig_ekf, use_container_width=True, config={"displayModeBar":False})


# ── ROW 3: Report | DS Raw Masses ────────────────────────────────
r1, r2 = st.columns([1.8, 1], gap="medium")

with r1:
    st.markdown(f"""
    <div class="card">
      <div class="card-title">Intelligence Report — Auto-Generated from Fusion Output</div>
      <div class="report-box">{report["report"]}</div>
      <div class="report-footer">
        GENERATED: {now} &nbsp;//&nbsp;
        CLASSIFICATION: UNCLASSIFIED &nbsp;//&nbsp;
        PLATFORM: UAV-IFS v1.0 &nbsp;//&nbsp;
        METHOD: DEMPSTER-SHAFER EVIDENCE THEORY
      </div>
    </div>
    """, unsafe_allow_html=True)
with r2:
    st.markdown('<div class="card"><div class="card-title">Raw Evidence Masses — DS Theory</div>', unsafe_allow_html=True)
    for src, label in [("vision","VISION"), ("cyber","CYBER"), ("context","CONTEXT")]:
        d = ev[src]
        st.markdown(f"""
        <div class="ds-source">{label}</div>
        <div class="ds-row">
            <span class="ds-key">m( THREAT )</span>
            <span class="ds-t">{d['threat']}</span>
        </div>
        <div class="ds-row">
            <span class="ds-key">m( BENIGN )</span>
            <span class="ds-b">{d['benign']}</span>
        </div>
        <div class="ds-row">
            <span class="ds-key">m( UNKNOWN )</span>
            <span class="ds-u">{d['uncertain']}</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)