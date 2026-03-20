# ==========================================================
#  BCA STUDENT CHURN ANALYSIS -- Complete Data Science App
#  Font   : Times New Roman
#  Run    : streamlit run churn_app.py
#  Model  : CatBoost / 45 features (no current_semester)
#           Features: gender, fees_type, year_gap, College_enc,
#           exam_hsc/ssc, spec_*, perf_bracket, board_*, cast_*,
#           rel_*, dist_* (18 named + dist_other),
#           perf_x_cast_obc, perf_x_cast_scst, perf_x_cast_open
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report, roc_curve)
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------------------------------------
#  FOLDER -- change to your folder
# ----------------------------------------------------------
FOLDER = r""
# ----------------------------------------------------------

st.set_page_config(
    page_title="BCA Student Churn Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
#  COLOUR PALETTE
# ==========================================================
GOLD  = "#C8A96E"
NAVY  = "#1C2B3A"
GREEN = "#2D6A4F"
RED   = "#8B2525"
AMBER = "#9B6A1A"
CREAM = "#FAFAF7"

def rgba(hex_color, alpha=0.10):
    """Convert #RRGGBB to rgba(r,g,b,alpha) -- Plotly-safe."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# ==========================================================
#  CSS
# ==========================================================
st.markdown(f"""
<style>
/* 1. GLOBAL FONT */
*, *::before, *::after {{
    font-family: "Times New Roman", Times, serif !important;
    box-sizing: border-box;
}}
/* 2. PAGE BG */
.stApp {{ background-color: {CREAM}; }}

/* 3. SIDEBAR */
[data-testid="stSidebar"] {{
    background-color: {NAVY};
    border-right: 4px solid {GOLD};
}}
[data-testid="stSidebar"] * {{
    color: #EDE8DC !important;
    font-family: "Times New Roman", Times, serif !important;
}}
[data-testid="stSidebar"] .stRadio label {{
    font-size: 14px !important;
    padding: 5px 0;
    display: block;
    letter-spacing: 0.3px;
}}
[data-testid="stSidebar"] hr {{
    border-color: rgba(200,169,110,0.35) !important;
    margin: 10px 0;
}}

/* 4. HEADINGS */
h1 {{
    font-size: 2.1rem !important;
    font-weight: 700 !important;
    color: {NAVY} !important;
    border-bottom: 3px solid {GOLD};
    padding-bottom: 9px;
    margin-bottom: 8px;
}}
h2 {{
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: {NAVY} !important;
    margin-top: 16px !important;
}}
h3 {{
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: #2D4A5E !important;
}}

/* 5. METRIC CARDS */
[data-testid="metric-container"] {{
    background: #FFFFFF;
    border: 1px solid #D4C5A9;
    border-top: 4px solid {GOLD};
    border-radius: 0;
    padding: 14px 16px;
    box-shadow: 2px 2px 7px rgba(0,0,0,0.06);
}}
[data-testid="stMetricValue"] {{
    color: {NAVY} !important;
    font-size: 1.85rem !important;
    font-weight: 700 !important;
}}
[data-testid="stMetricLabel"] {{
    color: #5C6B7A !important;
    font-size: 0.76rem !important;
    text-transform: uppercase;
    letter-spacing: 1.2px;
}}

/* 6. TABS */
.stTabs [data-baseweb="tab-list"] {{
    background: {NAVY};
    border-radius: 0;
    gap: 0;
    padding: 0;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent;
    color: {GOLD} !important;
    font-size: 12.5px !important;
    font-weight: 600;
    letter-spacing: 0.4px;
    padding: 10px 18px;
    border-right: 1px solid rgba(200,169,110,0.2);
    border-radius: 0;
}}
.stTabs [aria-selected="true"] {{
    background: {GOLD} !important;
    color: {NAVY} !important;
}}

/* 7. BUTTONS */
.stButton > button,
[data-testid="stFormSubmitButton"] > button {{
    background-color: {NAVY};
    color: {GOLD};
    border: 2px solid {GOLD};
    border-radius: 0;
    font-size: 13.5px;
    font-weight: 700;
    padding: 11px 26px;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    transition: background-color 0.18s, color 0.18s;
    width: 100%;
}}
.stButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {{
    background-color: {GOLD};
    color: {NAVY};
}}

/* 8. FORM INPUTS */
.stSelectbox > label,
.stNumberInput > label,
.stSlider > label {{
    font-weight: 700 !important;
    color: {NAVY} !important;
    font-size: 11.5px !important;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 2px;
}}
.stSelectbox [data-baseweb="select"] > div {{
    border-radius: 0 !important;
    border-color: #D4C5A9 !important;
    font-size: 13px;
}}

/* 9. DATAFRAME */
.stDataFrame {{ border: 1px solid #D4C5A9; }}

/* 10. ALERTS & DIVIDERS */
.stAlert {{ border-radius: 0; }}
hr {{ border-color: #D4C5A9 !important; margin: 16px 0; }}

/* 11. PLOTLY */
.stPlotlyChart {{
    border: 1px solid #D4C5A9;
    background: #FFFFFF;
    padding: 3px;
}}
</style>
""", unsafe_allow_html=True)


# ==========================================================
#  UI HELPERS
# ==========================================================
def section_header(title, subtitle=""):
    sub = (f"<div style='font-size:12px;color:#5C6B7A;margin-top:4px'>{subtitle}</div>"
           if subtitle else "")
    st.markdown(f"""
    <div style='border-left:5px solid {GOLD};padding:9px 17px;
         background:#FFFFFF;margin-bottom:17px;
         box-shadow:2px 2px 5px rgba(0,0,0,0.05)'>
      <div style='font-size:18px;font-weight:700;color:{NAVY}'>{title}</div>
      {sub}
    </div>""", unsafe_allow_html=True)


def info_card(html_text, color=NAVY):
    st.markdown(f"""
    <div style='background:#FFFFFF;border:1px solid #D4C5A9;
         border-left:5px solid {color};padding:12px 16px;
         margin-bottom:10px;font-size:13px;color:#2C3E50;line-height:1.75'>
      {html_text}
    </div>""", unsafe_allow_html=True)


def section_banner(text):
    st.markdown(f"""
    <div style='background:{NAVY};color:{GOLD};font-size:14px;font-weight:700;
         letter-spacing:1.3px;text-transform:uppercase;
         padding:10px 20px;margin:22px 0 16px 0'>
      &#9672; &nbsp; {text}
    </div>""", unsafe_allow_html=True)


def tnr_fig(fig, h=360):
    fig.update_layout(
        height=h,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FEFEFE",
        font=dict(family="Times New Roman, Times, serif", color=NAVY, size=12),
        title_font=dict(family="Times New Roman, Times, serif", color=NAVY, size=14),
        xaxis=dict(gridcolor="#EAE3D5", linecolor=GOLD,
                   tickfont=dict(family="Times New Roman", size=11)),
        yaxis=dict(gridcolor="#EAE3D5", linecolor=GOLD,
                   tickfont=dict(family="Times New Roman", size=11)),
        legend=dict(bgcolor="rgba(255,255,255,0.95)",
                    bordercolor=GOLD, borderwidth=1,
                    font=dict(family="Times New Roman", size=11)),
        margin=dict(t=48, b=40, l=50, r=20),
    )
    return fig


def get_risk(p):
    if   p < 0.20: return "Low Risk",       "#2D6A4F"
    elif p < 0.40: return "Moderate Risk",  "#9B6A1A"
    elif p < 0.65: return "High Risk",      "#8B2525"
    else:          return "Very High Risk", "#5C0A0A"


def form_label(text):
    st.markdown(f"""
    <div style='font-size:11px;font-weight:700;color:{GOLD};
         text-transform:uppercase;letter-spacing:0.8px;
         border-bottom:1px solid rgba(200,169,110,0.5);
         padding-bottom:4px;margin:14px 0 10px 0'>{text}</div>
    """, unsafe_allow_html=True)


# ==========================================================
#  FEATURE ENGINEERING -- matches what the model was trained on
# ==========================================================

# Districts that have their own column in the model
NAMED_DISTRICTS = [
    "AHMEDABAD", "AMRELI", "ARVALLI", "BANASKANTHA", "BHAVNAGAR",
    "BOTAD", "GANDHINAGAR", "GIR SOMNATH", "JAMNAGAR", "JUNAGADH",
    "KHEDA", "KUTCH", "MEHSANA", "MORBI", "PATAN",
    "RAJKOT", "SABARKANTHA", "SURENDRANAGAR",
]

# Districts that fold into dist_other = 1
OTHER_DISTRICTS = [
    "ANAND", "AURANGABAD", "DAHOD", "DEVBHUMI DWARKA", "DUNGARPUR",
    "GONDIYA", "HALVAD", "HIMMATNAGAR", "MAHARASHTRA", "MAHISAGAR",
    "NAVSARI", "PORBANDAR", "SIROHI", "SURAT",
]

ALL_DISTRICTS = sorted(NAMED_DISTRICTS + OTHER_DISTRICTS)


def build_feature_row(gender, college, year_gap,
                      perf_v, spec, exam_type, board,
                      caste, religion, district):
    """
    Build a dict with exactly the 45 features the model expects.
    Mirrors the feature engineering done during notebook training.
    """
    # -- Basic encodings ------------------------------------
    g_enc    = 0 if gender == "Male" else 1
    fees_enc = 0 if "BPCCS" in college else 1
    col_enc  = 0 if "BPCCS" in college else 1

    # -- Exam flags -----------------------------------------
    hsc_enc  = 1 if exam_type == "HSC" else 0
    ssc_enc  = 1 if exam_type == "SSC" else 0

    # -- Stream one-hot --------------------------------------
    arts_enc = 1 if spec == "ARTS"     else 0
    comm_enc = 1 if spec == "COMMERCE" else 0
    sci_enc  = 1 if spec == "SCIENCE"  else 0

    # -- Board flags -----------------------------------------
    bg_enc   = 1 if board in ["G.H.S.E.B", "GSEB"] else 0
    bcbse    = 1 if board == "CBSE"      else 0
    bghseb   = 1 if board == "G.H.S.E.B" else 0
    bgseb    = 1 if board == "GSEB"      else 0
    both_enc = 1 if board == "Other"     else 0

    # -- Caste one-hot ---------------------------------------
    obc_enc  = 1 if caste == "OBC"  else 0
    open_enc = 1 if caste == "OPEN" else 0
    scst_enc = 1 if caste == "SCST" else 0
    sebc_enc = 1 if caste == "SEBC" else 0

    # -- Religion one-hot ------------------------------------
    rchr_enc = 1 if religion == "Christian" else 0
    rhin_enc = 1 if religion == "Hindu"     else 0
    rjai_enc = 1 if religion == "Jain"      else 0
    rmus_enc = 1 if religion == "Muslim"    else 0

    # -- District flags --------------------------------------
    dist_vals = {
        f"dist_{d.lower().replace(' ', '_')}": (1 if d == district else 0)
        for d in NAMED_DISTRICTS
    }
    # dist_other = 1 when student is from any "other" district
    dist_vals["dist_other"] = 1 if district in OTHER_DISTRICTS else 0

    # -- Interaction features (trained as part of model) -----
    perf_x_obc  = perf_v * obc_enc
    perf_x_scst = perf_v * scst_enc
    perf_x_open = perf_v * open_enc

    row = {
        "gender":          g_enc,
        "fees_type":       fees_enc,
        "year_gap":        year_gap,
        "College_enc":     col_enc,
        "exam_hsc":        hsc_enc,
        "exam_ssc":        ssc_enc,
        "spec_arts":       arts_enc,
        "spec_commerce":   comm_enc,
        "spec_science":    sci_enc,
        "perf_bracket":    perf_v,
        "board_gseb_group": bg_enc,
        "board_cbse":      bcbse,
        "board_ghseb":     bghseb,
        "board_gseb":      bgseb,
        "board_other":     both_enc,
        "cast_obc":        obc_enc,
        "cast_open":       open_enc,
        "cast_scst":       scst_enc,
        "cast_sebc":       sebc_enc,
        "rel_christian":   rchr_enc,
        "rel_hindu":       rhin_enc,
        "rel_jain":        rjai_enc,
        "rel_muslim":      rmus_enc,
        **dist_vals,
        "perf_x_cast_obc":  perf_x_obc,
        "perf_x_cast_scst": perf_x_scst,
        "perf_x_cast_open": perf_x_open,
    }
    return row


def apply_feature_engineering(df):
    """
    Apply the same engineering to the binary CSV for get_test_set().
    Adds dist_other and interaction features so feats aligns with model.
    """
    df = df.copy()

    # dist_other: 1 if none of the named district columns are 1
    named_cols = [f"dist_{d.lower().replace(' ', '_')}" for d in NAMED_DISTRICTS]
    existing_named = [c for c in named_cols if c in df.columns]
    df["dist_other"] = (df[existing_named].sum(axis=1) == 0).astype(int)

    # Interaction features
    df["perf_x_cast_obc"]  = df["perf_bracket"] * df["cast_obc"]
    df["perf_x_cast_scst"] = df["perf_bracket"] * df["cast_scst"]
    df["perf_x_cast_open"] = df["perf_bracket"] * df["cast_open"]

    return df


# ==========================================================
#  DATA & MODEL LOADING
# ==========================================================
@st.cache_data
def load_raw():
    df = pd.read_csv(os.path.join(FOLDER, "latest.csv"))
    df["is_churned"]  = df["student_status_new"].apply(
        lambda x: 1 if "dropout" in str(x).lower() else 0)
    df["Churn Label"] = df["is_churned"].map({0: "Active", 1: "Churned"})
    return df


@st.cache_data
def load_binary():
    return pd.read_csv(os.path.join(FOLDER, "SIMPLE_Fixed_binary.csv"))


@st.cache_resource
def load_model():
    m  = joblib.load(os.path.join(FOLDER, "churn_model.pkl"))
    sc = joblib.load(os.path.join(FOLDER, "churn_scaler.pkl"))
    th = joblib.load(os.path.join(FOLDER, "churn_threshold.pkl"))
    ft = joblib.load(os.path.join(FOLDER, "churn_feature_names.pkl"))
    return m, sc, th, ft


@st.cache_data
def get_test_set(feats):
    """
    Reconstruct the exact test set the model was evaluated on.
    Applies feature engineering before selecting feats.
    """
    df2 = load_binary()
    df2 = apply_feature_engineering(df2)   # adds dist_other + interactions
    X   = df2[feats]                        # now all 45 features exist
    y   = df2["is_churned"]
    _, Xt, _, yt = train_test_split(X, y, test_size=0.30,
                                    random_state=42, stratify=y)
    return Xt, yt


try:
    raw    = load_raw()
    binary = load_binary()
    model, scaler, thresh, feats = load_model()
    X_test, y_test = get_test_set(feats)
    Xs_test   = scaler.transform(X_test)
    all_probs = model.predict_proba(Xs_test)[:, 1]
    all_preds = (all_probs >= thresh).astype(int)
    LOADED    = True
except Exception as e:
    LOADED = False
    ERR    = str(e)


# ==========================================================
#  SIDEBAR
# ==========================================================
with st.sidebar:
    st.markdown(f"""
    <div style='padding:24px 18px 14px;text-align:center;
         border-bottom:2px solid rgba(200,169,110,0.4)'>
      <div style='font-size:36px;margin-bottom:6px'>&#127891;</div>
      <div style='font-size:18px;font-weight:700;color:{GOLD};letter-spacing:1px'>
        BCA CHURN</div>
      <div style='font-size:10px;color:#8FA3B8;margin-top:4px;
           letter-spacing:2.5px;text-transform:uppercase'>Analysis Project</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    page = st.radio("", [
        "I.    Introduction",
        "II.   Dataset Overview",
        "III.  Data Cleaning",
        "IV.   EDA",
        "V.    Feature Engineering",
        "VI.   Model & Prediction",
    ], label_visibility="collapsed")

    if LOADED:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='font-size:11px;color:#8FA3B8;padding:0 4px;line-height:2.2'>
          <span style='color:{GOLD}'>&#9632;</span>&nbsp; Model: {type(model).__name__}<br>
          <span style='color:{GOLD}'>&#9632;</span>&nbsp; Threshold: {thresh:.2f}<br>
          <span style='color:{GOLD}'>&#9632;</span>&nbsp; Features: {len(feats)}<br>
          <span style='color:{GOLD}'>&#9632;</span>&nbsp; Dataset: 842 students
        </div>""", unsafe_allow_html=True)

if not LOADED:
    st.error(f"**Files not found or error loading model in** `{FOLDER}`\n\n`{ERR}`")
    st.stop()


# ==========================================================
#  I. INTRODUCTION
# ==========================================================
if "Introduction" in page:
    st.title("BCA Student Churn Analysis")
    st.markdown("<p style='font-size:15px;color:#5C6B7A;font-style:italic;margin-bottom:20px'>"
                "A complete data science study on student dropout patterns in BCA -- Batch 2023-24.</p>",
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students",   "842")
    c2.metric("Active Students",  "735")
    c3.metric("Churned Students", "107")
    c4.metric("Churn Rate",       "12.7%")

    st.divider()
    col_l, col_r = st.columns([3, 2])

    with col_l:
        section_header("What is Student Churn?")
        info_card(
            "In higher education, <b>student churn</b> refers to students who enrolled "
            "in the BCA programme but did not complete it -- either by formally cancelling "
            "their admission or stopping attendance after a semester.<br><br>"
            "Early identification enables the institution to intervene with targeted "
            "academic support, financial counselling, or peer mentorship before the "
            "student makes the irreversible decision to leave.",
            color=GOLD)

        section_header("Project Objectives")
        for obj in [
            "Perform comprehensive EDA on 842 BCA students.",
            "Engineer meaningful features from raw admission-time data.",
            "Train a machine learning model to predict dropout probability.",
            "Build an interactive prediction interface for institutional use.",
            "Deliver honest, accurate results with all model features.",
        ]:
            st.markdown(f"""
            <div style='font-size:13px;color:#2C3E50;padding:6px 0;
                 border-bottom:1px dotted #D4C5A9'>
              <span style='color:{GOLD};font-weight:700'>&#9658;</span>&nbsp; {obj}
            </div>""", unsafe_allow_html=True)

    with col_r:
        fig_d = go.Figure(go.Pie(
            labels=["Active (735)", "Churned (107)"],
            values=[735, 107], hole=0.60,
            marker=dict(colors=[GREEN, RED], line=dict(color="#FFF", width=3)),
            textinfo="label+percent",
            textfont=dict(family="Times New Roman", size=12),
        ))
        fig_d.add_annotation(
            text="<b>842</b><br><span style='font-size:11px'>Students</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, family="Times New Roman", color=NAVY))
        tnr_fig(fig_d, 280)
        fig_d.update_layout(title="Class Distribution",
                            legend=dict(orientation="h", y=-0.06))
        st.plotly_chart(fig_d, use_container_width=True)

        st.markdown(f"""
        <div style='background:#FFF;border:1px solid #D4C5A9;padding:15px 17px'>
          <div style='font-weight:700;font-size:13.5px;color:{NAVY};
               border-bottom:2px solid {GOLD};padding-bottom:6px;margin-bottom:10px'>
            Dropout Types
          </div>
          <div style='font-size:12px;line-height:2.3'>
            <span style='color:{RED};font-weight:700'>dropout_sem1</span>
            -- 107 students &rarr; <code>Churn = 1</code><br>
            Left during Semester 1 (formal cancellation)<br><br>
            <span style='color:{AMBER};font-weight:700'>dropout_mid</span>
            -- 238 students &rarr; <code>Churn = 0</code><br>
            Stopped mid-programme (active in binary label)<br><br>
            <span style='color:{GREEN};font-weight:700'>active</span>
            -- 497 students &rarr; <code>Churn = 0</code><br>
            Currently enrolled and continuing
          </div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    section_header("Project Methodology")
    steps = [
        ("I",   "Data Collection",     "Raw records from BPCCS and SVICS-G"),
        ("II",  "Data Cleaning",       "Nulls, typos, standardisation"),
        ("III", "EDA",                 "Univariate and Bivariate analysis"),
        ("IV",  "Feature Engineering", "Encoding, binning, interaction terms"),
        ("V",   "Model Training",      "CatBoost + SMOTE, 70/30 split"),
        ("VI",  "Prediction App",      "Interactive churn prediction UI"),
    ]
    cols = st.columns(6)
    for col, (n, t, d) in zip(cols, steps):
        col.markdown(f"""
        <div style='background:{NAVY};padding:14px 8px;text-align:center;
             border-top:4px solid {GOLD}'>
          <div style='font-size:18px;font-weight:700;color:{GOLD}'>{n}</div>
          <div style='font-size:10px;font-weight:700;color:#EDE8DC;
               margin:4px 0;text-transform:uppercase;letter-spacing:0.5px'>{t}</div>
          <div style='font-size:9.5px;color:#8FA3B8;line-height:1.5'>{d}</div>
        </div>""", unsafe_allow_html=True)


# ==========================================================
#  II. DATASET OVERVIEW
# ==========================================================
elif "Dataset" in page:
    st.title("Dataset Overview")
    st.markdown("<p style='font-size:14px;color:#5C6B7A;font-style:italic'>"
                "Structure, types, and completeness of the raw dataset -- latest.csv</p>",
                unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["  Column Guide  ", "  Sample Data  ", "  Class Balance  "])

    with t1:
        section_header("Column Reference Table")
        col_guide = [
            ("Roll No",                 "ID",         "Unique student identifier -- not used in model"),
            ("Institute",               "Nominal",    "BPCCS (588 students) . SVICS-G (254 students)"),
            ("Current Semester",        "Ordinal",    "Raw dataset: 1-6 . NOT used in model (admission-time model)"),
            ("Gender",                  "Binary",     "Male . Female"),
            ("Admission Cast Category", "Nominal",    "OPEN . OBC . SEBC . SCST"),
            ("Religion",                "Nominal",    "Hindu . Muslim . Christian . Jain"),
            ("Permanent District",      "Nominal",    "Gujarat district of student's home address"),
            ("Total Fees",              "Numeric",    "Rs.18,000 (BPCCS) . Rs.27,000 (SVICS-G)"),
            ("Last Exam",               "Binary",     "HSC (833 students) . SSC (9 students)"),
            ("Last Exam Percentage",    "Continuous", "0-100 . mean approx 61.1%"),
            ("Last Exam Passing",       "Ordinal",    "Academic year e.g. 2022-23"),
            ("Last Exam Board/Uni.",    "Nominal",    "G.H.S.E.B . CBSE . GSEB . Other"),
            ("Specialisation",          "Nominal",    "COMMERCE . SCIENCE . ARTS"),
            ("student_status_new",      "Target",     "active . dropout_sem1 . dropout_mid"),
        ]
        tc = {"ID":"#8FA3B8","Nominal":GOLD,"Binary":GREEN,
              "Ordinal":AMBER,"Continuous":RED,"Target":NAVY,"Numeric":"#5C6B7A"}
        for col, dtype, desc in col_guide:
            cc = tc.get(dtype, "#888")
            st.markdown(f"""
            <div style='display:flex;align-items:center;padding:8px 12px;
                 background:#FFF;border-bottom:1px solid #EDE8DC'>
              <code style='min-width:220px;font-size:12.5px;color:{NAVY};
                    font-weight:700'>{col}</code>
              <span style='min-width:90px;background:{cc};color:#FAFAF7;
                    font-size:10px;font-weight:700;padding:2px 7px;
                    text-transform:uppercase;letter-spacing:0.5px'>{dtype}</span>
              <span style='color:#4A5568;font-size:13px;padding-left:12px'>{desc}</span>
            </div>""", unsafe_allow_html=True)

    with t2:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Rows",    "842")
        c2.metric("Total Columns", "14")
        c3.metric("Missing Values","0 (clean)")
        st.divider()
        st.markdown("#### First 20 Records")
        st.dataframe(raw.head(20), use_container_width=True)
        st.markdown("#### Descriptive Statistics")
        nc = ["Current Semester", "Last Exam Percentage", "Total Fees", "is_churned"]
        st.dataframe(raw[nc].describe().round(2), use_container_width=True)

    with t3:
        section_header("Class Balance Analysis")
        ca, cb = st.columns(2)
        with ca:
            dd = raw["student_status_new"].value_counts().reset_index()
            dd.columns = ["Status", "Count"]
            fig_b = px.bar(dd, x="Status", y="Count", color="Status",
                           color_discrete_map={"active":GREEN,"dropout_sem1":RED,"dropout_mid":AMBER},
                           text="Count", title="Student Status Breakdown")
            fig_b.update_traces(textposition="outside")
            tnr_fig(fig_b, 320); fig_b.update_layout(showlegend=False)
            st.plotly_chart(fig_b, use_container_width=True)
        with cb:
            info_card(f"""
            <b>842 total students:</b><br><br>
            <b style='color:{GREEN}'>497 Active</b> -- 59.0% still enrolled<br>
            <b style='color:{AMBER}'>238 dropout_mid</b> -- 28.3% left mid-programme<br>
            <b style='color:{RED}'>107 dropout_sem1</b> -- 12.7% left in Semester 1<br><br>
            Binary target <b>is_churned</b>: dropout_sem1 = 1, others = 0<br>
            <b>87.3% vs 12.7%</b> imbalance -- handled via SMOTE.
            """, color=GOLD)
            sd = raw["Current Semester"].value_counts().sort_index().reset_index()
            sd.columns = ["Semester", "Count"]
            fig_s = px.bar(sd, x="Semester", y="Count", color="Count",
                           color_continuous_scale=[[0,RED],[0.5,AMBER],[1,GREEN]],
                           title="Students per Semester", text="Count")
            fig_s.update_traces(textposition="outside")
            tnr_fig(fig_s, 240); fig_s.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_s, use_container_width=True)


# ==========================================================
#  III. DATA CLEANING
# ==========================================================
elif "Cleaning" in page:
    st.title("Data Cleaning")
    st.markdown("<p style='font-size:14px;color:#5C6B7A;font-style:italic'>"
                "Steps taken to standardise and validate the raw dataset.</p>",
                unsafe_allow_html=True)

    section_header("Cleaning Steps Applied")
    steps_list = [
        ("1","Null Value Check",
         "All 14 columns had zero null values after merging the master file with Book1.csv.", GREEN),
        ("2","District Name Standardisation",
         "35+ spelling variations corrected: GANDHIANAGR to GANDHINAGAR, AHEMDABAD to AHMEDABAD, etc.", AMBER),
        ("3","Exam Passing Year Format Fix",
         "'Mar-23', 'Jul-22' converted to '2022-23', '2023-24'. Outlier '1999-00' replaced with column mode.", AMBER),
        ("4","Board/University Standardisation",
         "G.S.E.B and G.S.E.B. unified to GSEB. NIOS, RBSE, MSBSHSE merged into 'Other'.", AMBER),
        ("5","Specialisation Typo Fix",
         "COMERECE to COMMERCE, COMM to COMMERCE, COMERCE to COMMERCE, '-' to COMMERCE.", AMBER),
        ("6","Caste Category Merge",
         "SC and ST merged into SCST per government reservation category structure.", AMBER),
        ("7","Target Column Creation",
         "is_churned: dropout_sem1 = 1 (Churned), active/dropout_mid = 0 (Active).", NAVY),
        ("8","Redundant Column Removal",
         "Sr No, Name, Admission Date, Registration Date, Birth Date removed -- no predictive value.", RED),
        ("9","District Grouping",
         "14 low-frequency districts (Anand, Surat, Navsari, etc.) merged into dist_other flag.", AMBER),
        ("10","Interaction Features",
         "perf_x_cast_obc, perf_x_cast_scst, perf_x_cast_open created as perf_bracket x caste flag.", NAVY),
    ]
    for num, title, desc, color in steps_list:
        st.markdown(f"""
        <div style='display:flex;align-items:flex-start;background:#FFF;
             border:1px solid #D4C5A9;border-left:5px solid {color};
             padding:13px 17px;margin-bottom:8px'>
          <div style='font-size:21px;font-weight:700;color:{color};
               min-width:40px;padding-top:1px'>{num}.</div>
          <div style='padding-left:12px'>
            <div style='font-weight:700;font-size:14px;color:{NAVY};margin-bottom:3px'>{title}</div>
            <div style='font-size:13px;color:#4A5568;line-height:1.6'>{desc}</div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    section_header("Before vs After -- District Names")
    ca, cb = st.columns(2)
    with ca:
        st.markdown(f"""
        <div style='background:rgba(139,37,37,0.07);border:1px solid rgba(139,37,37,0.3);
             padding:14px 17px'>
          <div style='font-weight:700;color:{RED};margin-bottom:8px'>Before (raw)</div>
          <code style='font-size:12px;color:#4A5568;display:block;line-height:2'>
            GANDHIANAGR . AHEMDABAD<br>MAHESANA . BANSKANTHA<br>
            HIMATNAGAR . GIRSOMNATH<br>KACHCHH . JAMNAGR
          </code>
        </div>""", unsafe_allow_html=True)
    with cb:
        st.markdown(f"""
        <div style='background:rgba(45,106,79,0.07);border:1px solid rgba(45,106,79,0.3);
             padding:14px 17px'>
          <div style='font-weight:700;color:{GREEN};margin-bottom:8px'>After (standardised)</div>
          <code style='font-size:12px;color:#4A5568;display:block;line-height:2'>
            GANDHINAGAR . AHMEDABAD<br>MEHSANA . BANASKANTHA<br>
            HIMMATNAGAR . GIR SOMNATH<br>KUTCH . JAMNAGAR
          </code>
        </div>""", unsafe_allow_html=True)


# ==========================================================
#  IV. EDA
# ==========================================================
elif "EDA" in page:
    st.title("Exploratory Data Analysis")
    st.markdown("<p style='font-size:14px;color:#5C6B7A;font-style:italic'>"
                "Univariate distributions and bivariate relationships with churn.</p>",
                unsafe_allow_html=True)

    # -- SECTION A: UNIVARIATE ------------------------------
    section_banner("SECTION A  --  UNIVARIATE ANALYSIS")

    u1, u2, u3 = st.tabs([
        "  Categorical Features  ",
        "  Numerical Features  ",
        "  Geographic  ",
    ])

    with u1:
        section_header("Distribution of Each Categorical Feature")

        ca, cb = st.columns(2)
        with ca:
            inst = raw["Institute"].value_counts().reset_index()
            inst.columns = ["Institute", "Count"]
            fig = px.bar(inst, x="Institute", y="Count", text="Count", color="Institute",
                         color_discrete_map={"BPCCS":NAVY,"SVICS-G":GOLD},
                         title="Students per Institute")
            fig.update_traces(textposition="outside")
            tnr_fig(fig, 285); fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            info_card("BPCCS 588 (69.8%) at Rs.18,000. SVICS-G 254 (30.2%) at Rs.27,000.", color=NAVY)

        with cb:
            gen = raw["Gender"].value_counts().reset_index()
            gen.columns = ["Gender", "Count"]
            fig2 = go.Figure(go.Pie(
                labels=gen["Gender"], values=gen["Count"], hole=0.5,
                marker=dict(colors=[NAVY, GOLD], line=dict(color="#FFF", width=2)),
                textinfo="label+percent", textfont=dict(family="Times New Roman", size=12)
            ))
            tnr_fig(fig2, 285); fig2.update_layout(title="Gender Distribution")
            st.plotly_chart(fig2, use_container_width=True)
            info_card("Male 64.7%, Female 35.3%. Male churn rate slightly higher (13.6% vs 11.1%).", color=NAVY)

        ca2, cb2 = st.columns(2)
        with ca2:
            cast = raw["Admission Cast Category"].value_counts().reset_index()
            cast.columns = ["Caste", "Count"]
            fig3 = px.bar(cast, x="Count", y="Caste", orientation="h", text="Count",
                          color="Caste", color_discrete_sequence=[NAVY,GOLD,GREEN,RED],
                          title="Students by Caste Category")
            fig3.update_traces(textposition="outside")
            tnr_fig(fig3, 285); fig3.update_layout(showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)
            info_card("OPEN is largest (470). OBC (94) shows the highest churn rate at 18.1%.", color=AMBER)

        with cb2:
            rel = raw["Religion"].value_counts().reset_index()
            rel.columns = ["Religion", "Count"]
            fig4 = px.bar(rel, x="Religion", y="Count", text="Count", color="Religion",
                          color_discrete_sequence=[NAVY,GOLD,GREEN,RED],
                          title="Religious Diversity")
            fig4.update_traces(textposition="outside")
            tnr_fig(fig4, 285); fig4.update_layout(showlegend=False)
            st.plotly_chart(fig4, use_container_width=True)
            info_card("Hindu 766, Muslim 69, Christian 4, Jain 3.", color=NAVY)

        ca3, cb3 = st.columns(2)
        with ca3:
            spec = raw["Specialisation"].value_counts().reset_index()
            spec.columns = ["Specialisation", "Count"]
            fig5 = go.Figure(go.Pie(
                labels=spec["Specialisation"], values=spec["Count"], hole=0.5,
                marker=dict(colors=[NAVY,GOLD,GREEN], line=dict(color="#FFF",width=2)),
                textinfo="label+percent", textfont=dict(family="Times New Roman",size=12)
            ))
            tnr_fig(fig5, 285); fig5.update_layout(title="12th Specialisation")
            st.plotly_chart(fig5, use_container_width=True)
            info_card("Commerce 726 (86.2%). ARTS shows slightly higher churn risk.", color=AMBER)

        with cb3:
            board = raw["Last Exam Board/Uni."].value_counts().reset_index()
            board.columns = ["Board", "Count"]
            fig6 = px.bar(board, x="Count", y="Board", orientation="h", text="Count",
                          color="Count",
                          color_continuous_scale=[[0,GOLD],[1,NAVY]],
                          title="Exam Board Distribution")
            fig6.update_traces(textposition="outside")
            tnr_fig(fig6, 285); fig6.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig6, use_container_width=True)
            info_card("G.H.S.E.B dominant (812). Nearly all from Gujarat State Board.", color=NAVY)

        ca4, cb4 = st.columns(2)
        with ca4:
            exam = raw["Last Exam"].value_counts().reset_index()
            exam.columns = ["Exam", "Count"]
            fig7 = px.bar(exam, x="Exam", y="Count", text="Count", color="Exam",
                          color_discrete_map={"HSC":NAVY,"SSC":GOLD},
                          title="Last Exam Type")
            fig7.update_traces(textposition="outside")
            tnr_fig(fig7, 285); fig7.update_layout(showlegend=False)
            st.plotly_chart(fig7, use_container_width=True)
            info_card("833 students (98.9%) for HSC. Only 9 (1.1%) for SSC.", color=NAVY)

        with cb4:
            yr = raw["Last Exam Passing"].value_counts().reset_index()
            yr.columns = ["Year", "Count"]
            fig8 = px.bar(yr, x="Year", y="Count", text="Count", color="Count",
                          color_continuous_scale=[[0,GOLD],[1,NAVY]],
                          title="Last Exam Passing Year")
            fig8.update_traces(textposition="outside")
            tnr_fig(fig8, 285); fig8.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig8, use_container_width=True)
            info_card("779 (92.5%) passed 12th in 2023-24 -- straight from school.", color=AMBER)

    with u2:
        section_header("Distribution of Numerical Features")
        ca, cb = st.columns(2)
        with ca:
            fig_h = px.histogram(raw, x="Last Exam Percentage", nbins=25,
                                 color_discrete_sequence=[NAVY],
                                 title="Last Exam % -- Overall Distribution")
            fig_h.add_vline(x=raw["Last Exam Percentage"].mean(),
                            line_dash="dash", line_color=GOLD,
                            annotation_text=f"Mean {raw['Last Exam Percentage'].mean():.1f}%",
                            annotation_font_color=GOLD)
            tnr_fig(fig_h, 300)
            st.plotly_chart(fig_h, use_container_width=True)
            info_card(f"Mean: {raw['Last Exam Percentage'].mean():.1f}% . "
                      f"Median: {raw['Last Exam Percentage'].median():.1f}% . "
                      f"Std: {raw['Last Exam Percentage'].std():.1f}%.", color=NAVY)

        with cb:
            sem = raw["Current Semester"].value_counts().sort_index().reset_index()
            sem.columns = ["Semester", "Count"]
            fig_sem = px.bar(sem, x="Semester", y="Count", text="Count", color="Count",
                             color_continuous_scale=[[0,RED],[0.5,AMBER],[1,GREEN]],
                             title="Current Semester Distribution (raw data)")
            fig_sem.update_traces(textposition="outside")
            tnr_fig(fig_sem, 300); fig_sem.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_sem, use_container_width=True)
            info_card("NOTE: The model does NOT use current_semester. "
                      "This is an admission-time model -- predictions are made at enrolment.", color=RED)

    with u3:
        section_header("Geographic Distribution -- Permanent Districts")
        top_d = raw["Permanent District"].value_counts().head(15).reset_index()
        top_d.columns = ["District", "Count"]
        fig_geo = px.bar(top_d, x="Count", y="District", orientation="h",
                         text="Count", color="Count",
                         color_continuous_scale=[[0,GOLD],[1,NAVY]],
                         title="Top 15 Districts by Student Count")
        fig_geo.update_traces(textposition="outside")
        tnr_fig(fig_geo, 460); fig_geo.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_geo, use_container_width=True)
        info_card("Gandhinagar (360) + Ahmedabad (184) = 64.7% of students. "
                  "14 low-frequency districts are grouped into dist_other in the model.", color=NAVY)

    st.markdown("<br>", unsafe_allow_html=True)

    # -- SECTION B: BIVARIATE --------------------------------
    section_banner("SECTION B  --  BIVARIATE ANALYSIS (vs Churn)")

    b1, b2, b3 = st.tabs([
        "  Categorical vs Churn  ",
        "  Numerical vs Churn  ",
        "  Semester Deep Dive  ",
    ])

    with b1:
        section_header("Churn Rate by Categorical Feature",
                        "Each bar = % of that category who churned")
        cat_pairs = [
            ("Admission Cast Category", "Caste vs Churn Rate"),
            ("Gender",                  "Gender vs Churn Rate"),
            ("Institute",               "Institute vs Churn Rate"),
            ("Specialisation",          "Specialisation vs Churn Rate"),
            ("Religion",                "Religion vs Churn Rate"),
            ("Last Exam Board/Uni.",    "Exam Board vs Churn Rate"),
        ]
        ca, cb = st.columns(2)
        for i, (col, title) in enumerate(cat_pairs):
            grp = raw.groupby(col)["is_churned"].agg(["mean","sum","count"]).reset_index()
            grp.columns = [col, "Churn Rate", "Churned", "Total"]
            grp["Churn %"] = (grp["Churn Rate"] * 100).round(1)
            grp = grp.sort_values("Churn %", ascending=False)
            fig = px.bar(grp, x=col, y="Churn %",
                         color="Churn %",
                         color_continuous_scale=[[0,GREEN],[0.5,GOLD],[1,RED]],
                         text="Churn %", title=title,
                         hover_data={"Churned":True,"Total":True})
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            tnr_fig(fig, 285); fig.update_layout(coloraxis_showscale=False)
            with (ca if i % 2 == 0 else cb):
                st.plotly_chart(fig, use_container_width=True)

        section_header("Institute x Caste x Churn -- Sunburst")
        grp_sun = raw.groupby(
            ["Institute","Admission Cast Category","Churn Label"]
        ).size().reset_index(name="Count")
        fig_sun = px.sunburst(grp_sun,
                              path=["Institute","Admission Cast Category","Churn Label"],
                              values="Count",
                              color="Churn Label",
                              color_discrete_map={"Active":GREEN,"Churned":RED},
                              title="Institute > Caste > Churn Status")
        tnr_fig(fig_sun, 440)
        st.plotly_chart(fig_sun, use_container_width=True)
        info_card("Click any segment to zoom. Inner = Institute, middle = Caste, outer = Churn.", color=NAVY)

    with b2:
        section_header("Numerical Features vs Churn Status")
        ca, cb = st.columns(2)
        with ca:
            fig_bx = px.box(raw, x="Churn Label", y="Last Exam Percentage",
                            color="Churn Label",
                            color_discrete_map={"Active":GREEN,"Churned":RED},
                            title="Last Exam % -- Box Plot by Churn", points="outliers")
            fig_bx.update_layout(showlegend=False)
            tnr_fig(fig_bx, 300)
            st.plotly_chart(fig_bx, use_container_width=True)
            info_card("Churned students have noticeably lower median HSC%. "
                      "Below 50% = higher risk.", color=GREEN)

        with cb:
            fig_ov = px.histogram(raw, x="Last Exam Percentage", color="Churn Label",
                                  nbins=25, barmode="overlay", opacity=0.72,
                                  color_discrete_map={"Active":GREEN,"Churned":RED},
                                  title="HSC % Overlay -- Active vs Churned")
            tnr_fig(fig_ov, 300)
            st.plotly_chart(fig_ov, use_container_width=True)
            info_card("Red bars cluster in 45-65%. Students below 50% at highest risk.", color=RED)

        fig_sc = px.scatter(raw, x="Last Exam Percentage", y="Current Semester",
                            color="Churn Label",
                            color_discrete_map={"Active":GREEN,"Churned":RED},
                            title="HSC % vs Current Semester (colour = Churn)",
                            opacity=0.60,
                            hover_data=["Admission Cast Category","Specialisation"])
        tnr_fig(fig_sc, 360)
        st.plotly_chart(fig_sc, use_container_width=True)
        info_card("Red dots exclusively in Sem 1 -- data recording artefact. "
                  "This is why current_semester was excluded from the model.", color=NAVY)

    with b3:
        section_header("Semester-Wise Churn Deep Dive")
        st.warning("Sem 1 = 100% churn is a recording artefact -- all 107 churned students "
                   "are frozen at Sem 1. The model EXCLUDES current_semester to avoid this leakage.")

        ca, cb = st.columns(2)
        with ca:
            s_cnt = raw.groupby(["Current Semester","Churn Label"]).size().reset_index(name="Count")
            fig_s1 = px.bar(s_cnt, x="Current Semester", y="Count", color="Churn Label",
                            barmode="group",
                            color_discrete_map={"Active":GREEN,"Churned":RED},
                            title="Count per Semester (Active vs Churned)", text="Count")
            fig_s1.update_traces(textposition="outside")
            tnr_fig(fig_s1, 310)
            st.plotly_chart(fig_s1, use_container_width=True)

        with cb:
            s_rate = raw.groupby("Current Semester")["is_churned"].agg(
                ["mean","sum","count"]).reset_index()
            s_rate["Churn %"] = (s_rate["mean"] * 100).round(1)
            fig_s2 = px.bar(s_rate, x="Current Semester", y="Churn %",
                            color="Churn %", text="Churn %",
                            color_continuous_scale=[[0,GREEN],[0.5,GOLD],[1,RED]],
                            title="Churn Rate % per Semester")
            fig_s2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            tnr_fig(fig_s2, 310); fig_s2.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_s2, use_container_width=True)

        s_sum = raw.groupby("Current Semester").agg(
            Students=("is_churned","count"), Churned=("is_churned","sum"),
            Avg_HSC=("Last Exam Percentage","mean")).reset_index()
        s_sum["Churn Rate %"] = (s_sum["Churned"] / s_sum["Students"] * 100).round(1)
        s_sum["Avg HSC %"]    = s_sum["Avg_HSC"].round(1)
        st.dataframe(s_sum[["Current Semester","Students","Churned","Churn Rate %","Avg HSC %"]],
                     use_container_width=True, hide_index=True)


# ==========================================================
#  V. FEATURE ENGINEERING
# ==========================================================
elif "Feature Engineering" in page:
    st.title("Feature Engineering")
    st.markdown("<p style='font-size:14px;color:#5C6B7A;font-style:italic'>"
                "How raw columns were transformed into the 45 features the model uses.</p>",
                unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["  Transformations  ","  Feature Importance  ","  Correlation  "])

    with t1:
        section_header("All 45 Model Features", "SIMPLE_Fixed_binary.csv + engineered columns")
        eng = [
            ("gender",           "Label Encode",  "Male=0 . Female=1",           "Binary; no ordering needed."),
            ("fees_type",        "Value Map",     "Rs.18,000=0 . Rs.27,000=1",   "Captures institute + fee structure."),
            ("year_gap",         "Arithmetic",    "2024 minus last exam year",   "1=direct; higher=gap-year student."),
            ("College_enc",      "Label Encode",  "BPCCS=0 . SVICS-G=1",        "Explicit institute identifier."),
            ("exam_hsc/ssc",     "One-Hot",       "HSC=(1,0) . SSC=(0,1)",      "Mutually exclusive exam flags."),
            ("spec_arts/commerce/science","One-Hot","One flag per stream",       "Exactly one is 1 per student."),
            ("perf_bracket",     "Binning",       "0-44%=0 . 45-60%=1 . 60-75%=2 . 75%+=3","HSC % as 4 risk tiers."),
            ("board_gseb_group", "Binary Flag",   "G.H.S.E.B or GSEB=1 . else=0","Groups both GSEB variants."),
            ("board_cbse/ghseb/gseb/other","One-Hot","One flag per board",       "Individual board flags."),
            ("cast_obc/open/scst/sebc","One-Hot", "One flag per caste",          "One hot per caste category."),
            ("rel_christian/hindu/jain/muslim","One-Hot","One flag per religion", "One hot per religion."),
            ("dist_* (18 named)","One-Hot",       "One column per named district","18 high-frequency districts."),
            ("dist_other",       "Catchall Flag", "1 if from any of 14 low-freq districts","Groups rare districts."),
            ("perf_x_cast_obc",  "Interaction",   "perf_bracket x cast_obc",    "OBC students with low marks = highest risk."),
            ("perf_x_cast_scst", "Interaction",   "perf_bracket x cast_scst",   "SCST + academic performance interaction."),
            ("perf_x_cast_open", "Interaction",   "perf_bracket x cast_open",   "OPEN caste + performance interaction."),
        ]
        for feat, method, formula, reason in eng:
            color = GOLD if "One-Hot" in method or "Interaction" in method else NAVY
            st.markdown(f"""
            <div style='background:#FFF;border:1px solid #D4C5A9;
                 border-left:5px solid {color};padding:12px 16px;margin-bottom:7px'>
              <div style='display:flex;align-items:center;
                   justify-content:space-between;margin-bottom:5px'>
                <code style='font-size:13px;font-weight:700;color:{NAVY}'>{feat}</code>
                <span style='background:{color};color:#FAFAF7;font-size:10px;
                      font-weight:700;padding:2px 8px;text-transform:uppercase'>{method}</span>
              </div>
              <div style='background:#F5F2EB;padding:5px 10px;font-size:12px;
                   color:{NAVY};margin-bottom:4px'>{formula}</div>
              <div style='font-size:13px;color:#5C6B7A'>{reason}</div>
            </div>""", unsafe_allow_html=True)

        st.divider()
        info_card(
            f"<b>EXCLUDED FROM MODEL:</b> "
            f"<code>current_semester</code> (data leakage -- Sem 1 = 100% churned) &nbsp;|&nbsp; "
            f"<code>Roll No</code> (ID only) &nbsp;|&nbsp; "
            f"<code>Total Fees</code> (1:1 with Institute)",
            color=RED)

        binary_eng = load_binary().pipe(apply_feature_engineering)
        st.markdown("#### Engineered Dataset Preview (first 5 rows, selected columns)")
        show_cols = ["perf_bracket","cast_obc","perf_x_cast_obc",
                     "cast_scst","perf_x_cast_scst","dist_gandhinagar","dist_other","is_churned"]
        st.dataframe(binary_eng[show_cols].head(5), use_container_width=True)
        st.caption(f"Full dataset: {binary_eng.shape[0]} rows . Model uses {len(feats)} of these columns")

    with t2:
        section_header(f"Feature Importance -- {type(model).__name__}")
        if hasattr(model, 'feature_importances_'):
            fi = pd.DataFrame({"Feature": feats,
                               "Importance": model.feature_importances_}
                              ).sort_values("Importance", ascending=True)
            fig_fi = px.bar(fi, x="Importance", y="Feature", orientation="h",
                            color="Importance",
                            color_continuous_scale=[[0,GOLD],[1,NAVY]],
                            title=f"Feature Importance ({len(feats)} features)")
            tnr_fig(fig_fi, 650)
            fig_fi.update_layout(coloraxis_showscale=False,
                                 yaxis=dict(tickfont=dict(size=10)))
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("This model type does not expose feature_importances_ directly.")

    with t3:
        section_header("Pearson Correlation Matrix")
        binary_eng = load_binary().pipe(apply_feature_engineering)
        sel = ["is_churned","perf_bracket","year_gap",
               "gender","fees_type","College_enc",
               "exam_hsc","spec_commerce","spec_arts",
               "cast_obc","cast_open","cast_scst",
               "perf_x_cast_obc","perf_x_cast_scst",
               "rel_hindu","rel_muslim",
               "dist_gandhinagar","dist_ahmedabad","dist_other"]
        ex = [f for f in sel if f in binary_eng.columns]
        corr = binary_eng[ex].corr().round(2)
        fig_c = px.imshow(corr, text_auto=True,
                          color_continuous_scale="RdBu_r",
                          zmin=-1, zmax=1, aspect="auto",
                          title="Pearson Correlation (key features)")
        fig_c.update_traces(textfont_size=10)
        tnr_fig(fig_c, 540)
        st.plotly_chart(fig_c, use_container_width=True)
        info_card("perf_x_cast_obc shows higher correlation with churn than raw "
                  "cast_obc alone -- confirming interaction features add signal.", color=NAVY)


# ==========================================================
#  VI. MODEL & PREDICTION
# ==========================================================
elif "Model" in page:
    st.title("Model & Prediction")
    st.markdown(f"<p style='font-size:14px;color:#5C6B7A;font-style:italic'>"
                f"{type(model).__name__} . 45 admission-time features . "
                f"NO current_semester . 70/30 split.</p>",
                unsafe_allow_html=True)

    t1, t2, t3 = st.tabs([
        "  Model Performance  ",
        "  Predict a Student  ",
        "  All Test Predictions  ",
    ])

    # -- TAB 1: PERFORMANCE ----------------------------------
    with t1:
        acc = accuracy_score(y_test, all_preds)
        f1  = f1_score(y_test, all_preds, zero_division=0)
        auc = roc_auc_score(y_test, all_probs)
        cm  = confusion_matrix(y_test, all_preds)
        tn, fp, fn, tp = cm.ravel()

        section_header("Evaluation Metrics -- 30% Test Set")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("ROC-AUC",   f"{auc:.3f}")
        m2.metric("Recall",    f"{tp/(tp+fn):.1%}")
        m3.metric("F1 Score",  f"{f1:.3f}")
        m4.metric("Threshold", f"{thresh:.2f}")
        m5.metric("Accuracy",  f"{acc*100:.1f}%")

        ca, cb = st.columns(2)
        with ca:
            section_header("Confusion Matrix")
            fig_cm = go.Figure(go.Heatmap(
                z=cm,
                x=["Pred: Active", "Pred: Churned"],
                y=["Actual: Active", "Actual: Churned"],
                text=cm, texttemplate="<b>%{text}</b>",
                textfont={"size": 24},
                colorscale=[[0,"#F0ECE4"],[1,NAVY]],
            ))
            tnr_fig(fig_cm, 280)
            st.plotly_chart(fig_cm, use_container_width=True)

            st.markdown(f"""
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:7px'>
              <div style='background:rgba(45,106,79,0.11);border-left:4px solid {GREEN};
                   padding:10px;text-align:center'>
                <div style='font-size:24px;font-weight:700;color:{GREEN}'>{tp}</div>
                <div style='font-size:10px;color:{GREEN};text-transform:uppercase;
                     letter-spacing:0.5px'>TP -- Caught</div></div>
              <div style='background:rgba(139,37,37,0.1);border-left:4px solid {RED};
                   padding:10px;text-align:center'>
                <div style='font-size:24px;font-weight:700;color:{RED}'>{fn}</div>
                <div style='font-size:10px;color:{RED};text-transform:uppercase;
                     letter-spacing:0.5px'>FN -- Missed</div></div>
              <div style='background:rgba(155,106,26,0.1);border-left:4px solid {AMBER};
                   padding:10px;text-align:center'>
                <div style='font-size:24px;font-weight:700;color:{AMBER}'>{fp}</div>
                <div style='font-size:10px;color:{AMBER};text-transform:uppercase;
                     letter-spacing:0.5px'>FP -- False Alarm</div></div>
              <div style='background:rgba(28,43,58,0.09);border-left:4px solid {NAVY};
                   padding:10px;text-align:center'>
                <div style='font-size:24px;font-weight:700;color:{NAVY}'>{tn}</div>
                <div style='font-size:10px;color:{NAVY};text-transform:uppercase;
                     letter-spacing:0.5px'>TN -- Correct</div></div>
            </div>""", unsafe_allow_html=True)

        with cb:
            section_header("Probability Distribution")
            df_p = pd.DataFrame({
                "Probability": all_probs,
                "Actual": ["Churned" if v==1 else "Active" for v in y_test.values],
            })
            fig_pd = px.histogram(df_p, x="Probability", color="Actual",
                                  nbins=30, barmode="overlay", opacity=0.75,
                                  color_discrete_map={"Active":GREEN,"Churned":RED})
            fig_pd.add_vline(x=thresh, line_dash="dash", line_color=GOLD,
                             annotation_text=f"Threshold {thresh:.2f}",
                             annotation_font_color=GOLD)
            tnr_fig(fig_pd, 280)
            st.plotly_chart(fig_pd, use_container_width=True)

            section_header("Classification Report")
            rep    = classification_report(y_test, all_preds, output_dict=True, zero_division=0)
            rep_df = pd.DataFrame(rep).T.reset_index()
            rep_df.columns = ["Class","Precision","Recall","F1","Support"]
            rep_df = rep_df[rep_df["Class"].isin(["0","1","macro avg","weighted avg"])]
            rep_df["Class"] = rep_df["Class"].map(
                {"0":"Active","1":"Churned","macro avg":"Macro Avg","weighted avg":"Weighted Avg"})
            st.dataframe(rep_df.round(3), use_container_width=True, hide_index=True)

        section_header("ROC Curve")
        fpr_r, tpr_r, _ = roc_curve(y_test, all_probs)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr_r, y=tpr_r,
            name=f"{type(model).__name__} (AUC = {auc:.3f})",
            line=dict(color=NAVY, width=2.5),
            fill="tozeroy",
            fillcolor=rgba(NAVY, 0.09)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0,1], y=[0,1],
            name="Random baseline (0.500)",
            line=dict(color=GOLD, dash="dash", width=1.5)
        ))
        fig_roc.update_layout(title="ROC Curve -- 70/30 Stratified Split",
                              xaxis_title="False Positive Rate",
                              yaxis_title="True Positive Rate")
        tnr_fig(fig_roc, 390)
        st.plotly_chart(fig_roc, use_container_width=True)

    # -- TAB 2: PREDICT --------------------------------------
    with t2:
        section_header("Predict Churn Risk for a New Student",
                        "45 admission-time features -- fill every field for accurate prediction")

        with st.form("predict_form"):
            st.markdown(f"""
            <div style='background:{NAVY};color:{GOLD};font-size:12px;font-weight:700;
                 letter-spacing:1.2px;padding:9px 14px;text-transform:uppercase;
                 margin-bottom:16px'>
              COMPLETE STUDENT ADMISSION PROFILE
            </div>""", unsafe_allow_html=True)

            # -- GROUP 1: Institute --------------------------
            form_label("1 -- Institute & Admission")
            g1a, g1b = st.columns(2)
            college  = g1a.selectbox("College",
                                     ["BPCCS  (Rs.18,000)", "SVICS-G  (Rs.27,000)"],
                                     help="BPCCS = fees_type 0  |  SVICS-G = fees_type 1")
            year_gap = g1b.selectbox("Years Since 12th Passed", [1, 2, 3, 4, 5],
                                     help="1 = passed 2023-24 (direct entry)  |  2+ = gap year")

            # -- GROUP 2: Demographics -----------------------
            form_label("2 -- Demographics")
            g2a, g2b, g2c, g2d = st.columns(4)
            gender   = g2a.selectbox("Gender",   ["Male", "Female"])
            caste    = g2b.selectbox("Caste",    ["OPEN", "OBC", "SEBC", "SCST"])
            religion = g2c.selectbox("Religion", ["Hindu", "Muslim", "Jain", "Christian"])
            district = g2d.selectbox("Permanent District", ALL_DISTRICTS,
                                     help="Districts in CAPS with * are grouped into dist_other")

            # -- GROUP 3: Academic ---------------------------
            form_label("3 -- Academic Background")
            g3a, g3b, g3c, g3d = st.columns(4)
            perf      = g3a.selectbox("HSC Performance",
                                      ["0 -- Below 45%", "1 -- 45% to 60%",
                                       "2 -- 60% to 75%", "3 -- Above 75%"],
                                      help="perf_bracket: 0 = highest risk tier")
            spec      = g3b.selectbox("12th Stream", ["COMMERCE", "SCIENCE", "ARTS"])
            exam_type = g3c.selectbox("Exam Type",   ["HSC", "SSC"])
            board     = g3d.selectbox("Exam Board",
                                      ["G.H.S.E.B", "GSEB", "CBSE", "Other"])

            st.markdown("<br>", unsafe_allow_html=True)

            # Show district category note
            if district in OTHER_DISTRICTS:
                st.markdown(f"""
                <div style='background:rgba(200,169,110,0.12);border:1px solid {GOLD};
                     padding:8px 14px;font-size:12px;color:{NAVY};margin-bottom:8px'>
                  <b>Note:</b> {district} is grouped into <code>dist_other = 1</code>
                  (low-frequency district category in the model).
                </div>""", unsafe_allow_html=True)

            submitted = st.form_submit_button("PREDICT CHURN RISK")

        # -- RESULT -----------------------------------------
        if submitted:
            perf_v = int(perf[0])

            row    = build_feature_row(gender, college, year_gap, perf_v,
                                       spec, exam_type, board, caste, religion, district)
            inp_df = pd.DataFrame([row])[feats]
            inp_sc = scaler.transform(inp_df)
            prob   = model.predict_proba(inp_sc)[0][1]
            pred   = int(prob >= thresh)
            rlabel, rcolor = get_risk(prob)
            badge_t = "FLAGGED AS CHURN RISK" if pred else "FLAGGED AS ACTIVE"
            badge_c = RED if pred else GREEN

            # -- Big result box -----------------------------
            st.markdown(f"""
            <div style='background:#FFFFFF;border:3px solid {rcolor};
                 padding:26px;text-align:center;margin:12px 0'>
              <div style='font-size:56px;font-weight:700;color:{rcolor};
                   letter-spacing:-2px;line-height:1'>{prob*100:.1f}%</div>
              <div style='font-size:15px;color:#5C6B7A;margin:7px 0'>Churn Probability</div>
              <div style='font-size:18px;font-weight:700;color:{rcolor}'>{rlabel}</div>
              <div style='display:inline-block;margin-top:11px;background:{badge_c};
                   color:#FAFAF7;font-size:12px;font-weight:700;letter-spacing:1px;
                   padding:5px 18px;text-transform:uppercase'>{badge_t}</div>
            </div>""", unsafe_allow_html=True)

            # -- Gauge -------------------------------------
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=prob * 100,
                number={"suffix":"%","font":{"color":rcolor,"size":42,"family":"Times New Roman"}},
                gauge={
                    "axis": {"range":[0,100],"tickcolor":GOLD},
                    "bar":  {"color":rcolor,"thickness":0.22},
                    "bgcolor": "white",
                    "steps": [
                        {"range":[0,20],  "color":"#E8F5E9"},
                        {"range":[20,40], "color":"#FFF8E1"},
                        {"range":[40,65], "color":"#FFEBEE"},
                        {"range":[65,100],"color":"#FFDDE1"},
                    ],
                    "threshold":{"line":{"color":GOLD,"width":3},
                                 "thickness":0.75,"value":thresh*100},
                },
            ))
            fig_g.update_layout(
                height=260, margin=dict(t=8,b=8,l=28,r=28),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Times New Roman, Times, serif", color=NAVY),
            )
            st.plotly_chart(fig_g, use_container_width=True)
            st.caption(f"Zones: 0-20% Low  |  20-40% Moderate  |  "
                       f"40-65% High  |  65-100% Very High  |  Threshold: {thresh:.2f}")

            # -- Feature values used -----------------------
            with st.expander("View exact feature values sent to model"):
                disp = pd.DataFrame([row]).T.reset_index()
                disp.columns = ["Feature", "Value"]
                disp["In Model"] = disp["Feature"].apply(lambda x: "Yes" if x in feats else "No")
                st.dataframe(disp[disp["In Model"]=="Yes"].drop(columns="In Model"),
                             use_container_width=True, hide_index=True)

            # -- Interaction feature values -----------------
            obc_enc  = 1 if caste == "OBC"  else 0
            scst_enc = 1 if caste == "SCST" else 0
            open_enc = 1 if caste == "OPEN" else 0
            st.markdown("#### Interaction Feature Values")
            icols = st.columns(3)
            icols[0].metric("perf_x_cast_obc",  perf_v * obc_enc,
                            help="perf_bracket x cast_obc")
            icols[1].metric("perf_x_cast_scst", perf_v * scst_enc,
                            help="perf_bracket x cast_scst")
            icols[2].metric("perf_x_cast_open", perf_v * open_enc,
                            help="perf_bracket x cast_open")

            # -- Risk factor cards -------------------------
            st.markdown("#### Risk Factor Breakdown")
            factors = [
                ("Performance",  perf.split("--")[1].strip(),
                 RED if perf_v==0 else AMBER if perf_v==1 else GREEN),
                ("Caste",        caste,
                 AMBER if caste in ["OBC","SCST"] else GREEN),
                ("Stream",       spec,
                 AMBER if spec=="ARTS" else GREEN),
                ("Year Gap",     f"{year_gap} yr",
                 AMBER if year_gap >= 3 else GREEN),
                ("College",      college.split("(")[0].strip(),
                 AMBER if "SVICS" in college else GREEN),
                ("District",     district,
                 GREEN if district in ["GANDHINAGAR","AHMEDABAD"]
                 else RED if district in OTHER_DISTRICTS else AMBER),
                ("Exam Board",   board,
                 GREEN if board in ["G.H.S.E.B","GSEB"] else AMBER),
                ("Religion",     religion, NAVY),
            ]
            fcols = st.columns(4)
            for i, (fname, fval, fclr) in enumerate(factors):
                fcols[i%4].markdown(f"""
                <div style='background:#FFF;border:1px solid #D4C5A9;
                     border-top:4px solid {fclr};padding:10px;
                     margin-bottom:7px;text-align:center'>
                  <div style='font-size:9px;color:#8FA3B8;font-weight:700;
                       text-transform:uppercase;letter-spacing:0.5px'>{fname}</div>
                  <div style='font-size:13px;font-weight:700;color:{fclr};
                       margin-top:4px'>{fval}</div>
                </div>""", unsafe_allow_html=True)

            # -- Recommendations ---------------------------
            recs = []
            if perf_v == 0:
                recs.append("HSC below 45% -- academic bridging programme required immediately.")
            elif perf_v == 1:
                recs.append("HSC 45-60% -- monitor first internal assessment closely.")
            if caste == "SCST":
                recs.append("Verify scholarship application and confirm disbursement date.")
            if caste == "OBC":
                recs.append("OBC category has the highest observed churn rate (18.1%) -- proactive welfare check advised.")
            if spec == "ARTS":
                recs.append("ARTS stream for BCA -- confirm career clarity and student motivation.")
            if year_gap >= 3:
                recs.append("3+ year gap since 12th -- check for unresolved personal or financial issues.")
            if district in OTHER_DISTRICTS:
                recs.append(f"{district} is a low-frequency (dist_other) district -- verify travel distance and accommodation.")
            elif district not in ["GANDHINAGAR","AHMEDABAD"]:
                recs.append("Out-of-area student -- assign peer mentor; verify commute or hostel situation.")
            if pred and not recs:
                recs.append(f"Model gives {prob*100:.1f}% from combined profile -- schedule a general welfare check-in.")

            if recs:
                st.markdown("#### Recommended Interventions")
                for r in recs:
                    st.markdown(f"""
                    <div style='background:#FFF;border:1px solid #D4C5A9;
                         border-left:5px solid {GOLD};padding:10px 14px;
                         margin-bottom:7px;font-size:13px;color:{NAVY}'>
                      &nbsp; {r}
                    </div>""", unsafe_allow_html=True)
            elif not pred:
                st.success("No major risk factors detected. Student profile appears stable.")

    # -- TAB 3: ALL PREDICTIONS ------------------------------
    with t3:
        section_header("All Test Student Predictions")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Total Test",     X_test.shape[0])
        mc2.metric("Pred Churned",   int(all_preds.sum()))
        mc3.metric("Actual Churned", int(y_test.sum()))
        mc4.metric("Correct",        int((all_preds == y_test.values).sum()))
        st.divider()

        res = pd.DataFrame({
            "Actual":    ["Churned" if v==1 else "Active" for v in y_test.values],
            "Prob %":    (all_probs * 100).round(1),
            "Predicted": ["Churned" if v==1 else "Active" for v in all_preds],
            "Risk":      [get_risk(p)[0] for p in all_probs],
            "Correct":   ["YES" if p==a else "NO"
                          for p,a in zip(all_preds, y_test.values)],
        }).reset_index(drop=True)

        fc1, fc2, fc3 = st.columns(3)
        fa  = fc1.selectbox("Filter Actual",    ["All","Churned","Active"])
        fp2 = fc2.selectbox("Filter Predicted", ["All","Churned","Active"])
        fcr = fc3.selectbox("Filter Correct",   ["All","YES","NO"])

        flt = res.copy()
        if fa  != "All": flt = flt[flt["Actual"]    == fa]
        if fp2 != "All": flt = flt[flt["Predicted"] == fp2]
        if fcr != "All": flt = flt[flt["Correct"]   == fcr]

        st.markdown(f"**Showing {len(flt)} of {len(res)} students**")
        st.dataframe(flt, use_container_width=True, height=360)

        fig_h = px.histogram(res, x="Prob %", color="Actual",
                             nbins=30, barmode="overlay", opacity=0.75,
                             color_discrete_map={"Active":GREEN,"Churned":RED},
                             title="Churn Probability Distribution -- All Test Students")
        fig_h.add_vline(x=thresh*100, line_dash="dash", line_color=GOLD,
                        annotation_text=f"Threshold {thresh:.2f}",
                        annotation_font_color=GOLD)
        tnr_fig(fig_h, 340)
        st.plotly_chart(fig_h, use_container_width=True)
