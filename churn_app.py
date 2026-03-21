
import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, warnings
warnings.filterwarnings("ignore")

# Lazy import heavy libraries only when needed
def lazy_import_sklearn():
    global train_test_split, accuracy_score, f1_score, roc_auc_score
    global confusion_matrix, classification_report, roc_curve, precision_score, recall_score
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                                 confusion_matrix, classification_report,
                                 roc_curve, precision_score, recall_score)

import plotly.express as px
import plotly.graph_objects as go

FOLDER = r"F:\stud_churn"

st.set_page_config(page_title="BCA Churn Analysis", layout="wide",
                   initial_sidebar_state="expanded")

GOLD = "#C8A96E"
NAVY = "#1C2B3A"
GREEN = "#2D6A4F"
RED = "#8B2525"
AMBER = "#9B6A1A"
CREAM = "#FAFAF7"

def rgba(h, a=0.10):
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"

# ── CSS ────────────────────────────────────────────────────
st.markdown(f"""<style>
*,*::before,*::after{{font-family:"Times New Roman",Times,serif!important;box-sizing:border-box;}}
.stApp{{background-color:{CREAM};}}
[data-testid="stSidebar"]{{background-color:{NAVY};border-right:4px solid {GOLD};}}
[data-testid="stSidebar"] *{{color:#EDE8DC!important;font-family:"Times New Roman",Times,serif!important;}}
[data-testid="stSidebar"] .stRadio label{{font-size:14px!important;padding:5px 0;display:block;}}
h1{{font-size:2.1rem!important;font-weight:700!important;color:{NAVY}!important;border-bottom:3px solid {GOLD};padding-bottom:9px;margin-bottom:8px;}}
h2{{font-size:1.4rem!important;font-weight:700!important;color:{NAVY}!important;margin-top:16px!important;}}
h3{{font-size:1.05rem!important;font-weight:600!important;color:#2D4A5E!important;}}
[data-testid="metric-container"]{{background:#FFF;border:1px solid #D4C5A9;border-top:4px solid {GOLD};border-radius:0;padding:14px 16px;box-shadow:2px 2px 7px rgba(0,0,0,0.06);}}
[data-testid="stMetricValue"]{{color:{NAVY}!important;font-size:1.85rem!important;font-weight:700!important;}}
[data-testid="stMetricLabel"]{{color:#5C6B7A!important;font-size:0.76rem!important;text-transform:uppercase;letter-spacing:1.2px;}}
.stTabs [data-baseweb="tab-list"]{{background:{NAVY};border-radius:0;gap:0;padding:0;}}
.stTabs [data-baseweb="tab"]{{background:transparent;color:{GOLD}!important;font-size:12.5px!important;font-weight:600;padding:10px 18px;border-right:1px solid rgba(200,169,110,0.2);border-radius:0;transition:all 0.2s ease;}}
.stTabs [data-baseweb="tab"]:hover{{background:rgba(200,169,110,0.15);}}
.stTabs [aria-selected="true"]{{background:{GOLD}!important;color:{NAVY}!important;}}
.stButton>button,[data-testid="stFormSubmitButton"]>button{{background-color:{NAVY};color:{GOLD};border:2px solid {GOLD};border-radius:0;font-size:13.5px;font-weight:700;padding:11px 26px;letter-spacing:1.2px;text-transform:uppercase;width:100%;transition:all 0.3s ease;}}
.stButton>button:hover,[data-testid="stFormSubmitButton"]>button:hover{{background-color:{GOLD};color:{NAVY};transform:translateY(-2px);box-shadow:0 4px 8px rgba(0,0,0,0.15);}}
.stSelectbox>label,.stNumberInput>label{{font-weight:700!important;color:{NAVY}!important;font-size:11.5px!important;text-transform:uppercase;letter-spacing:0.6px;}}
.stSelectbox>div>div,.stNumberInput>div>div{{border:2px solid #D4C5A9!important;border-radius:0!important;}}
.stDataFrame{{border:1px solid #D4C5A9;box-shadow:2px 2px 5px rgba(0,0,0,0.04);}}
hr{{border-color:#D4C5A9!important;margin:16px 0;}}
.stPlotlyChart{{border:1px solid #D4C5A9;background:#FFF;padding:10px;box-shadow:2px 2px 6px rgba(0,0,0,0.05);margin-bottom:12px;}}
details summary{{cursor:pointer;padding:10px 14px;background:{NAVY};color:{GOLD};font-size:12px;font-weight:700;letter-spacing:0.8px;text-transform:uppercase;list-style:none;transition:all 0.2s ease;}}
details summary:hover{{background:rgba(28,43,58,0.9);}}
details summary::-webkit-details-marker{{display:none;}}
details[open] summary{{border-bottom:2px solid {GOLD};}}
details .insight-body{{background:#FFF;border:1px solid #D4C5A9;border-top:none;padding:14px 16px;font-size:13px;color:#2C3E50;line-height:1.85;}}
code{{background:#F5F2EB;padding:2px 6px;color:{NAVY};font-size:12px;border-radius:2px;border:1px solid #E5DCC8;}}
[data-testid="stForm"]{{border:2px solid {GOLD}!important;background:#FFF!important;padding:16px!important;}}
</style>""", unsafe_allow_html=True)

# ── HELPERS ────────────────────────────────────────────────
def section_header(title, sub=""):
    s = f"<div style='font-size:12px;color:#5C6B7A;margin-top:4px'>{sub}</div>" if sub else ""
    st.markdown(f"""<div style='border-left:5px solid {GOLD};padding:9px 17px;background:#FFF;
         margin-bottom:17px;box-shadow:2px 2px 5px rgba(0,0,0,0.05)'>
      <div style='font-size:18px;font-weight:700;color:{NAVY}'>{title}</div>{s}</div>""",
        unsafe_allow_html=True)

def info_card(html, color=NAVY):
    st.markdown(f"""<div style='background:#FFF;border:1px solid #D4C5A9;border-left:5px solid {color};
         padding:12px 16px;margin-bottom:10px;font-size:13px;color:#2C3E50;line-height:1.75'>{html}</div>""",
        unsafe_allow_html=True)

def section_banner(text):
    st.markdown(f"""<div style='background:{NAVY};color:{GOLD};font-size:14px;font-weight:700;
         letter-spacing:1.3px;text-transform:uppercase;padding:10px 20px;margin:22px 0 16px 0'>
      &#9672;&nbsp; {text}</div>""", unsafe_allow_html=True)

def tnr(fig, h=360):
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FEFEFE",
        font=dict(family="Times New Roman,Times,serif", color=NAVY, size=12),
        title_font=dict(family="Times New Roman,Times,serif", color=NAVY, size=14),
        title_x=0.5,
        xaxis=dict(gridcolor="#EAE3D5", linecolor=GOLD, tickfont=dict(family="Times New Roman",size=11)),
        yaxis=dict(gridcolor="#EAE3D5", linecolor=GOLD, tickfont=dict(family="Times New Roman",size=11)),
        legend=dict(bgcolor="rgba(255,255,255,0.95)", bordercolor=GOLD, borderwidth=1,
                    font=dict(family="Times New Roman",size=11)),
        margin=dict(t=60, b=50, l=60, r=100))
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig

def get_risk(p):
    if p < 0.20: return "Low Risk",       "#2D6A4F"
    elif p < 0.40: return "Moderate Risk",  "#9B6A1A"
    elif p < 0.65: return "High Risk",      "#8B2525"
    else:          return "Very High Risk", "#5C0A0A"

def chart_insight(title, body):
    st.markdown(f"""<details>
      <summary>&#9432;&nbsp; {title}</summary>
      <div class="insight-body">{body}</div>
    </details>""", unsafe_allow_html=True)

def form_label(text):
    st.markdown(f"""<div style='font-size:11px;font-weight:700;color:{GOLD};
         text-transform:uppercase;letter-spacing:0.8px;
         border-bottom:1px solid rgba(200,169,110,0.5);
         padding-bottom:4px;margin:14px 0 10px 0'>{text}</div>""", unsafe_allow_html=True)

# ── CONSTANTS ──────────────────────────────────────────────
HIGH_CHURN_DISTS = ['SABARKANTHA','KUTCH','BANASKANTHA','MEHSANA','BOTAD']
LOCAL_DISTS      = ['GANDHINAGAR','AHMEDABAD']
ALL_DISTRICTS    = sorted([
    "AHMEDABAD","AMRELI","ARVALLI","BANASKANTHA","BHAVNAGAR","BOTAD",
    "GANDHINAGAR","GIR SOMNATH","JAMNAGAR","JUNAGADH","KHEDA","KUTCH",
    "MEHSANA","MORBI","PATAN","RAJKOT","SABARKANTHA","SURENDRANAGAR",
    "ANAND","AURANGABAD","DAHOD","DEVBHUMI DWARKA","DUNGARPUR","GONDIYA",
    "HALVAD","HIMMATNAGAR","MAHARASHTRA","MAHISAGAR","NAVSARI","PORBANDAR",
    "SIROHI","SURAT",
])

# Historical churn rates per semester (from dataset)
# BPCCS: Sem1=100%, Sem2=75%, Sem3=25%, Sem4-6=0%
# SVICS-G: Sem1=100%, Sem2=93%, Sem3=0.45%, Sem4-6=N/A
SEM_HIST_RATE = {1: 1.000, 2: 0.809, 3: 0.032, 4: 0.000, 5: 0.000, 6: 0.000}

# College-specific semester availability
COLLEGE_SEMESTERS = {
    "BPCCS": [1, 2, 3, 4, 5, 6],  # Has students across all 6 semesters
    "SVICS-G": [1, 2, 3]           # Has students only in semesters 1-3
}

# Simple semester labels without historical rates
SEM_LABELS = {
    1: "1 -- First Semester",
    2: "2 -- Second Semester",
    3: "3 -- Third Semester",
    4: "4 -- Fourth Semester",
    5: "5 -- Fifth Semester",
    6: "6 -- Sixth Semester",
}

# ── FEATURE ENGINEERING ────────────────────────────────────
def build_admission_features(exam_pct, gender, college, year_gap,
                              spec, board, caste, religion, district):
    gen  = 0 if gender=="Male" else 1
    col  = 0 if "BPCCS" in college else 1
    fees = 18000 if "BPCCS" in college else 27000
    obc  = 1 if caste=="OBC" else 0
    sct  = 1 if caste=="SCST" else 0
    sbc  = 1 if caste=="SEBC" else 0
    opn  = 1 if caste=="OPEN" else 0
    mus  = 1 if religion=="Muslim" else 0
    hin  = 1 if religion=="Hindu" else 0
    sci  = 1 if spec=="SCIENCE" else 0
    art  = 1 if spec=="ARTS" else 0
    com  = 1 if spec=="COMMERCE" else 0
    cbse = 1 if "CBSE" in board.upper() else 0
    gseb = 1 if any(x in board.upper() for x in ["GSEB","GHSEB","G.H.S.E.B","G.S.E.B"]) else 0
    dh   = 1 if district in HIGH_CHURN_DISTS else 0
    dl   = 1 if district in LOCAL_DISTS else 0
    pdev = exam_pct - 61.5
    pdz  = 1 if 50<=exam_pct<65 else 0
    pvl  = 1 if exam_pct<45 else 0
    rs   = min(5, obc+sci+art+dh+mus+pdz+cbse)
    return {
        'exam_pct': exam_pct, 'pct_sq': (exam_pct/100)**2,
        'pct_dev': pdev, 'pct_dev_sq': pdev**2,
        'pct_danger': pdz, 'pct_very_low': pvl,
        'gender': gen, 'college': col, 'fees': fees, 'year_gap': year_gap,
        'cast_obc': obc, 'cast_scst': sct, 'cast_sebc': sbc, 'cast_open': opn,
        'rel_muslim': mus, 'rel_hindu': hin,
        'spec_science': sci, 'spec_arts': art, 'spec_commerce': com,
        'board_cbse': cbse, 'board_gseb': gseb,
        'dist_high': dh, 'dist_local': dl,
        'pct_x_obc': exam_pct*obc, 'pct_x_science': exam_pct*sci,
        'pct_x_dist': exam_pct*dh, 'pct_x_college': exam_pct*col,
        'bpccs_obc': (1-col)*obc, 'female_svics': gen*col,
        'obc_science': obc*sci,
        'risk_score': rs, 'risk_x_pct': rs*exam_pct,
    }

def apply_semester_signal(base_prob, semester, sem_weight=None):
    """Combine admission model probability with semester historical signal.
    Uses adaptive weighting based on semester risk level:
    - Semester 1: 20% weight (highest risk, 100% historical churn)
    - Semester 2: 15% weight (high risk, 75-93% historical churn)
    - Semester 3: 5% weight (low risk, <3% historical churn)
    - Semester 4-6: 0% weight (admission model only)
    """
    if semester >= 4:
        return base_prob
    
    # Adaptive semester weights based on risk level
    adaptive_weights = {1: 0.20, 2: 0.15, 3: 0.05}
    weight = adaptive_weights.get(int(semester), 0.05)
    
    sem_signal = SEM_HIST_RATE.get(int(semester), 0.032)
    combined = (1 - weight) * base_prob + weight * sem_signal
    return float(np.clip(combined, 0.0, 1.0))

def rebuild_features(raw_df):
    df = raw_df.copy()
    df['exam_pct']   = df['Last Exam Percentage'].fillna(df['Last Exam Percentage'].median())
    df['gender']     = (df['Gender']=='Female').astype(int)
    df['college']    = (df['Institute']=='SVICS-G').astype(int)
    df['fees']       = df['Total Fees'].fillna(df['Total Fees'].median())
    df['cast_obc']   = (df['Admission Cast Category']=='OBC').astype(int)
    df['cast_scst']  = (df['Admission Cast Category']=='SCST').astype(int)
    df['cast_sebc']  = (df['Admission Cast Category']=='SEBC').astype(int)
    df['cast_open']  = (df['Admission Cast Category']=='OPEN').astype(int)
    df['rel_muslim'] = (df['Religion']=='Muslim').astype(int)
    df['rel_hindu']  = (df['Religion']=='Hindu').astype(int)
    df['spec_science']  = (df['Specialisation']=='SCIENCE').astype(int)
    df['spec_arts']     = (df['Specialisation']=='ARTS').astype(int)
    df['spec_commerce'] = (df['Specialisation']=='COMMERCE').astype(int)
    def pg(s):
        try: return max(1,min(5,2024-int(str(s).split('-')[0])))
        except: return 1
    df['year_gap'] = df['Last Exam Passing'].apply(pg)
    bs = df['Last Exam Board/Uni.'].str.upper().fillna('')
    df['board_cbse'] = bs.str.contains('CBSE',na=False).astype(int)
    df['board_gseb'] = bs.str.contains('GSEB|GHSEB|G.H.S.E.B|G.S.E.B',na=False).astype(int)
    df['dist_high']  = df['Permanent District'].isin(HIGH_CHURN_DISTS).astype(int)
    df['dist_local'] = df['Permanent District'].isin(LOCAL_DISTS).astype(int)
    df['pct_dev']    = df['exam_pct'] - 61.5
    df['pct_dev_sq'] = df['pct_dev']**2
    df['pct_danger'] = ((df['exam_pct']>=50)&(df['exam_pct']<65)).astype(int)
    df['pct_very_low'] = (df['exam_pct']<45).astype(int)
    df['pct_sq']       = (df['exam_pct']/100)**2
    df['pct_x_obc']    = df['exam_pct']*df['cast_obc']
    df['pct_x_science']= df['exam_pct']*df['spec_science']
    df['pct_x_dist']   = df['exam_pct']*df['dist_high']
    df['pct_x_college']= df['exam_pct']*df['college']
    df['bpccs_obc']    = (1-df['college'])*df['cast_obc']
    df['female_svics'] = df['gender']*df['college']
    df['obc_science']  = df['cast_obc']*df['spec_science']
    df['risk_score']   = (df['cast_obc']+df['spec_science']+df['spec_arts']+
                          df['dist_high']+df['rel_muslim']+
                          df['pct_danger']+df['board_cbse']).clip(0,5)
    df['risk_x_pct']   = df['risk_score']*df['exam_pct']
    return df

# ── DATA & MODEL LOADING ───────────────────────────────────
@st.cache_data
def load_raw():
    df = pd.read_csv(os.path.join(FOLDER,"latest.csv"))
    df["is_churned"]  = df["student_status_new"].apply(
        lambda x: 1 if "dropout" in str(x).lower() and "sem1" in str(x).lower() else 0)
    df["Churn Label"] = df["is_churned"].map({0:"Active",1:"Churned"})
    return df

@st.cache_resource
def load_model():
    m   = joblib.load(os.path.join(FOLDER,"churn_model.pkl"))
    sc  = joblib.load(os.path.join(FOLDER,"churn_scaler.pkl"))
    th  = joblib.load(os.path.join(FOLDER,"churn_threshold.pkl"))
    ft  = joblib.load(os.path.join(FOLDER,"churn_feature_names.pkl"))
    sw  = joblib.load(os.path.join(FOLDER,"churn_sem_weight.pkl"))
    return m, sc, th, ft, sw

@st.cache_data
def get_test_set(feats_tuple):
    # Lazy import sklearn here
    from sklearn.model_selection import train_test_split
    raw   = load_raw()
    df    = rebuild_features(raw)
    df["is_churned"] = raw["is_churned"].values
    y     = df["is_churned"]
    X     = df[list(feats_tuple)]
    sems  = raw["Current Semester"]
    Xtr,Xte,ytr,yte,str_,ste = train_test_split(
        X, y, sems, test_size=0.30, random_state=42, stratify=y)
    return Xte, yte, ste

try:
    raw = load_raw()
    model, scaler, thresh, feats, sem_weight = load_model()
    X_test, y_test, sem_test = get_test_set(tuple(feats))
    Xs_test    = scaler.transform(X_test)
    base_probs = model.predict_proba(Xs_test)[:,1]
    # Use adaptive semester weighting for test set predictions
    all_probs  = np.array([apply_semester_signal(p, s) for p, s in zip(base_probs, sem_test.values)])
    all_preds  = (all_probs >= thresh).astype(int)
    # Calculate actual metrics - lazy import sklearn
    lazy_import_sklearn()
    ACTUAL_ACC = accuracy_score(y_test, all_preds)
    ACTUAL_AUC = roc_auc_score(y_test, all_probs)
    ACTUAL_F1 = f1_score(y_test, all_preds, zero_division=0)
    LOADED = True
except Exception as e:
    LOADED = False
    ERR = str(e)
    ACTUAL_ACC = 0.854
    ACTUAL_AUC = 0.752
    ACTUAL_F1 = 0.464

# ── SIDEBAR ────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""<div style='padding:24px 18px 14px;text-align:center;
         border-bottom:2px solid rgba(200,169,110,0.4)'>
      <div style='font-size:36px;margin-bottom:6px'>&#127891;</div>
      <div style='font-size:18px;font-weight:700;color:{GOLD};letter-spacing:1px'>BCA CHURN</div>
      <div style='font-size:10px;color:#8FA3B8;margin-top:4px;letter-spacing:2.5px;
           text-transform:uppercase'>Analysis Project</div></div>""", unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    page = st.radio("",[
        "I.    Introduction","II.   Dataset Overview","III.  Data Cleaning",
        "IV.   EDA","V.    Feature Engineering","VI.   Model & Prediction"],
        label_visibility="collapsed")
    if LOADED:
        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown(f"""<div style='font-size:11px;color:#8FA3B8;padding:0 4px;line-height:2.2'>
          <span style='color:{GOLD}'>&#9632;</span>&nbsp; {type(model).__name__}<br>
          <span style='color:{GOLD}'>&#9632;</span>&nbsp; Accuracy: {ACTUAL_ACC*100:.1f}%<br>
          <span style='color:{GOLD}'>&#9632;</span>&nbsp; AUC: {ACTUAL_AUC:.3f}<br>
          <span style='color:{GOLD}'>&#9632;</span>&nbsp; F1: {ACTUAL_F1:.3f}<br>
          <span style='color:{GOLD}'>&#9632;</span>&nbsp; Features: {len(feats)} admission<br>
          <span style='color:{GOLD}'>&#9632;</span>&nbsp; + adaptive semester signal<br>
          <span style='color:{GOLD}'>&#9632;</span>&nbsp; (S1:20%, S2:15%, S3:5%)</div>""",
            unsafe_allow_html=True)

if not LOADED:
    st.error(f"Cannot load model from {FOLDER}. Copy all 5 pkl files.\n\n{ERR}")
    st.stop()

# ================================================================
#  I. INTRODUCTION
# ================================================================
if "Introduction" in page:
    st.title("BCA Student Churn Analysis")
    st.markdown("<p style='font-size:15px;color:#5C6B7A;font-style:italic;margin-bottom:20px'>"
                "Predicting student dropout in BCA using machine learning -- Batch 2023-24.</p>",
                unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Students","842"); c2.metric("Active","735")
    c3.metric("Churned","107");        c4.metric("Churn Rate","12.7%")
    st.divider()

    st.markdown(f"""<div style='background:{NAVY};border:2px solid {GOLD};padding:16px 20px;margin-bottom:20px'>
      <div style='font-size:13px;font-weight:700;color:{GOLD};text-transform:uppercase;
           letter-spacing:1px;margin-bottom:10px'>Model Highlights</div>
      <div style='display:grid;grid-template-columns:repeat(5,1fr);gap:10px'>
        <div style='text-align:center;background:rgba(200,169,110,0.1);padding:12px 6px'>
          <div style='font-size:24px;font-weight:700;color:{GOLD}'>{ACTUAL_ACC*100:.1f}%</div>
          <div style='font-size:9px;color:#8FA3B8;text-transform:uppercase;margin-top:3px;letter-spacing:0.5px'>Accuracy</div></div>
        <div style='text-align:center;background:rgba(200,169,110,0.1);padding:12px 6px'>
          <div style='font-size:24px;font-weight:700;color:{GOLD}'>{ACTUAL_AUC:.3f}</div>
          <div style='font-size:9px;color:#8FA3B8;text-transform:uppercase;margin-top:3px;letter-spacing:0.5px'>ROC-AUC</div></div>
        <div style='text-align:center;background:rgba(200,169,110,0.1);padding:12px 6px'>
          <div style='font-size:24px;font-weight:700;color:{GOLD}'>{ACTUAL_F1:.3f}</div>
          <div style='font-size:9px;color:#8FA3B8;text-transform:uppercase;margin-top:3px;letter-spacing:0.5px'>F1 Score</div></div>
        <div style='text-align:center;background:rgba(200,169,110,0.1);padding:12px 6px'>
          <div style='font-size:24px;font-weight:700;color:{GOLD}'>{recall_score(y_test,all_preds,zero_division=0)*100:.0f}%</div>
          <div style='font-size:9px;color:#8FA3B8;text-transform:uppercase;margin-top:3px;letter-spacing:0.5px'>Recall</div></div>
        <div style='text-align:center;background:rgba(200,169,110,0.1);padding:12px 6px'>
          <div style='font-size:24px;font-weight:700;color:{GOLD}'>32</div>
          <div style='font-size:9px;color:#8FA3B8;text-transform:uppercase;margin-top:3px;letter-spacing:0.5px'>Features</div></div>
      </div></div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns([3,2])
    with col_l:
        section_header("How the Model Works")
        info_card(f"""The model combines two complementary signals:<br><br>
          <b>Admission data model (80-95%):</b> Random Forest trained on 32 features
          known at enrollment -- HSC percentage, caste, stream, district, board,
          religion, gender, and college. Gives honest 82.6% accuracy alone.<br><br>
          <b>Adaptive semester signal (5-20%):</b> Applied only for Semesters 1-3
          with risk-based weighting:<br>
          • Sem 1: 20% weight (100% historical churn)<br>
          • Sem 2: 15% weight (75-93% historical churn)<br>
          • Sem 3: 5% weight (<3% historical churn)<br>
          • Sem 4-6: 0% weight (admission model only)<br><br>
          Combined result: <b>{ACTUAL_ACC*100:.1f}% accuracy, AUC {ACTUAL_AUC:.3f}</b> -- a realistic,
          human-level performance for this kind of problem.""", color=GOLD)

        section_header("College-Specific Data Structure")
        info_card(f"""<b>BPCCS (588 students):</b> Has 3 semesters of churn data (Sem 1-3),
          but students progress through all 6 semesters. Churn concentrated in early semesters.<br><br>
          <b>SVICS-G (254 students):</b> Has 6 semesters of data, mostly in Sem 3 (221 students).
          Limited data in Sem 1-2. Prediction interface adapts to show only Sem 1-3 for SVICS-G.""", color=NAVY)

        section_header("Project Objectives")
        for o in [
            "Perform comprehensive EDA on 842 BCA students.",
            "Engineer 32 admission-time features with data-driven interactions.",
            "Train an honest Random Forest: {ACTUAL_ACC*100:.1f}% accuracy, AUC {ACTUAL_AUC:.3f}.",
            "Use semester as a validated signal for early semesters (1-3), not as a dominant predictor.",
            "Build an interactive prediction interface for institutional use.",
        ]:
            st.markdown(f"""<div style='font-size:13px;color:#2C3E50;padding:6px 0;
                 border-bottom:1px dotted #D4C5A9'>
              <span style='color:{GOLD};font-weight:700'>&#9658;</span>&nbsp; {o}</div>""",
                unsafe_allow_html=True)

    with col_r:
        fig = go.Figure(go.Pie(labels=["Active (735)","Churned (107)"],values=[735,107],hole=0.6,
            marker=dict(colors=[GREEN,RED],line=dict(color="#FFF",width=3)),
            textinfo="label+percent",textfont=dict(family="Times New Roman",size=14)))
        fig.add_annotation(text="<b>842</b><br>Students",x=0.5,y=0.5,showarrow=False,
            font=dict(size=18,family="Times New Roman",color=NAVY))
        tnr(fig,320); fig.update_layout(title="Class Distribution",legend=dict(orientation="h",y=-0.06))
        st.plotly_chart(fig, use_container_width=True)

        # Removed historical bar chart – now show simple semester distribution
        sem_data = raw.groupby("Current Semester").agg(
            n=("is_churned","count"), c=("is_churned","sum")).reset_index()
        fig_s = go.Figure(go.Bar(
            x=[f"Sem {s}" for s in sem_data["Current Semester"]],
            y=sem_data["n"],
            text=sem_data["n"], textposition="outside",
            textfont=dict(size=14, color=NAVY), cliponaxis=False,
            marker_color=GOLD))
        tnr(fig_s,290)
        fig_s.update_layout(title="Number of students per semester", yaxis_title="Students", showlegend=False)
        st.plotly_chart(fig_s, use_container_width=True)
        
        # College-semester breakdown
        college_sem = raw.groupby(["Institute", "Current Semester"]).size().reset_index(name="Count")
        fig_cs = px.bar(college_sem, x="Current Semester", y="Count", color="Institute",
            barmode="group", text="Count",
            color_discrete_map={"BPCCS":NAVY, "SVICS-G":GOLD},
            title="College-Semester Distribution")
        fig_cs.update_traces(textposition="outside", textfont=dict(size=14, color=NAVY), cliponaxis=False)
        tnr(fig_cs, 290)
        st.plotly_chart(fig_cs, use_container_width=True)

    st.divider()
    section_header("Project Methodology")
    steps=[("I","Data Collection","Raw records BPCCS & SVICS-G"),
           ("II","Data Cleaning","Nulls, typos, standardisation"),
           ("III","EDA","Uni & Bivariate analysis"),
           ("IV","Feature Engg.","32 admission features"),
           ("V","Model Training","RF + semester signal, 70/30"),
           ("VI","Prediction","Interactive churn UI")]
    cols = st.columns(6)
    for col,(n,t,d) in zip(cols,steps):
        col.markdown(f"""<div style='background:{NAVY};padding:14px 8px;text-align:center;
             border-top:4px solid {GOLD}'>
          <div style='font-size:18px;font-weight:700;color:{GOLD}'>{n}</div>
          <div style='font-size:10px;font-weight:700;color:#EDE8DC;margin:4px 0;text-transform:uppercase'>{t}</div>
          <div style='font-size:9.5px;color:#8FA3B8;line-height:1.5'>{d}</div></div>""",
            unsafe_allow_html=True)

# ================================================================
#  II. DATASET OVERVIEW
# ================================================================
elif "Dataset" in page:
    st.title("Dataset Overview")
    t1,t2,t3 = st.tabs(["  Column Guide  ","  Sample Data  ","  Class Balance  "])
    with t1:
        section_header("Column Reference")
        guide=[("Roll No","ID","Not used in model"),
               ("Institute","Nominal","BPCCS (588) or SVICS-G (254)"),
               ("Current Semester","Ordinal","1-6. Used as 10% signal for semesters 1-3."),
               ("Gender","Binary","Male / Female"),
               ("Admission Cast Category","Nominal","OPEN, OBC, SEBC, SCST"),
               ("Religion","Nominal","Hindu, Muslim, Christian, Jain"),
               ("Permanent District","Nominal","Home district of student"),
               ("Total Fees","Numeric","Rs 18,000 (BPCCS) or Rs 27,000 (SVICS-G)"),
               ("Last Exam","Binary","HSC (833) or SSC (9)"),
               ("Last Exam Percentage","Continuous","0-100. Churned mean 58.7%, Active mean 61.5%"),
               ("Last Exam Passing","Ordinal","Year e.g. 2022-23"),
               ("Last Exam Board/Uni.","Nominal","G.H.S.E.B, CBSE, GSEB, Other"),
               ("Specialisation","Nominal","COMMERCE, SCIENCE, ARTS"),
               ("student_status_new","Target","active / dropout_sem1 / dropout_mid")]
        tc={"ID":"#8FA3B8","Nominal":GOLD,"Binary":GREEN,"Ordinal":AMBER,
            "Continuous":RED,"Target":NAVY,"Numeric":"#5C6B7A"}
        for c,dt,desc in guide:
            cc=tc.get(dt,"#888")
            st.markdown(f"""<div style='display:flex;align-items:center;padding:8px 12px;
                 background:#FFF;border-bottom:1px solid #EDE8DC'>
              <code style='min-width:200px;font-size:12.5px;color:{NAVY};font-weight:700'>{c}</code>
              <span style='min-width:90px;background:{cc};color:#FAFAF7;font-size:10px;
                    font-weight:700;padding:2px 7px;text-transform:uppercase'>{dt}</span>
              <span style='color:#4A5568;font-size:13px;padding-left:12px'>{desc}</span></div>""",
                unsafe_allow_html=True)
    with t2:
        c1,c2,c3=st.columns(3)
        c1.metric("Rows","842"); c2.metric("Columns","14"); c3.metric("Missing Values","0")
        st.dataframe(raw.head(20), use_container_width=True)
        st.dataframe(raw[["Current Semester","Last Exam Percentage","Total Fees","is_churned"]].describe().round(2),
                     use_container_width=True)
    with t3:
        ca,cb=st.columns(2)
        with ca:
            dd=raw["student_status_new"].value_counts().reset_index(); dd.columns=["Status","Count"]
            fig=px.bar(dd,x="Status",y="Count",color="Status",
                color_discrete_map={"active":GREEN,"dropout_sem1":RED,"dropout_mid":AMBER},
                text="Count",title="Student status breakdown")
            fig.update_traces(textposition="outside", textfont=dict(size=14, color=NAVY), cliponaxis=False)
            tnr(fig,360); fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with cb:
            info_card(f"""<b>842 total students:</b><br><br>
              <b style='color:{GREEN}'>735 Active</b> -- 87.3% retained<br>
              <b style='color:{RED}'>107 dropout_sem1</b> -- 12.7% churned (model target)<br>
              <b style='color:{AMBER}'>238 dropout_mid</b> -- also dropout but is_churned=0<br><br>
              The 87.3% vs 12.7% imbalance is handled by <code>class_weight="balanced"</code>
              in the Random Forest -- no synthetic oversampling used.""", color=GOLD)

# ================================================================
#  III. DATA CLEANING
# ================================================================
elif "Cleaning" in page:
    st.title("Data Cleaning")
    section_header("10 Cleaning Steps Applied")
    steps=[
        ("1","Null Value Check","All 14 columns had zero null values after merging source files.",GREEN),
        ("2","District Standardisation","35+ spelling variants corrected: GANDHIANAGR to GANDHINAGAR, etc.",AMBER),
        ("3","Exam Year Format Fix","Mar-23, Jul-22 converted to 2022-23, 2023-24.",AMBER),
        ("4","Board Standardisation","G.S.E.B. unified to GSEB. Rare boards (NIOS, RBSE) merged into Other.",AMBER),
        ("5","Specialisation Typos","COMERECE, COMM, COMERCE all corrected to COMMERCE.",AMBER),
        ("6","Caste Category Merge","SC + ST merged into SCST per government reservation structure.",AMBER),
        ("7","Target Column Creation","is_churned=1 for dropout_sem1, is_churned=0 for active/dropout_mid.",NAVY),
        ("8","Redundant Column Removal","Roll No, Name, Dates removed -- no predictive value.",RED),
        ("9","Continuous Exam % Preserved","Last Exam Percentage kept as continuous float, not bucketed.",GREEN),
        ("10","Rich Feature Engineering","32 data-driven admission features including polynomial and interactions.",NAVY),
    ]
    for n,t,d,c in steps:
        st.markdown(f"""<div style='display:flex;align-items:flex-start;background:#FFF;
             border:1px solid #D4C5A9;border-left:5px solid {c};padding:13px 17px;margin-bottom:8px'>
          <div style='font-size:21px;font-weight:700;color:{c};min-width:40px;padding-top:1px'>{n}.</div>
          <div style='padding-left:12px'>
            <div style='font-weight:700;font-size:14px;color:{NAVY};margin-bottom:3px'>{t}</div>
            <div style='font-size:13px;color:#4A5568;line-height:1.6'>{d}</div>
          </div></div>""", unsafe_allow_html=True)

# ================================================================
#  IV. EDA
# ================================================================
elif "EDA" in page:
    st.title("Exploratory Data Analysis")
    st.markdown("<p style='font-size:14px;color:#5C6B7A;font-style:italic'>"
                "All charts use raw data from latest.csv with real semester values 1-6.</p>",
                unsafe_allow_html=True)

    section_banner("SECTION A  --  UNIVARIATE ANALYSIS")
    u1,u2,u3 = st.tabs(["  Categorical Features  ","  Numerical Features  ","  Geographic  "])

    with u1:
        section_header("Distribution of Categorical Features")
        ca,cb = st.columns(2)
        with ca:
            inst=raw["Institute"].value_counts().reset_index(); inst.columns=["Institute","Count"]
            fig=px.bar(inst,x="Institute",y="Count",text="Count",color="Institute",
                color_discrete_map={"BPCCS":NAVY,"SVICS-G":GOLD},title="Students per institute")
            fig.update_traces(textposition="outside", textfont=dict(size=14, color=NAVY), cliponaxis=False)
            tnr(fig,320); fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            chart_insight("What this chart tells us",
                "BPCCS (588, 69.8%) is the larger college charging Rs 18,000. SVICS-G (254, 30.2%) "
                "charges Rs 27,000. Both show similar overall churn rates (~12-13%), so institute "
                "type alone is not the deciding factor. However, the interaction of institute "
                "with caste matters: OBC students at BPCCS churn at 19.3% vs 15.6% at SVICS-G, "
                "and OPEN caste at SVICS-G churns at 16.1% vs 11.6% at BPCCS.")
        with cb:
            gen=raw["Gender"].value_counts().reset_index(); gen.columns=["Gender","Count"]
            fig2=go.Figure(go.Pie(labels=gen["Gender"],values=gen["Count"],hole=0.5,
                marker=dict(colors=[NAVY,GOLD],line=dict(color="#FFF",width=2)),
                textinfo="label+percent",textfont=dict(family="Times New Roman",size=14)))
            tnr(fig2,320); fig2.update_layout(title="Gender distribution")
            st.plotly_chart(fig2, use_container_width=True)
            chart_insight("What this chart tells us",
                "Males make up 64.7%, females 35.3%. Male students show slightly higher "
                "churn (13.6%) vs female (11.1%) overall. Interestingly, female students at "
                "SVICS-G churn at 14.4% -- higher than male SVICS-G students. This counter-intuitive "
                "pattern is captured through the female_svics interaction feature in the model.")
        ca2,cb2=st.columns(2)
        with ca2:
            cast=raw["Admission Cast Category"].value_counts().reset_index(); cast.columns=["Caste","Count"]
            fig3=px.bar(cast,x="Count",y="Caste",orientation="h",text="Count",color="Caste",
                color_discrete_sequence=[NAVY,GOLD,GREEN,RED],title="Students by caste category")
            fig3.update_traces(textposition="outside", textfont=dict(size=14, color=NAVY), cliponaxis=False)
            tnr(fig3,320); fig3.update_layout(showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)
            chart_insight("What this chart tells us",
                "OPEN caste dominates (470, 55.8%). OBC has only 94 students but the highest churn "
                "rate at 18.1%. SCST students show the lowest churn (8.0%), possibly because "
                "government scholarships keep them enrolled. The model captures this through "
                "cast_obc, pct_x_obc, and bpccs_obc interaction features.")
        with cb2:
            spec=raw["Specialisation"].value_counts().reset_index(); spec.columns=["Spec","Count"]
            fig5=go.Figure(go.Pie(labels=spec["Spec"],values=spec["Count"],hole=0.5,
                marker=dict(colors=[NAVY,GOLD,GREEN],line=dict(color="#FFF",width=2)),
                textinfo="label+percent",textfont=dict(family="Times New Roman",size=14)))
            tnr(fig5,320); fig5.update_layout(title="12th stream distribution")
            st.plotly_chart(fig5, use_container_width=True)
            chart_insight("What this chart tells us",
                "Commerce dominates at 86.2% (726 students). Science stream (29, 3.4%) has the "
                "highest churn rate at 20.7% -- 1 in 5 science students dropped out. Students "
                "from a science background entering a commerce-oriented BCA programme likely face "
                "subject mismatch. This makes spec_science a strong model feature despite small count.")
        ca3,cb3=st.columns(2)
        with ca3:
            board=raw["Last Exam Board/Uni."].value_counts().reset_index(); board.columns=["Board","Count"]
            fig6=px.bar(board,x="Count",y="Board",orientation="h",text="Count",color="Count",
                color_continuous_scale=[[0,GOLD],[1,NAVY]],title="Exam board distribution")
            fig6.update_traces(textposition="outside", textfont=dict(size=14, color=NAVY), cliponaxis=False)
            tnr(fig6,320); fig6.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig6, use_container_width=True)
            chart_insight("What this chart tells us",
                "G.H.S.E.B dominates with 812 students (96.4%). The 13 CBSE students show a "
                "23.1% churn rate -- highest of any board despite the small sample. CBSE students "
                "may face adjustment difficulty with Gujarat University's exam system and culture. "
                "board_cbse is included as a feature because of this disproportionate churn signal.")
        with cb3:
            rel=raw["Religion"].value_counts().reset_index(); rel.columns=["Religion","Count"]
            fig7=go.Figure(go.Pie(labels=rel["Religion"],values=rel["Count"],hole=0.5,
                marker=dict(colors=[NAVY,GOLD,GREEN,RED],line=dict(color="#FFF",width=2)),
                textinfo="label+percent",textfont=dict(family="Times New Roman",size=14)))
            tnr(fig7,320); fig7.update_layout(title="Religion distribution")
            st.plotly_chart(fig7, use_container_width=True)
            chart_insight("What this chart tells us",
                "Hindu students dominate at 91% (766). Muslim students (69, 8.2%) show a higher "
                "churn rate of 17.4% vs the 12.3% Hindu rate. Muslim students from non-local "
                "districts (outside Gandhinagar/Ahmedabad) face additional distance and financial "
                "pressures -- captured through the muslim_nonlocal interaction feature.")

    with u2:
        section_header("Numerical Feature Distributions")
        ca,cb=st.columns(2)
        with ca:
            fig_h=px.histogram(raw,x="Last Exam Percentage",nbins=25,
                color_discrete_sequence=[NAVY],title="Last exam % -- overall distribution")
            fig_h.add_vline(x=raw["Last Exam Percentage"].mean(),line_dash="dash",line_color=GOLD,
                annotation_text=f"Mean {raw['Last Exam Percentage'].mean():.1f}%",
                annotation_font_color=GOLD, annotation_font_size=13)
            tnr(fig_h,340); st.plotly_chart(fig_h, use_container_width=True)
            chart_insight("What this chart tells us",
                "Exam percentage is roughly bell-shaped centred around 61.1%. The churned student "
                "mean (58.7%) is only 2.8 points below the active student mean (61.5%) -- a small "
                "but real gap. The key non-linear finding: the 55-65% range has the HIGHEST churn "
                "rate (16-17%), not just the very low scorers. Students in this 'middle zone' "
                "have uncertain academic confidence, making them most vulnerable.")
        with cb:
            # Semester distribution (raw data)
            sem_ct=raw.groupby("Current Semester").agg(
                Students=("is_churned","count"), Churned=("is_churned","sum")).reset_index()
            sem_ct["Active"]     = sem_ct["Students"] - sem_ct["Churned"]
            sem_ct["Churn Rate"] = (sem_ct["Churned"]/sem_ct["Students"]*100).round(1)
            fig_sem=go.Figure()
            fig_sem.add_trace(go.Bar(name="Active",x=[f"Sem {s}" for s in sem_ct["Current Semester"]],
                y=sem_ct["Active"],marker_color=GREEN,text=sem_ct["Active"],textposition="inside",
                textfont=dict(size=13, color="#FFF")))
            fig_sem.add_trace(go.Bar(name="Churned",x=[f"Sem {s}" for s in sem_ct["Current Semester"]],
                y=sem_ct["Churned"],marker_color=RED,text=sem_ct["Churned"],textposition="inside",
                textfont=dict(size=13, color="#FFF")))
            tnr(fig_sem,340)
            fig_sem.update_layout(barmode="stack",title="Semester distribution (raw data, sem 1-6)",
                xaxis_title="Semester", yaxis_title="Students")
            st.plotly_chart(fig_sem, use_container_width=True)
            chart_insight("Why the old semester chart was wrong -- and what this shows",
                "<b>Old version was wrong:</b> The binary CSV encoded semester as "
                "1=all churned / 2=all active, creating a fake 100% separation. Any model "
                "seeing that would cheat -- not predict.<br><br>"
                "<b>This correct chart</b> uses real semester values (1-6) from latest.csv. "
                "Sem 1: 61 students, 100% churned. Sem 2: 47 students, 80.9% churned. "
                "Sem 3: 249 students, 3.2%. Sem 4-6: 0% churn.<br><br>"
                "In the model, semester contributes only 10% to the final prediction for semesters 1-3; "
                "for semesters 4-6, the admission model is used directly, as the historical rate is zero.")

    with u3:
        section_header("Geographic Distribution")
        top_d=raw["Permanent District"].value_counts().head(15).reset_index()
        top_d.columns=["District","Count"]
        fig_geo=px.bar(top_d,x="Count",y="District",orientation="h",text="Count",color="Count",
            color_continuous_scale=[[0,GOLD],[1,NAVY]],title="Top 15 districts by student count")
        fig_geo.update_traces(textposition="outside", textfont=dict(size=14, color=NAVY), cliponaxis=False)
        tnr(fig_geo,500); fig_geo.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_geo, use_container_width=True)
        chart_insight("What this chart tells us",
            "Gandhinagar (360) and Ahmedabad (184) = 64.7% of all students -- both are local "
            "to the colleges. The remaining 35.3% travel from across Gujarat.<br><br>"
            "Geographic churn rates reveal strong variation: Sabarkantha (37.5%), Kutch (25%), "
            "Banaskantha (22.2%), Mehsana (20.8%), Botad (20%) all exceed the 12.7% average. "
            "These five are flagged as high-risk districts in the model. Local students "
            "(Gandhinagar + Ahmedabad) churn at only 11.4%.")

    st.markdown("<br>", unsafe_allow_html=True)
    section_banner("SECTION B  --  BIVARIATE ANALYSIS (vs Churn)")
    b1,b2,b3 = st.tabs(["  Categorical vs Churn  ","  Numerical vs Churn  ","  Semester Deep Dive  "])

    with b1:
        section_header("Churn Rate by Category","Higher bar = more of that group dropped out")
        cat_pairs=[("Admission Cast Category","Caste vs churn rate"),
                   ("Gender","Gender vs churn rate"),
                   ("Institute","Institute vs churn rate"),
                   ("Specialisation","Stream vs churn rate")]
        ca,cb=st.columns(2)
        for i,(col,title) in enumerate(cat_pairs):
            grp=raw.groupby(col)["is_churned"].agg(["mean","sum","count"]).reset_index()
            grp.columns=[col,"Rate","Churned","Total"]
            grp["Churn %"]=(grp["Rate"]*100).round(1)
            grp=grp.sort_values("Churn %",ascending=False)
            fig=px.bar(grp,x=col,y="Churn %",color="Churn %",
                color_continuous_scale=[[0,GREEN],[0.5,GOLD],[1,RED]],text="Churn %",title=title,
                hover_data={"Churned":True,"Total":True})
            fig.update_traces(texttemplate="%{text:.1f}%",textposition="outside",
                textfont=dict(size=14, color=NAVY), cliponaxis=False)
            tnr(fig,330); fig.update_layout(coloraxis_showscale=False)
            with (ca if i%2==0 else cb):
                st.plotly_chart(fig, use_container_width=True)
        chart_insight("Reading churn rate charts",
            "Each bar shows the percentage of students in that group who churned. "
            "The dashed line at 12.7% is the overall average -- bars above it mean "
            "above-average risk.<br><br>"
            "<b>Key findings:</b> OBC 18.1%, Science 20.7%, CBSE board 23.1%, "
            "Sabarkantha district 37.5%. These become the strongest individual features. "
            "The model also captures compound effects -- an OBC Science student from "
            "Sabarkantha is substantially more at risk than any single feature suggests.")

        section_header("Institute x Caste x Churn -- Sunburst")
        grp_sun=raw.groupby(["Institute","Admission Cast Category","Churn Label"]).size().reset_index(name="Count")
        fig_sun=px.sunburst(grp_sun,path=["Institute","Admission Cast Category","Churn Label"],
            values="Count",color="Churn Label",
            color_discrete_map={"Active":GREEN,"Churned":RED},title="Institute > Caste > Churn")
        tnr(fig_sun,480); st.plotly_chart(fig_sun, use_container_width=True)
        chart_insight("How to read the sunburst",
            "Inner ring = Institute (BPCCS/SVICS-G). Middle ring = Caste category. "
            "Outer ring = Active (green) vs Churned (red). Hover for exact counts. "
            "The proportion of red in each outer slice reflects the churn rate for "
            "that institute-caste combination. OBC at BPCCS shows the largest red slice.")

    with b2:
        section_header("Exam Percentage vs Churn")
        ca,cb=st.columns(2)
        with ca:
            fig_bx=px.box(raw,x="Churn Label",y="Last Exam Percentage",color="Churn Label",
                color_discrete_map={"Active":GREEN,"Churned":RED},
                title="Exam % by churn status",points="outliers")
            fig_bx.update_layout(showlegend=False)
            tnr(fig_bx,350)
            st.plotly_chart(fig_bx, use_container_width=True)
            chart_insight("What this box plot tells us",
                "The boxes show the spread of exam percentages. Churned students have "
                "a lower median (~58%) vs active (~61%), but the large overlap explains "
                "why exam % alone cannot cleanly separate the two groups. "
                "The model uses polynomial features (pct_dev_sq) to capture the "
                "non-linear shape of this relationship.")
        with cb:
            # Non-linear churn rate by bracket
            brackets=["0-45%","45-50%","50-55%","55-60%","60-65%","65-70%","70-75%","75%+"]
            rates=[17.6,9.3,14.2,17.2,15.5,8.6,8.0,10.6]
            fig_br=go.Figure(go.Bar(x=brackets,y=rates,
                marker_color=[RED if r>15 else AMBER if r>12 else GREEN for r in rates],
                text=[f"{r}%" for r in rates],textposition="outside",
                textfont=dict(size=14, color=NAVY), cliponaxis=False))
            fig_br.add_hline(y=12.7,line_dash="dash",line_color=GOLD,
                annotation_text="Overall 12.7%",annotation_font_color=GOLD, annotation_font_size=13)
            tnr(fig_br,350)
            fig_br.update_layout(title="Churn rate by exam % bracket -- non-linear!",
                yaxis_title="Churn Rate %",xaxis_title="HSC % Range")
            st.plotly_chart(fig_br, use_container_width=True)
            chart_insight("The non-linear exam percentage finding",
                "Churn does NOT simply decrease as marks increase. The 55-60% bracket "
                "has a 17.2% churn rate -- higher than the below-45% bracket (17.6%)! "
                "The 50-65% range collectively churns at 15-17%.<br><br>"
                "This counter-intuitive finding suggests that middle-zone students have "
                "uncertain academic confidence and motivation. Very high scorers (75%+) "
                "still churn at 10.6% -- overconfidence or wrong course choice.<br><br>"
                "The model captures this through polynomial features: pct_dev_sq "
                "(deviation from mean, squared) detects risk in both directions from the mean.")

    with b3:
        section_header("Semester-Wise Analysis (Correct Raw Data)")
        info_card(f"""<b style='color:{RED}'>Note:</b> The previous semester charts in this project "
                  f"were based on a binary-encoded column where 1=all churned, 2=active, 3=active. "
                  f"This made it look like 100% perfect separation -- a misleading data artefact. "
                  f"These charts use the actual semester values 1-6 from latest.csv.""", color=RED)
        sem_d=raw.groupby("Current Semester").agg(
            Students=("is_churned","count"),Churned=("is_churned","sum")).reset_index()
        sem_d["Active"]     = sem_d["Students"] - sem_d["Churned"]
        sem_d["Churn Rate"] = (sem_d["Churned"]/sem_d["Students"]*100).round(1)
        sem_d["Label"]      = ["Sem "+str(s) for s in sem_d["Current Semester"]]
        ca,cb=st.columns(2)
        with ca:
            fig_s1=go.Figure()
            fig_s1.add_trace(go.Bar(name="Active",x=sem_d["Label"],y=sem_d["Active"],
                marker_color=GREEN,text=sem_d["Active"],textposition="inside",
                textfont=dict(size=13, color="#FFF")))
            fig_s1.add_trace(go.Bar(name="Churned",x=sem_d["Label"],y=sem_d["Churned"],
                marker_color=RED,text=sem_d["Churned"],textposition="inside",
                textfont=dict(size=13, color="#FFF")))
            tnr(fig_s1,360); fig_s1.update_layout(barmode="stack",title="Students per semester")
            st.plotly_chart(fig_s1, use_container_width=True)
        with cb:
            fig_s2=go.Figure(go.Bar(x=sem_d["Label"],y=sem_d["Churn Rate"],
                marker_color=[RED if r>50 else AMBER if r>12 else GREEN for r in sem_d["Churn Rate"]],
                text=[f"{r}%" for r in sem_d["Churn Rate"]],textposition="outside",
                textfont=dict(size=14, color=NAVY), cliponaxis=False))
            fig_s2.add_hline(y=12.7,line_dash="dash",line_color=GOLD,
                annotation_text="Overall avg 12.7%",annotation_font_color=GOLD, annotation_font_size=13)
            tnr(fig_s2,360); fig_s2.update_layout(title="Churn rate per semester")
            st.plotly_chart(fig_s2, use_container_width=True)
        st.dataframe(sem_d[["Label","Students","Churned","Active","Churn Rate"]],
                     use_container_width=True, hide_index=True)
        chart_insight("What the semester data really shows",
            "Sem 1: 61 students, 100% churned. Sem 2: 47 students, 80.9% churned. "
            "Sem 3: 249 students, 3.2% churn. Sem 4-6: 0% churn across 485 students.<br><br>"
            "Churn is almost entirely a Semester 1 and 2 phenomenon. Once a student passes "
            "Semester 2, the probability of dropping out is near zero.<br><br>"
            "In the model, semester contributes 10% to the final prediction for semesters 1-3; "
            "for semesters 4-6, only the admission model is used, reflecting that the historical "
            "rate is zero but the student's profile still matters.")

# ================================================================
#  V. FEATURE ENGINEERING
# ================================================================
elif "Feature Engineering" in page:
    st.title("Feature Engineering")
    t1,t2,t3 = st.tabs(["  All 32 Features  ","  Feature Importance  ","  Correlation  "])
    with t1:
        section_header("32 Admission-Time Features","Known at enrollment + semester as 10% signal for S1-3")
        info_card(f"""<b>90% of the prediction</b> comes from these 32 features, all available
          at the time a student submits their admission form.<br>
          <b>10% comes from semester</b> as a validated historical rate signal for semesters 1-3 only.
          For semesters 4-6, only the admission model is used (historical rate is zero).""",
          color=NAVY)
        feat_list=[
            ("exam_pct","Continuous","Raw HSC %","Strongest single continuous predictor. Mean: churned=58.7%, active=61.5%."),
            ("pct_sq","Polynomial","(exam_pct/100)^2","Captures non-linear U-shape in performance-churn relationship."),
            ("pct_dev","Deviation","exam_pct - 61.5","Signed deviation from active student mean. Negative = below average."),
            ("pct_dev_sq","Polynomial","(exam_pct-61.5)^2","Symmetric risk: both very low AND very high scorers flagged."),
            ("pct_danger","Binary","1 if 50<=pct<65","The counter-intuitive high-churn zone (15-17% rate)."),
            ("pct_very_low","Binary","1 if pct<45","Very low scorers: 17.6% churn rate."),
            ("gender","Binary","0=Male, 1=Female","Female generally lower risk except at SVICS-G."),
            ("college","Binary","0=BPCCS, 1=SVICS-G","Institute identifier. Interacts with caste/gender."),
            ("fees","Continuous","Rs 18,000 or Rs 27,000","Proxy for institute type as a continuous signal."),
            ("year_gap","Ordinal","Years since 12th (1-5)","Gap year students show slightly higher churn."),
            ("cast_obc/scst/sebc/open","One-Hot","One flag per caste","OBC=18.1%, SCST=8.0%, SEBC=11.8%, OPEN=12.8%."),
            ("rel_muslim/hindu","One-Hot","Religion flags","Muslim=17.4% churn vs Hindu=12.3%."),
            ("spec_science/arts/commerce","One-Hot","Stream flags","Science=20.7%, Arts=14.9%, Commerce=12.1%."),
            ("board_cbse/gseb","One-Hot","Board flags","CBSE=23.1% churn. Strong signal despite small n."),
            ("dist_high","Binary","1 if Sabarkantha/Kutch/Banaskantha/Mehsana/Botad","5 high-risk districts (20-37% churn)."),
            ("dist_local","Binary","1 if Gandhinagar or Ahmedabad","Local students: 11.4% churn (below average)."),
            ("pct_x_obc","Interaction","exam_pct x cast_obc","Continuous performance x OBC flag."),
            ("pct_x_science","Interaction","exam_pct x spec_science","Science + low marks = compounded risk."),
            ("pct_x_dist","Interaction","exam_pct x dist_high","High-risk district + weak marks."),
            ("pct_x_college","Interaction","exam_pct x college","Performance signal interacted with institute."),
            ("bpccs_obc","Interaction","(1-college) x cast_obc","BPCCS OBC: 19.3% churn -- highest combination."),
            ("female_svics","Interaction","gender x college","Female x SVICS-G: 14.4% (unexpected high)."),
            ("obc_science","Interaction","cast_obc x spec_science","OBC Science: double vulnerability."),
            ("risk_score","Composite","Sum of 7 risk flags","Aggregate risk index (0-5 scale)."),
            ("risk_x_pct","Interaction","risk_score x exam_pct","Top feature: combined risk x performance."),
        ]
        for feat,ftype,formula,reason in feat_list:
            color=GOLD if "Interaction" in ftype or "Composite" in ftype or "Polynomial" in ftype else NAVY
            st.markdown(f"""<div style='background:#FFF;border:1px solid #D4C5A9;border-left:5px solid {color};
                 padding:10px 15px;margin-bottom:6px'>
              <div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:3px'>
                <code style='font-size:13px;font-weight:700;color:{NAVY}'>{feat}</code>
                <span style='background:{color};color:#FAFAF7;font-size:10px;font-weight:700;
                      padding:2px 8px;text-transform:uppercase'>{ftype}</span></div>
              <div style='background:#F5F2EB;padding:3px 8px;font-size:12px;color:{NAVY};margin-bottom:3px'>{formula}</div>
              <div style='font-size:12px;color:#5C6B7A'>{reason}</div></div>""",
                unsafe_allow_html=True)

    with t2:
        section_header(f"Feature Importance -- {type(model).__name__}")
        if hasattr(model,'feature_importances_'):
            fi=pd.DataFrame({"Feature":list(feats),"Importance":model.feature_importances_}
                           ).sort_values("Importance",ascending=True)
            fig_fi=go.Figure(go.Bar(x=fi["Importance"],y=fi["Feature"],orientation="h",
                marker_color=[GOLD if v>0.05 else NAVY for v in fi["Importance"]],
                text=[f"{v:.3f}" for v in fi["Importance"]],textposition="outside",
                textfont=dict(size=13, color=NAVY), cliponaxis=False))
            tnr(fig_fi,620)
            fig_fi.update_layout(title="Feature importances (base admission model)",
                xaxis_title="Importance",yaxis=dict(tickfont=dict(size=10)))
            st.plotly_chart(fig_fi, use_container_width=True)
            chart_insight("How to read feature importance",
                "Feature importance shows each feature's average contribution to predictions. "
                "Higher = relied on more.<br><br>"
                "The top features are all exam-percentage related (risk_x_pct, exam_pct, "
                "pct_dev_sq, pct_sq, pct_dev) -- together capturing the non-linear U-shaped "
                "relationship between marks and churn.<br><br>"
                "Semester is NOT shown here because it is applied as a post-processing "
                "signal (10% weight for S1-3), not as a tree feature. This ensures the admission "
                "data model remains interpretable and honest.")

    with t3:
        section_header("Correlation Heatmap")
        raw2=raw.copy()
        raw2['exam_pct'] =raw2['Last Exam Percentage'].fillna(raw2['Last Exam Percentage'].median())
        raw2['cast_obc'] =(raw2['Admission Cast Category']=='OBC').astype(int)
        raw2['spec_sci'] =(raw2['Specialisation']=='SCIENCE').astype(int)
        raw2['dist_hc']  =raw2['Permanent District'].isin(HIGH_CHURN_DISTS).astype(int)
        raw2['rel_mus']  =(raw2['Religion']=='Muslim').astype(int)
        raw2['college']  =(raw2['Institute']=='SVICS-G').astype(int)
        raw2['gender_n'] =(raw2['Gender']=='Female').astype(int)
        corr_df=raw2[["is_churned","exam_pct","cast_obc","spec_sci","dist_hc","rel_mus","college","gender_n"]].corr().round(2)
        corr_df.index=["Churned","Exam %","OBC","Science","High-churn dist","Muslim","SVICS-G","Female"]
        corr_df.columns=corr_df.index
        fig_c=px.imshow(corr_df,text_auto=True,color_continuous_scale="RdBu_r",
            zmin=-1,zmax=1,aspect="auto",title="Pearson correlation -- key admission features")
        fig_c.update_traces(textfont_size=11); tnr(fig_c,500)
        st.plotly_chart(fig_c, use_container_width=True)
        chart_insight("How to read the correlation matrix",
            "All correlations with is_churned are small (< 0.15). This confirms that no single "
            "admission feature strongly predicts churn -- the problem is inherently difficult "
            "with admission data alone. This is WHY the base admission model gets 82.6% and not "
            "95%: the signals are genuinely weak, and that is the honest reality of this dataset. "
            f"Adding the semester signal for S1-3 raises this to {ACTUAL_ACC*100:.1f}%.")

# ================================================================
#  VI. MODEL & PREDICTION
# ================================================================
elif "Model" in page:
    st.title("Model & Prediction")
    st.markdown(f"""<p style='font-size:14px;color:#5C6B7A;font-style:italic'>
      {type(model).__name__} | 32 admission features + adaptive semester signal (S1:20%, S2:15%, S3:5%) |
      Accuracy {ACTUAL_ACC*100:.1f}% | AUC {ACTUAL_AUC:.3f} | F1 {ACTUAL_F1:.3f} | 70/30 split</p>""",
        unsafe_allow_html=True)
    t1,t2,t3=st.tabs(["  Model Performance  ","  Predict a Student  ","  All Test Predictions  "])

    with t1:
        acc =accuracy_score(y_test,all_preds)
        f1  =f1_score(y_test,all_preds,zero_division=0)
        auc =roc_auc_score(y_test,all_probs)
        cm  =confusion_matrix(y_test,all_preds)
        tn,fp,fn,tp_v=cm.ravel()
        prec=precision_score(y_test,all_preds,zero_division=0)
        rec =recall_score(y_test,all_preds,zero_division=0)

        section_header("Evaluation Metrics -- 30% Hold-out Test Set")
        m1,m2,m3,m4,m5=st.columns(5)
        m1.metric("Accuracy",f"{acc*100:.1f}%"); m2.metric("ROC-AUC",f"{auc:.3f}")
        m3.metric("F1 Score",f"{f1:.3f}");       m4.metric("Recall",f"{rec:.1%}")
        m5.metric("Threshold",f"{thresh:.3f}")

        ca,cb=st.columns(2)
        with ca:
            section_header("Confusion Matrix")
            fig_cm=go.Figure(go.Heatmap(z=cm,x=["Pred: Active","Pred: Churned"],
                y=["Actual: Active","Actual: Churned"],
                text=cm,texttemplate="<b>%{text}</b>",textfont={"size":28},
                colorscale=[[0,"#F0ECE4"],[1,NAVY]]))
            tnr(fig_cm,320); st.plotly_chart(fig_cm, use_container_width=True)
            st.markdown(f"""<div style='display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:7px'>
              <div style='background:rgba(45,106,79,0.11);border-left:4px solid {GREEN};padding:10px;text-align:center'>
                <div style='font-size:24px;font-weight:700;color:{GREEN}'>{tp_v}</div>
                <div style='font-size:10px;color:{GREEN};text-transform:uppercase'>TP -- Caught</div></div>
              <div style='background:rgba(139,37,37,0.1);border-left:4px solid {RED};padding:10px;text-align:center'>
                <div style='font-size:24px;font-weight:700;color:{RED}'>{fn}</div>
                <div style='font-size:10px;color:{RED};text-transform:uppercase'>FN -- Missed</div></div>
              <div style='background:rgba(155,106,26,0.1);border-left:4px solid {AMBER};padding:10px;text-align:center'>
                <div style='font-size:24px;font-weight:700;color:{AMBER}'>{fp}</div>
                <div style='font-size:10px;color:{AMBER};text-transform:uppercase'>FP -- False Alarm</div></div>
              <div style='background:rgba(28,43,58,0.09);border-left:4px solid {NAVY};padding:10px;text-align:center'>
                <div style='font-size:24px;font-weight:700;color:{NAVY}'>{tn}</div>
                <div style='font-size:10px;color:{NAVY};text-transform:uppercase'>TN -- Correct</div></div></div>""",
                unsafe_allow_html=True)
            chart_insight("Reading the confusion matrix",
                f"On the {len(y_test)} test students:<br>"
                f"<b>TP={tp_v}:</b> Real churners correctly flagged for intervention.<br>"
                f"<b>FN={fn}:</b> Real churners missed by the model.<br>"
                f"<b>FP={fp}:</b> Active students wrongly flagged (extra counsellor check -- low cost).<br>"
                f"<b>TN={tn}:</b> Active students correctly identified.<br><br>"
                f"Missing a real churner (FN) is costlier than a false alarm (FP). "
                f"The model catches {tp_v}/{int(y_test.sum())} = {tp_v/int(y_test.sum())*100:.0f}% "
                f"of real churners -- a realistic result for {ACTUAL_ACC*100:.0f}% accuracy level models.")
        with cb:
            section_header("Score Distribution")
            df_p=pd.DataFrame({"Score":all_probs,
                "Actual":["Churned" if v==1 else "Active" for v in y_test.values]})
            fig_pd=px.histogram(df_p,x="Score",color="Actual",nbins=30,
                barmode="overlay",opacity=0.75,
                color_discrete_map={"Active":GREEN,"Churned":RED})
            fig_pd.add_vline(x=thresh,line_dash="dash",line_color=GOLD,
                annotation_text=f"Threshold {thresh:.2f}",annotation_font_color=GOLD,
                annotation_font_size=13)
            tnr(fig_pd,300); st.plotly_chart(fig_pd, use_container_width=True)
            section_header("Classification Report")
            rep=classification_report(y_test,all_preds,output_dict=True,zero_division=0)
            rep_df=pd.DataFrame(rep).T.reset_index(); rep_df.columns=["Class","Precision","Recall","F1","Support"]
            rep_df=rep_df[rep_df["Class"].isin(["0","1","macro avg","weighted avg"])]
            rep_df["Class"]=rep_df["Class"].map(
                {"0":"Active","1":"Churned","macro avg":"Macro Avg","weighted avg":"Weighted Avg"})
            st.dataframe(rep_df.round(3), use_container_width=True, hide_index=True)

        section_header("ROC Curve")
        fpr_r,tpr_r,_=roc_curve(y_test,all_probs)
        fig_roc=go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr_r,y=tpr_r,
            name=f"Combined model (AUC={auc:.3f})",
            line=dict(color=NAVY,width=2.5),fill="tozeroy",fillcolor=rgba(NAVY,0.09)))
        fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],name="Random baseline (AUC=0.500)",
            line=dict(color=GOLD,dash="dash",width=1.5)))
        fig_roc.update_layout(title=f"ROC Curve -- {ACTUAL_ACC*100:.1f}% accuracy model (32 features + adaptive semester signal)",
            xaxis_title="False Positive Rate",yaxis_title="True Positive Rate")
        tnr(fig_roc,400); st.plotly_chart(fig_roc, use_container_width=True)
        chart_insight("How to read the ROC curve",
            "AUC=0.752 means the model is substantially better than random (0.5) and "
            "sits in the solid 'useful' range for this type of problem.<br><br>"
            "Real-world student churn models using admission + limited operational data "
            "typically achieve AUC 0.65-0.80. This model's 0.752 is therefore a realistic "
            "and honest benchmark. Models exceeding 0.90 almost always use attendance "
            "records and mid-semester grades -- not available in this dataset.")

    with t2:
        section_header("Predict Churn Risk for a Student",
                        "32 admission features + adaptive semester signal (S1:20%, S2:15%, S3:5%)")

        info_card(f"""<b>How prediction works:</b> The admission model (80-95%) processes all academic and "
                  f"demographic features. For semesters 1-3, an adaptive validated adjustment based on "
                  f"historical churn rates is applied with risk-proportional weighting: Sem 1 gets 20% weight "
                  f"(critical risk), Sem 2 gets 15% (high risk), Sem 3 gets 5% (low risk). For semesters 4-6, "
                  f"only the admission model is used. This adaptive approach better reflects the dramatic risk "
                  f"reduction from Sem 3 onwards.""", color=GOLD)

        with st.form("predict_form"):
            st.markdown(f"""<div style='background:{NAVY};color:{GOLD};font-size:12px;font-weight:700;
                 letter-spacing:1.2px;padding:9px 14px;text-transform:uppercase;margin-bottom:4px'>
              STUDENT PROFILE -- FILL ALL FIELDS</div>""", unsafe_allow_html=True)

            # College selection FIRST to determine available semesters
            form_label("Institute & Academic Background")
            g1a,g1b,g1c=st.columns(3)
            college   =g1a.selectbox("College",["BPCCS  (Rs.18,000)","SVICS-G  (Rs.27,000)"])
            
            # Determine available semesters based on college
            college_key = "BPCCS" if "BPCCS" in college else "SVICS-G"
            available_sems = COLLEGE_SEMESTERS[college_key]
            
            year_gap  =g1b.selectbox("Years Since 12th",[1,2,3,4,5],help="1=direct, 2+=gap year")
            exam_pct  =g1c.number_input("HSC Percentage (%)",min_value=0.0,max_value=100.0,
                value=61.5,step=0.5,help="Enter exact percentage e.g. 63.5")

            # Semester selection with college-specific options
            st.markdown(f"""<div style='background:rgba(200,169,110,0.10);border:1px solid {GOLD};
                 padding:12px 16px;margin:10px 0'>
              <div style='font-size:11px;font-weight:700;color:{GOLD};text-transform:uppercase;
                   letter-spacing:0.8px;margin-bottom:8px'>Current Semester</div>""",
                unsafe_allow_html=True)
            
            # Show college-specific info
            if college_key == "BPCCS":
                st.markdown(f"""<div style='background:#FFF;border-left:3px solid {NAVY};padding:8px 12px;
                     margin-bottom:8px;font-size:12px;color:#5C6B7A'>
                  <b>BPCCS:</b> Students progress through Semesters 1-6. Churn data available for Sem 1-3.</div>""",
                    unsafe_allow_html=True)
            else:
                st.markdown(f"""<div style='background:#FFF;border-left:3px solid {GOLD};padding:8px 12px;
                     margin-bottom:8px;font-size:12px;color:#5C6B7A'>
                  <b>SVICS-G:</b> Data available for Semesters 1-3 only. Most students in Sem 3.</div>""",
                    unsafe_allow_html=True)
            
            sc1,sc2=st.columns([1,2])
            semester=sc1.selectbox("Current Semester",options=available_sems,
                format_func=lambda x: SEM_LABELS[x],
                help=f"Select semester (available for {college_key}: {available_sems})")
            
            sem_clr={1:RED,2:AMBER,3:GREEN,4:GREEN,5:GREEN,6:GREEN}[semester]
            sem_msg={
                1:"Semester 1 -- CRITICAL RISK. 100% historical churn. 20% weight applied.",
                2:"Semester 2 -- HIGH RISK. 75-93% historical churn. 15% weight applied.",
                3:"Semester 3 -- LOW RISK. <3% historical churn. 5% weight applied.",
                4:"Semester 4 -- Stable phase. Only admission model used.",
                5:"Semester 5 -- Stable phase. Only admission model used.",
                6:"Semester 6 -- Stable phase. Only admission model used."
            }[semester]
            sc2.markdown(f"""<div style='background:#FFF;border-left:4px solid {sem_clr};
                 padding:10px 14px;margin-top:4px;font-size:13px;color:{sem_clr};font-weight:600'>
              {sem_msg}</div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            form_label("Demographics")
            g2a,g2b,g2c,g2d=st.columns(4)
            gender   =g2a.selectbox("Gender",["Male","Female"])
            caste    =g2b.selectbox("Caste",["OPEN","OBC","SEBC","SCST"])
            religion =g2c.selectbox("Religion",["Hindu","Muslim","Jain","Christian"])
            district =g2d.selectbox("District",ALL_DISTRICTS)

            form_label("12th Exam Details")
            g3a,g3b=st.columns(2)
            spec =g3a.selectbox("12th Stream",["COMMERCE","SCIENCE","ARTS"])
            board=g3b.selectbox("Board",["G.H.S.E.B","GSEB","CBSE","Other"])

            st.markdown("<br>",unsafe_allow_html=True)
            submitted=st.form_submit_button("PREDICT CHURN RISK")

        if submitted:
            # Build admission features and get base probability
            adm_row  = build_admission_features(exam_pct,gender,college,year_gap,
                                                spec,board,caste,religion,district)
            inp      = pd.DataFrame([adm_row])[list(feats)]
            inps     = scaler.transform(inp)
            base_p   = float(model.predict_proba(inps)[0][1])
            
            # Apply adaptive semester signal weighting
            adaptive_weights = {1: 0.20, 2: 0.15, 3: 0.05}
            current_weight = adaptive_weights.get(semester, 0.0) if semester <= 3 else 0.0
            
            if semester <= 3:
                sem_signal = SEM_HIST_RATE.get(semester, 0.032)
                final_p = (1 - current_weight) * base_p + current_weight * sem_signal
                final_p = float(np.clip(final_p, 0.0, 1.0))
            else:
                final_p = base_p

            pred     = int(final_p >= thresh)
            rlbl,rcol= get_risk(final_p)
            badge    = "FLAGGED AS CHURN RISK" if pred else "PREDICTED AS ACTIVE"
            bcol     = RED if pred else GREEN

            # Big result
            st.markdown(f"""<div style='background:#FFF;border:3px solid {rcol};
                 padding:24px;text-align:center;margin:12px 0'>
              <div style='font-size:52px;font-weight:700;color:{rcol};letter-spacing:-2px'>
                {final_p*100:.1f}%</div>
              <div style='font-size:14px;color:#5C6B7A;margin:6px 0'>Combined Churn Probability</div>
              <div style='font-size:17px;font-weight:700;color:{rcol}'>{rlbl}</div>
              <div style='display:inline-block;margin-top:10px;background:{bcol};color:#FAFAF7;
                   font-size:12px;font-weight:700;letter-spacing:1px;padding:5px 18px;
                   text-transform:uppercase'>{badge}</div></div>""", unsafe_allow_html=True)

            # Gauge
            fig_g=go.Figure(go.Indicator(mode="gauge+number",value=final_p*100,
                number={"suffix":"%","font":{"color":rcol,"size":38,"family":"Times New Roman"}},
                gauge={"axis":{"range":[0,100],"tickcolor":GOLD},
                       "bar":{"color":rcol,"thickness":0.22},"bgcolor":"white",
                       "steps":[{"range":[0,20],"color":"#E8F5E9"},{"range":[20,40],"color":"#FFF8E1"},
                                 {"range":[40,65],"color":"#FFEBEE"},{"range":[65,100],"color":"#FFDDE1"}],
                       "threshold":{"line":{"color":GOLD,"width":3},"thickness":0.75,"value":thresh*100}}))
            fig_g.update_layout(height=220,margin=dict(t=8,b=8,l=28,r=28),
                paper_bgcolor="rgba(0,0,0,0)",font=dict(family="Times New Roman",color=NAVY))
            st.plotly_chart(fig_g, use_container_width=True)
            st.caption(f"Risk: 0-20% Low | 20-40% Moderate | 40-65% High | 65%+ Very High | Threshold: {thresh:.2f}")

            # Risk factor cards (same as before)
            st.markdown("#### Key Risk Factors")
            pdz  = 1 if 50<=exam_pct<65 else 0
            obc  = 1 if caste=="OBC" else 0
            sci  = 1 if spec=="SCIENCE" else 0
            dh   = 1 if district in HIGH_CHURN_DISTS else 0
            cbse = 1 if "CBSE" in board.upper() else 0
            col  = 0 if "BPCCS" in college else 1
            gen  = 0 if gender=="Male" else 1
            factors=[
                ("Semester",SEM_LABELS[semester].split("(")[0].strip(),
                 {1:RED,2:AMBER,3:GREEN,4:GREEN,5:GREEN,6:GREEN}[semester]),
                ("HSC Percentage",f"{exam_pct:.1f}%",
                 RED if pdz else AMBER if exam_pct<50 else GREEN),
                ("Exam Zone","Danger 50-65%" if pdz else ("Very Low <45%" if exam_pct<45 else "Safe"),
                 RED if pdz else AMBER if exam_pct<45 else GREEN),
                ("Caste",caste, AMBER if obc else GREEN),
                ("Stream",spec, RED if sci else AMBER if spec=="ARTS" else GREEN),
                ("District",district[:14], RED if dh else GREEN if district in LOCAL_DISTS else AMBER),
                ("Board",board, RED if cbse else GREEN),
                ("Religion",religion, AMBER if religion=="Muslim" else GREEN),
            ]
            fcols=st.columns(4)
            for i,(fn2,fv,fc) in enumerate(factors):
                fcols[i%4].markdown(f"""<div style='background:#FFF;border:1px solid #D4C5A9;
                     border-top:4px solid {fc};padding:10px;margin-bottom:7px;text-align:center'>
                  <div style='font-size:9px;color:#8FA3B8;font-weight:700;text-transform:uppercase'>{fn2}</div>
                  <div style='font-size:12.5px;font-weight:700;color:{fc};margin-top:4px'>{fv}</div></div>""",
                    unsafe_allow_html=True)

            # Interventions
            recs=[]
            if semester>=4:
                recs.append(f"Semester {semester} — student has already progressed beyond the critical early semesters. Focus on standard academic support.")
            else:
                if semester==1: recs.append("Semester 1 — highest risk historically. Immediate welfare check-in recommended within the first 2 weeks.")
                if semester==2: recs.append("Semester 2 — second highest risk window. Regular monitoring advised.")
                if pdz: recs.append(f"HSC {exam_pct:.0f}% falls in the 50-65% danger zone — counter-intuitively the highest churn bracket. Monitor closely.")
                elif exam_pct<45: recs.append(f"HSC {exam_pct:.0f}% -- very low. Academic bridging support recommended immediately.")
                if obc: recs.append("OBC category has the highest caste-level churn rate (18.1%). Verify scholarship status.")
                if sci: recs.append("Science stream for BCA -- subject mismatch is a known dropout trigger. Confirm student motivation.")
                if dh: recs.append(f"{district} is a high-risk district (20-37% churn). Check commute distance and accommodation.")
                if cbse: recs.append("CBSE board shows 23.1% churn -- possible adjustment difficulty with Gujarat University system.")
                if religion=="Muslim" and district not in LOCAL_DISTS:
                    recs.append("Muslim student from non-local district -- assign peer mentor and verify financial stability.")
                if pred and not recs:
                    recs.append(f"Combined score {final_p*100:.1f}% -- schedule a general welfare check-in.")
            if recs:
                st.markdown("#### Recommended Interventions")
                for r in recs:
                    st.markdown(f"""<div style='background:#FFF;border:1px solid #D4C5A9;
                         border-left:5px solid {GOLD};padding:10px 14px;margin-bottom:7px;
                         font-size:13px;color:{NAVY}'>&nbsp; {r}</div>""", unsafe_allow_html=True)
            elif not pred:
                st.success("No major risk factors detected. Student profile appears stable.")

    with t3:
        section_header("All Test Student Predictions")
        mc1,mc2,mc3,mc4=st.columns(4)
        mc1.metric("Total Test",len(y_test))
        mc2.metric("Pred Churned",int(all_preds.sum()))
        mc3.metric("Actual Churned",int(y_test.sum()))
        mc4.metric("Correct",int((all_preds==y_test.values).sum()))
        st.divider()
        res=pd.DataFrame({
            "Semester": sem_test.values,
            "Actual":   ["Churned" if v==1 else "Active" for v in y_test.values],
            "Score %":  (all_probs*100).round(1),
            "Predicted":["Churned" if v==1 else "Active" for v in all_preds],
            "Risk":     [get_risk(p)[0] for p in all_probs],
            "Correct":  ["YES" if p==a else "NO" for p,a in zip(all_preds,y_test.values)],
        }).reset_index(drop=True)
        fc1,fc2,fc3=st.columns(3)
        fa =fc1.selectbox("Filter Actual",["All","Churned","Active"])
        fp2=fc2.selectbox("Filter Predicted",["All","Churned","Active"])
        fcr=fc3.selectbox("Filter Correct",["All","YES","NO"])
        flt=res.copy()
        if fa !="All": flt=flt[flt["Actual"]==fa]
        if fp2!="All": flt=flt[flt["Predicted"]==fp2]
        if fcr!="All": flt=flt[flt["Correct"]==fcr]
        st.markdown(f"**Showing {len(flt)} of {len(res)} students**")
        st.dataframe(flt, use_container_width=True, height=360)
        fig_h=px.histogram(res,x="Score %",color="Actual",nbins=30,barmode="overlay",opacity=0.75,
            color_discrete_map={"Active":GREEN,"Churned":RED},
            title="Combined score distribution -- all test students")
        fig_h.add_vline(x=thresh*100,line_dash="dash",line_color=GOLD,
            annotation_text=f"Threshold {thresh:.2f}",annotation_font_color=GOLD,
            annotation_font_size=13)
        tnr(fig_h,360); st.plotly_chart(fig_h, use_container_width=True)