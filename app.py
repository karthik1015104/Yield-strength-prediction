"""
MPEA Yield Strength Predictor — Streamlit App
==============================================
Run with:  streamlit run app.py
Requires:  YS_model_v2.pkl and feature_columns.csv in the same directory
"""

import re
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MPEA Yield Strength Predictor",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .main { background-color: #0f1117; }
    .block-container { padding-top: 2rem; max-width: 1100px; }

    /* Hero */
    .hero {
        background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 60%, #0f1117 100%);
        border: 1px solid #2a3550;
        border-radius: 12px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle at 70% 50%, rgba(56,182,255,0.05) 0%, transparent 60%);
        pointer-events: none;
    }
    .hero h1 {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.9rem;
        font-weight: 600;
        color: #e8eaf0;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.5px;
    }
    .hero p {
        color: #7a8499;
        font-size: 0.95rem;
        margin: 0;
        font-weight: 300;
    }
    .hero .badge {
        display: inline-block;
        background: rgba(56,182,255,0.12);
        border: 1px solid rgba(56,182,255,0.3);
        color: #38b6ff;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        padding: 2px 10px;
        border-radius: 20px;
        margin-right: 6px;
        margin-bottom: 10px;
    }

    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #141928, #1a2235);
        border: 1px solid #2a3a5c;
        border-left: 4px solid #38b6ff;
        border-radius: 10px;
        padding: 1.8rem 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 3.2rem;
        font-weight: 600;
        color: #38b6ff;
        line-height: 1;
        margin: 0.4rem 0;
    }
    .result-label {
        font-size: 0.85rem;
        color: #7a8499;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
    }
    .result-context {
        font-size: 0.88rem;
        color: #9aa3b8;
        margin-top: 0.6rem;
    }

    /* Metric cards */
    .metric-row { display: flex; gap: 1rem; margin: 1rem 0; }
    .metric-card {
        flex: 1;
        background: #141928;
        border: 1px solid #2a3550;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-card .val {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.4rem;
        font-weight: 600;
        color: #e8eaf0;
    }
    .metric-card .lbl {
        font-size: 0.75rem;
        color: #5a6480;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 2px;
    }

    /* Section headers */
    .section-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: #38b6ff;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin: 1.8rem 0 0.8rem 0;
        padding-bottom: 6px;
        border-bottom: 1px solid #1e2840;
    }

    /* Strength indicator */
    .strength-bar-wrap { margin: 1rem 0; }
    .strength-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.82rem;
        color: #9aa3b8;
        display: flex;
        justify-content: space-between;
        margin-bottom: 4px;
    }

    /* Info box */
    .info-box {
        background: rgba(56,182,255,0.06);
        border: 1px solid rgba(56,182,255,0.18);
        border-radius: 8px;
        padding: 1rem 1.2rem;
        font-size: 0.88rem;
        color: #9aa3b8;
        margin: 0.8rem 0;
    }
    .info-box strong { color: #38b6ff; }

    /* Stmetric override */
    [data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace !important;
        color: #e8eaf0 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0a0d14 !important;
        border-right: 1px solid #1e2840;
    }
    [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ── Element & thermodynamic data ──────────────────────────────────────────────
ELEM_PROPS = {
    'Al': [3,143,1.61,933],   'Co': [9,125,1.88,1768],  'Cr': [6,128,1.66,2180],
    'Cu': [11,128,1.90,1358], 'Fe': [8,126,1.83,1811],   'Hf': [4,159,1.30,2506],
    'Mn': [7,127,1.55,1519],  'Mo': [6,139,2.16,2896],   'Nb': [5,146,1.60,2750],
    'Ni': [10,124,1.91,1728], 'Re': [7,137,1.90,3459],   'Si': [4,111,1.90,1687],
    'Ta': [5,146,1.50,3290],  'Ti': [4,147,1.54,1941],   'V':  [5,134,1.63,2183],
    'W':  [6,139,2.36,3695],  'Zr': [4,160,1.33,2128],   'B':  [3,87,2.04,2349],
    'C':  [4,77,2.55,3823],   'Ca': [2,197,1.00,1115],   'Ga': [3,122,1.81,303],
    'Li': [1,152,0.98,454],   'Mg': [2,160,1.31,923],    'Nd': [3,181,1.14,1297],
    'Pd': [10,137,2.20,1828], 'Sc': [3,162,1.36,1814],   'Sn': [4,140,1.96,505],
    'Ag': [11,144,1.93,1235], 'Y':  [3,180,1.22,1799],   'Zn': [12,134,1.65,693],
}
ALL_ELEMENTS = sorted(ELEM_PROPS.keys())

MIX_H = {
    ('Al','Co'):-19,('Al','Cr'):-10,('Al','Cu'):-1, ('Al','Fe'):-11,('Al','Hf'):-39,
    ('Al','Mn'):-19,('Al','Mo'):-2, ('Al','Nb'):-18,('Al','Ni'):-22,('Al','Si'):-19,
    ('Al','Ta'):-19,('Al','Ti'):-30,('Al','V'):-16, ('Al','W'):-2,  ('Al','Zr'):-44,
    ('Co','Cr'):-4, ('Co','Cu'):6,  ('Co','Fe'):-1, ('Co','Mn'):-5, ('Co','Mo'):-5,
    ('Co','Nb'):-25,('Co','Ni'):0,  ('Co','Ta'):-24,('Co','Ti'):-28,('Co','V'):-14,
    ('Co','W'):-1,  ('Co','Zr'):-41,('Cr','Cu'):12, ('Cr','Fe'):-1, ('Cr','Mn'):2,
    ('Cr','Mo'):0,  ('Cr','Nb'):-7, ('Cr','Ni'):-7, ('Cr','Ta'):-7, ('Cr','Ti'):-7,
    ('Cr','V'):-2,  ('Cr','W'):0,   ('Cr','Zr'):-12,('Cu','Fe'):13, ('Cu','Mn'):4,
    ('Cu','Mo'):19, ('Cu','Nb'):3,  ('Cu','Ni'):4,  ('Cu','Ti'):-9, ('Cu','V'):5,
    ('Cu','Zr'):-23,('Fe','Mn'):0,  ('Fe','Mo'):-2, ('Fe','Nb'):-16,('Fe','Ni'):-2,
    ('Fe','Ta'):-15,('Fe','Ti'):-17,('Fe','V'):-7,  ('Fe','W'):0,   ('Fe','Zr'):-25,
    ('Mn','Mo'):-5, ('Mn','Nb'):-12,('Mn','Ni'):-8, ('Mn','Ti'):-20,('Mn','V'):-1,
    ('Mn','Zr'):-28,('Mo','Nb'):-6, ('Mo','Ni'):-7, ('Mo','Ta'):-5, ('Mo','Ti'):-4,
    ('Mo','V'):-1,  ('Mo','W'):0,   ('Mo','Zr'):-6, ('Nb','Ni'):-30,('Nb','Ta'):0,
    ('Nb','Ti'):2,  ('Nb','V'):-1,  ('Nb','W'):-9,  ('Nb','Zr'):4,
    ('Ni','Si'):-13,('Ni','Ta'):-29,('Ni','Ti'):-35,('Ni','V'):-18, ('Ni','W'):-3,
    ('Ni','Zr'):-49,('Si','Ti'):-28,('Ta','Ti'):-9, ('Ta','V'):-1,  ('Ta','W'):-7,
    ('Ta','Zr'):-2, ('Ti','V'):-2,  ('Ti','W'):-7,  ('Ti','Zr'):0,  ('V','W'):-1,
    ('V','Zr'):-4,  ('W','Zr'):-9,
}


def parse_formula(formula):
    matches = re.findall(r'([A-Z][a-z]?)([\d.]+)', str(formula))
    if not matches:
        return {}
    comp = {el: float(amt) for el, amt in matches}
    total = sum(comp.values())
    return {el: amt / total for el, amt in comp.items()}


def compute_features(formula):
    comp = parse_formula(formula)
    if not comp:
        return None
    elements = list(comp.keys())
    fracs    = list(comp.values())
    vec_vals = [ELEM_PROPS.get(el,[0,0,0,0])[0]    for el in elements]
    r_vals   = [ELEM_PROPS.get(el,[0,130,0,0])[1]  for el in elements]
    en_vals  = [ELEM_PROPS.get(el,[0,0,1.8,0])[2]  for el in elements]
    mp_vals  = [ELEM_PROPS.get(el,[0,0,0,1500])[3] for el in elements]
    VEC      = sum(f*v for f,v in zip(fracs,vec_vals))
    r_bar    = sum(f*r for f,r in zip(fracs,r_vals))
    delta_r  = np.sqrt(sum(f*(1-r/r_bar)**2 for f,r in zip(fracs,r_vals)))*100
    en_bar   = sum(f*e for f,e in zip(fracs,en_vals))
    delta_en = np.sqrt(sum(f*(e-en_bar)**2 for f,e in zip(fracs,en_vals)))
    S_mix    = -8.314*sum(f*np.log(f) for f in fracs if f>0)
    H_mix    = sum(4*fracs[i]*fracs[j]*MIX_H.get(
                   (min(elements[i],elements[j]),max(elements[i],elements[j])),0)
                   for i in range(len(elements)) for j in range(i+1,len(elements)))
    T_melt   = sum(f*m for f,m in zip(fracs,mp_vals))
    omega    = T_melt*S_mix/(abs(H_mix)*1000+1e-9)
    return {
        'num_elements': len(elements), 'VEC': VEC, 'delta_r': delta_r,
        'delta_en': delta_en, 'S_mix': S_mix, 'H_mix': H_mix,
        'T_melt': T_melt, 'omega': omega,
        **{f'x_{el}': comp.get(el,0.0) for el in ALL_ELEMENTS},
    }


def simplify_microstructure(m):
    if pd.isna(m) or m == 'Unknown': return 'Unknown'
    m = str(m)
    if m == 'BCC':  return 'BCC_single'
    if m == 'FCC':  return 'FCC_single'
    if 'FCC' in m and 'BCC' not in m and 'B2' not in m: return 'FCC_based'
    if 'BCC' in m and 'FCC' not in m: return 'BCC_based'
    return 'Mixed'


@st.cache_resource
def load_model():
    model     = joblib.load('YS_model_v2.pkl')
    feat_cols = pd.read_csv('feature_columns.csv', header=None)[0].tolist()
    return model, feat_cols


def build_input_row(feat, test_temp, test_type, micro, proc, grain_size):
    feat['test_temperature']      = test_temp
    feat['test_type_compression'] = 1 if test_type == 'Compression' else 0
    micro_group = simplify_microstructure(micro)
    for g in ['BCC_based','BCC_single','FCC_based','FCC_single','Mixed','Unknown']:
        feat[f'micro_{g}'] = 1 if micro_group == g else 0
    proc_clean = proc if proc in ['CAST','WROUGHT','POWDER','ANNEAL'] else 'OTHER'
    for p in ['ANNEAL','CAST','OTHER','POWDER','WROUGHT']:
        feat[f'proc_{p}'] = 1 if proc_clean == p else 0
    feat['grain_size_known'] = 0 if grain_size is None else 1
    feat['grain_size_log']   = np.log1p(grain_size) if grain_size else 0.0
    return feat


def ys_category(ys):
    if ys < 200:   return "Low", "#6ab0de"
    if ys < 500:   return "Moderate", "#5cb85c"
    if ys < 1000:  return "High", "#f0a500"
    if ys < 2000:  return "Very High", "#e07b39"
    return "Exceptional", "#e05c5c"


# ── Load model ────────────────────────────────────────────────────────────────
try:
    model, feat_cols = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error   = str(e)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='font-family: IBM Plex Mono, monospace; font-size:0.7rem;
                color:#38b6ff; letter-spacing:3px; text-transform:uppercase;
                margin-bottom:1.2rem; border-bottom:1px solid #1e2840; padding-bottom:8px;'>
        Model Info
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:0.83rem; color:#7a8499; line-height:1.8;'>
    <b style='color:#c8cfe0;'>Algorithm</b><br>XGBoost Regressor<br><br>
    <b style='color:#c8cfe0;'>Dataset</b><br>1,067 MPEA samples<br><br>
    <b style='color:#c8cfe0;'>Features</b><br>53 (thermo + composition + process)<br><br>
    <b style='color:#c8cfe0;'>Test R²</b><br><span style='color:#38b6ff; font-family:IBM Plex Mono,monospace;'>0.811</span><br><br>
    <b style='color:#c8cfe0;'>Test MAE</b><br><span style='color:#38b6ff; font-family:IBM Plex Mono,monospace;'>165 MPa</span><br><br>
    <b style='color:#c8cfe0;'>YS Range (train)</b><br>24 – 3,416 MPa
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family: IBM Plex Mono, monospace; font-size:0.7rem;
                color:#38b6ff; letter-spacing:3px; text-transform:uppercase;
                margin-bottom:0.8rem; border-bottom:1px solid #1e2840; padding-bottom:8px;'>
        Formula Guide
    </div>
    <div style='font-size:0.8rem; color:#7a8499; line-height:1.9; font-family: IBM Plex Mono, monospace;'>
    Al0.5 Co1 Cr1 Fe1 Ni1<br>
    Al1 Co1 Cr1 Fe1 Ni1<br>
    Ti1 Nb0.5 Mo0.5 V0.5<br>
    Cr1 Fe1 Ni1 Al0.3 Cu0.3
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.75rem; color:#3a4560; line-height:1.6;'>
    Built by Karthik · IIT Madras<br>
    MS AI/ML applicant · Fall 2028
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='hero'>
    <span class='badge'>XGBoost</span>
    <span class='badge'>Materials Informatics</span>
    <span class='badge'>MPEA</span>
    <h1>⚗️ MPEA Yield Strength Predictor</h1>
    <p>Predict the yield strength of Multi-Principal Element Alloys from elemental composition,
    microstructure, and processing conditions — powered by thermodynamic feature engineering
    and gradient boosting.</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"Could not load model: {load_error}\n\nMake sure `YS_model_v2.pkl` and `feature_columns.csv` are in the same directory as `app.py`.")
    st.stop()

# ── Input Section ─────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Input Conditions</div>", unsafe_allow_html=True)

col1, col2 = st.columns([1.2, 1])

with col1:
    formula_input = st.text_input(
        "Alloy Formula",
        value="Al0.5 Co1 Cr1 Fe1 Ni1",
        help="Space-separated element-amount pairs. Amounts are molar ratios (normalised automatically).",
        placeholder="e.g. Al0.5 Co1 Cr1 Fe1 Ni1"
    )

    test_temp = st.slider(
        "Test Temperature (°C)",
        min_value=-270, max_value=1600,
        value=25, step=25,
        help="Room temperature = 25°C"
    )

    test_type = st.radio(
        "Test Type",
        options=["Compression", "Tension"],
        horizontal=True
    )

with col2:
    microstructure = st.selectbox(
        "Crystal Structure / Microstructure",
        options=[
            "BCC", "FCC", "BCC+Sec.", "FCC+Sec.", "FCC+BCC",
            "BCC+B2", "BCC+Laves", "FCC+B2", "Other", "Unknown"
        ],
        index=0
    )

    processing = st.selectbox(
        "Processing Method",
        options=["CAST", "WROUGHT", "ANNEAL", "POWDER", "OTHER"],
        index=0
    )

    grain_size_known = st.checkbox("Grain size is known?", value=False)
    grain_size = None
    if grain_size_known:
        grain_size = st.number_input(
            "Grain Size (µm)",
            min_value=0.01, max_value=5000.0,
            value=17.1, step=0.1
        )

# ── Predict Button ─────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("⚡  Predict Yield Strength", type="primary", use_container_width=True)

if predict_btn:
    feat = compute_features(formula_input.strip())

    if feat is None or len(parse_formula(formula_input.strip())) == 0:
        st.error("Could not parse formula. Use format: `Al0.5 Co1 Cr1 Fe1 Ni1`")
    else:
        row = build_input_row(feat, test_temp, test_type, microstructure, processing, grain_size)
        X_input = pd.DataFrame([row])[feat_cols]
        ys_pred  = model.predict(X_input)[0]
        ys_pred  = max(0, ys_pred)

        category, cat_color = ys_category(ys_pred)

        # ── Result card ──
        st.markdown(f"""
        <div class='result-card'>
            <div class='result-label'>Predicted Yield Strength</div>
            <div class='result-value' style='color:{cat_color};'>{ys_pred:.0f}</div>
            <div style='font-family:IBM Plex Mono,monospace; font-size:1rem;
                        color:#7a8499; margin-top:2px;'>MPa</div>
            <div class='result-context'>
                Strength class: <strong style='color:{cat_color};'>{category}</strong>
                &nbsp;·&nbsp; ±165 MPa model MAE
                &nbsp;·&nbsp; {test_temp}°C · {test_type}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Thermodynamic descriptors ──
        comp = parse_formula(formula_input.strip())
        elements = list(comp.keys()); fracs = list(comp.values())
        r_vals  = [ELEM_PROPS.get(el,[0,130,0,0])[1] for el in elements]
        en_vals = [ELEM_PROPS.get(el,[0,0,1.8,0])[2] for el in elements]
        mp_vals = [ELEM_PROPS.get(el,[0,0,0,1500])[3] for el in elements]
        r_bar   = sum(f*r for f,r in zip(fracs,r_vals))

        VEC      = feat['VEC']
        delta_r  = feat['delta_r']
        S_mix    = feat['S_mix']
        H_mix    = feat['H_mix']
        T_melt   = feat['T_melt']
        omega    = feat['omega']

        st.markdown("<div class='section-header'>Thermodynamic Descriptors</div>", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("VEC",      f"{VEC:.2f}",     help="Valence Electron Concentration — <6.87 favours BCC, >8 favours FCC")
        c2.metric("δ (%)",    f"{delta_r:.2f}", help="Atomic size mismatch — drives solid solution strengthening")
        c3.metric("ΔS_mix",   f"{S_mix:.2f}",  help="Ideal mixing entropy (J/mol·K) — higher = more disordered")
        c4.metric("ΔH_mix",   f"{H_mix:.2f}",  help="Mixing enthalpy (kJ/mol) — negative = thermodynamically stable")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("T_melt (K)", f"{T_melt:.0f}", help="Weighted average melting point")
        c6.metric("Ω (omega)",  f"{omega:.2f}",  help="Yang's Ω = T_melt·ΔS / |ΔH| — Ω>1.1 indicates single-phase solid solution")
        c7.metric("# Elements", f"{feat['num_elements']}")
        c8.metric("Δχ (EN)",    f"{feat['delta_en']:.3f}", help="Electronegativity difference")

        # ── Phase stability guidance ──
        if VEC >= 8:
            phase_hint = "🔵 VEC ≥ 8 → FCC phase typically stable"
        elif VEC <= 6.87:
            phase_hint = "🟠 VEC ≤ 6.87 → BCC phase typically stable"
        else:
            phase_hint = "🟢 6.87 < VEC < 8 → mixed BCC+FCC possible"

        omega_hint = "✅ Ω > 1.1 → single-phase solid solution likely" if omega > 1.1 else "⚠️ Ω ≤ 1.1 → intermetallic/multi-phase possible"

        st.markdown(f"""
        <div class='info-box'>
            <strong>Phase Stability Indicators</strong><br>
            {phase_hint}<br>
            {omega_hint}<br>
            δ = {delta_r:.2f}% — {"high lattice distortion (strengthening effect)" if delta_r > 5 else "moderate lattice distortion"}
        </div>
        """, unsafe_allow_html=True)

        # ── Composition radar chart ──
        st.markdown("<div class='section-header'>Composition Breakdown</div>", unsafe_allow_html=True)

        comp_df = pd.DataFrame({'Element': list(comp.keys()),
                                 'Mole Fraction': list(comp.values())}).sort_values('Mole Fraction', ascending=True)

        fig = go.Figure(go.Bar(
            x=comp_df['Mole Fraction'],
            y=comp_df['Element'],
            orientation='h',
            marker=dict(
                color=comp_df['Mole Fraction'],
                colorscale=[[0,'#1a2a45'],[0.5,'#2a6496'],[1,'#38b6ff']],
                line=dict(color='rgba(0,0,0,0)', width=0)
            ),
            text=[f'{v:.3f}' for v in comp_df['Mole Fraction']],
            textposition='outside',
            textfont=dict(family='IBM Plex Mono', size=11, color='#9aa3b8')
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#0f1117',
            font=dict(family='IBM Plex Sans', color='#9aa3b8'),
            xaxis=dict(showgrid=False, zeroline=False, title='Mole Fraction',
                       color='#4a5570'),
            yaxis=dict(showgrid=False, zeroline=False, color='#9aa3b8'),
            margin=dict(l=10, r=60, t=10, b=30),
            height=max(180, len(comp)*45),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Temperature effect ──
        st.markdown("<div class='section-header'>Temperature Sensitivity</div>", unsafe_allow_html=True)

        temps  = [-196, 25, 200, 400, 600, 800, 1000, 1200]
        preds  = []
        for t in temps:
            f2 = compute_features(formula_input.strip())
            r2 = build_input_row(f2, t, test_type, microstructure, processing, grain_size)
            xi = pd.DataFrame([r2])[feat_cols]
            preds.append(max(0, model.predict(xi)[0]))

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=temps, y=preds,
            mode='lines+markers',
            line=dict(color='#38b6ff', width=2.5),
            marker=dict(size=7, color='#38b6ff',
                        line=dict(color='#0f1117', width=2)),
            fill='tozeroy',
            fillcolor='rgba(56,182,255,0.07)',
        ))
        fig2.add_vline(x=test_temp, line=dict(color='#f0a500', width=1.5, dash='dot'))
        fig2.add_annotation(x=test_temp, y=max(preds)*0.95,
                            text=f"  Selected: {test_temp}°C",
                            showarrow=False,
                            font=dict(color='#f0a500', size=11, family='IBM Plex Mono'))
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#0f1117',
            font=dict(family='IBM Plex Sans', color='#9aa3b8'),
            xaxis=dict(title='Test Temperature (°C)', color='#4a5570',
                       showgrid=True, gridcolor='#1a2030'),
            yaxis=dict(title='Predicted YS (MPa)', color='#4a5570',
                       showgrid=True, gridcolor='#1a2030'),
            margin=dict(l=10, r=10, t=10, b=40),
            height=260,
        )
        st.plotly_chart(fig2, use_container_width=True)


# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; font-size:0.75rem; color:#2a3550;
            font-family: IBM Plex Mono, monospace; border-top: 1px solid #1a2030;
            padding-top: 1rem;'>
    MPEA Yield Strength Predictor · XGBoost · Test R² = 0.811 · IIT Madras · 2024
</div>
""", unsafe_allow_html=True)
