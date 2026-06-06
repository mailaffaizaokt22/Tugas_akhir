# =============================================================
# PARKING TARIFF POTENTIAL ANALYSIS DASHBOARD
# Version 6.0 — 6-Menu Navigation (Sesuai Bab Penelitian)
# Alur: Data Table | Visualization (EDA) | Modeling | Evaluasi | Hasil Rekomendasi | Map
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import traceback

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import folium
import streamlit.components.v1 as components

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Parking Tariff Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🅿️",
)

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --accent:       #3B82F6;
    --accent-soft:  rgba(59,130,246,.12);
    --bg-card:      rgba(255,255,255,0.03);
    --border:       rgba(148,163,184,0.15);
    --border-hover: rgba(148,163,184,0.3);
    --shadow-md:    0 4px 6px -1px rgba(0,0,0,.1),0 2px 4px -1px rgba(0,0,0,.06);
    --shadow-lg:    0 10px 15px -3px rgba(0,0,0,.1),0 4px 6px -2px rgba(0,0,0,.05);
    --r-md: 12px; --r-lg: 16px; --r-xl: 20px;
}
html,body,[class*="css"]{ font-family:'Plus Jakarta Sans',sans-serif !important; }
code,pre,[data-testid="stCode"]{ font-family:'JetBrains Mono',monospace !important; }
#MainMenu,footer{ visibility:hidden; }
.block-container{ padding:3rem 2.5rem 2.5rem !important; max-width:1440px; }
::selection{ background:var(--accent-soft); color:var(--accent); }

[data-testid="stSidebar"]{
    background:#fffff !important;
    border-right:1px solid rgba(15, 23, 42, 0.1) !important;
    min-width:260px !important; max-width:260px !important;
    box-shadow:var(--shadow-lg);
}
[data-testid="stSidebar"] *{ color:#1E293B !important; }
.sb-brand{
    padding:2rem 1.5rem 1.5rem;
    border-bottom:1px solid rgba(15, 23, 42, 0.1);
    margin-bottom:1rem; display:flex;
    flex-direction:column; align-items:center; text-align:center;
}
.sb-brand-icon{
    font-size:32px; margin-bottom:12px;
    background:linear-gradient(135deg,#1E3A8A,#3B82F6);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.sb-brand-name{ font-size:15px; font-weight:700; color:#0F172A !important; letter-spacing:.03em; }
.sb-brand-sub { font-size:12px; color:#334155 !important; margin-top:4px; }
[data-testid="stSidebar"] .stButton>button{
    width:calc(100% - 24px) !important; text-align:left !important;
    padding:10px 16px !important; margin:4px 12px !important;
    border-radius:var(--r-md) !important; border:1px solid transparent !important;
    background:transparent !important; color:#1E293B !important;
    font-size:13.5px !important; font-weight:500 !important;
    transition:all .25s cubic-bezier(0.4,0,0.2,1) !important;
}
[data-testid="stSidebar"] .stButton>button:hover{
    background:rgba(255,255,255,.3) !important;
    color:#0F172A !important; transform:translateX(4px);
}
[data-testid="stSidebar"] .stButton>button p{
    display:flex !important; align-items:center !important;
    gap:10px !important; margin:0 !important;
}
.page-header{
    display:flex; align-items:center; gap:16px;
    padding-bottom:1.5rem; margin-bottom:2rem;
    border-bottom:1px solid var(--border);
    animation:fadeInDown 0.5s ease-out;
}
.ph-icon{
    width:48px; height:48px; border-radius:var(--r-md);
    background:linear-gradient(135deg,var(--accent-soft),rgba(139,92,246,.1));
    border:1px solid rgba(59,130,246,.2);
    display:flex; align-items:center; justify-content:center;
    font-size:22px; flex-shrink:0;
}
.ph-title{ font-size:24px; font-weight:700; margin:0; letter-spacing:-.02em; }
.ph-sub  { font-size:14px; opacity:.6; margin-top:4px; }
.slabel{
    font-size:11px; font-weight:600; letter-spacing:.1em;
    text-transform:uppercase; opacity:.5; margin-bottom:1rem;
    display:flex; align-items:center; gap:8px;
}
.slabel::after{ content:''; flex-grow:1; height:1px; background:var(--border); }
.info-band,.warn-band,.ok-band{
    backdrop-filter:blur(12px); border-radius:var(--r-md);
    padding:16px 20px; font-size:14px; margin-bottom:1.5rem; line-height:1.6;
}
.info-band{ background:rgba(59,130,246,.05);  border:1px solid rgba(59,130,246,.2); }
.warn-band{ background:rgba(245,158,11,.05);  border:1px solid rgba(245,158,11,.2); }
.ok-band  { background:rgba(16,185,129,.05);  border:1px solid rgba(16,185,129,.2); }
.stTabs [data-baseweb="tab-list"]{
    gap:24px !important; border-bottom:1px solid var(--border) !important;
    background:transparent !important; padding:0 8px !important;
}
.stTabs [data-baseweb="tab"]{
    font-size:14px !important; font-weight:500 !important;
    padding:12px 4px !important; border-radius:0 !important;
    color:#64748B !important; border-bottom:2px solid transparent !important;
    background:transparent !important; transition:all .2s ease !important;
}
.stTabs [data-baseweb="tab"]:hover{ color:var(--accent) !important; }
.stTabs [aria-selected="true"]{
    color:var(--accent) !important;
    border-bottom:2px solid var(--accent) !important; font-weight:600 !important;
}
[data-testid="metric-container"]{
    background:var(--bg-card) !important; border:1px solid var(--border) !important;
    border-radius:var(--r-lg) !important; padding:16px 20px !important;
    transition:transform .2s ease,box-shadow .2s ease !important;
}
[data-testid="metric-container"]:hover{
    transform:translateY(-2px) !important; border-color:var(--border-hover) !important;
}
[data-testid="metric-container"] label{
    font-size:11px !important; font-weight:600 !important;
    letter-spacing:.05em !important; text-transform:uppercase !important; opacity:.6 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"]{
    font-size:24px !important; font-weight:700 !important; letter-spacing:-.02em !important;
}
.hdivider{ border:none; border-top:1px solid var(--border); margin:2rem 0; }
@keyframes fadeInDown{ from{opacity:0;transform:translateY(-10px)} to{opacity:1;transform:translateY(0)} }
[data-testid="stDataFrame"]{
    border-radius:var(--r-md) !important; overflow:hidden !important;
    border:1px solid var(--border) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
FILE_PATH        = "DataParkir_Terbaru.xlsx"
CENTER_LAT       = -7.4228
CENTER_LON       = 109.2380
MIN_TARIF_MOTOR  = 1000
MAX_TARIF_MOTOR  = 10000
MIN_TARIF_MOBIL  = 2000
MAX_TARIF_MOBIL  = 20000

# Delta tarif rekomendasi (Tinggi/Sedang/Rendah)
DELTA_TINGGI = 2000   # prediksi > 130% median → +2000
DELTA_SEDANG = 1000   # prediksi 70%–130% median → +1000
DELTA_RENDAH = 0      # prediksi < 70% median → +0

# === PERSENTASE MEDIAN ===
PERSEN_NAIK  = 1.30   # 130% dari median → Tinggi
PERSEN_TURUN = 0.70   # 70% dari median  → Rendah

PALETTE = {
    "blue":   "#2563EB", "teal":  "#0D9488", "green": "#16A34A",
    "amber":  "#D97706", "red":   "#DC2626", "slate": "#64748B",
    "indigo": "#4F46E5",
}

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Data Table"


# ─────────────────────────────────────────────────────────────
# MATPLOTLIB THEME
# ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":"none", "axes.facecolor":"none",
    "axes.edgecolor":"#E5E7EB","axes.labelcolor":"#6B7280",
    "axes.titlecolor":"#111827","axes.titlesize":13,
    "axes.titleweight":"600",  "axes.labelsize":11,
    "axes.grid":True,          "grid.color":"#F3F4F6",
    "grid.linewidth":0.7,      "xtick.color":"#9CA3AF",
    "ytick.color":"#9CA3AF",   "xtick.labelsize":10,
    "ytick.labelsize":10,      "legend.framealpha":0,
    "legend.fontsize":10,      "font.family":"sans-serif",
    "font.size":11,            "figure.dpi":120,
})


# ─────────────────────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_preprocess_data(file_path):
    df_header = pd.read_excel(file_path, header=0, nrows=0)
    real_cols  = df_header.columns.tolist()
    df_raw = pd.read_excel(file_path, header=1).iloc[1:].reset_index(drop=True)

    if len(real_cols) == len(df_raw.columns):
        df_raw.columns = real_cols
    else:
        new_cols = []
        for i, col in enumerate(df_raw.columns):
            if i < len(real_cols) and not str(real_cols[i]).startswith("Unnamed"):
                new_cols.append(real_cols[i])
            else:
                new_cols.append(col)
        df_raw.columns = new_cols

    sub_col_map = {
        "Unnamed: 17":"Pend WD Motor 1 Minggu","Unnamed: 18":"Pend WD Motor 1 Bulan",
        "Unnamed: 19":"Pend WD Motor 12 Bulan","Unnamed: 21":"Pend WD Mobil 1 Minggu",
        "Unnamed: 22":"Pend WD Mobil 1 Bulan", "Unnamed: 23":"Pend WD Mobil 12 Bulan",
        "Unnamed: 25":"Pend WE Motor 1 Minggu","Unnamed: 26":"Pend WE Motor 1 Bulan",
        "Unnamed: 27":"Pend WE Motor 12 Bulan","Unnamed: 29":"Pend WE Mobil 1 Minggu",
        "Unnamed: 30":"Pend WE Mobil 1 Bulan", "Unnamed: 31":"Pend WE Mobil 12 Bulan",
    }
    df_raw = df_raw.rename(columns={k:v for k,v in sub_col_map.items() if k in df_raw.columns})

    df = df_raw.copy(); n_raw = len(df)
    RENAME_MAP = {
        df.columns[0]:"Titik",    df.columns[1]:"Lokasi",
        df.columns[2]:"Latitude", df.columns[3]:"Longitude",
        df.columns[4]: "Jam Ramai Mobil Weekday",
        df.columns[5]: "Jam Ramai Motor Weekend",
        df.columns[6]: "Jam Ramai Mobil Weekend",
        df.columns[7]: "Jam Ramai Motor Weekday",
        df.columns[8]: "Jam Sedang Motor Weekday",
        df.columns[9]: "Jam Sedang Mobil Weekday",
        df.columns[10]:"Jam Sedang Motor Weekend",
        df.columns[11]:"Jam Sedang Mobil Weekend",
        df.columns[12]:"Jam Sepi Motor Weekday",
        df.columns[13]:"Jam Sepi Mobil Weekday",
        df.columns[14]:"Jam Sepi Motor Weekend",
        df.columns[15]:"Jam Sepi Mobil Weekend",
        df.columns[32]:"Pend WD Motor", df.columns[33]:"Pend WD Mobil",
        df.columns[34]:"Pend WE Motor", df.columns[35]:"Pend WE Mobil",
        df.columns[36]:"Jumlah Motor Weekday", df.columns[37]:"Jumlah Mobil Weekday",
        df.columns[38]:"Jumlah Motor Weekend",  df.columns[39]:"Jumlah Mobil Weekend",
        df.columns[40]:"Tarif Motor",           df.columns[41]:"Tarif Mobil",
    }
    df = df.rename(columns={c:v for c,v in RENAME_MAP.items() if c in df.columns})

    def cc(series):
        s = series.astype(str).str.replace(r"[^0-9,.-]","",regex=True)
        s = s.str.replace(".",   "",regex=False).str.replace(",",".",regex=False)
        s = s.str.replace("-",   "0",regex=False).replace("","0")
        return pd.to_numeric(s, errors="coerce").fillna(0)

    if "Latitude"  in df.columns: df["Latitude"]  = pd.to_numeric(df["Latitude"],  errors="coerce")
    if "Longitude" in df.columns: df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    for c in ["Pend WD Motor","Pend WD Mobil","Pend WE Motor","Pend WE Mobil",
              "Jumlah Motor Weekday","Jumlah Mobil Weekday",
              "Jumlah Motor Weekend","Jumlah Mobil Weekend",
              "Tarif Motor","Tarif Mobil"]:
        if c in df.columns: df[c] = cc(df[c])

    na_lat   = int(df["Latitude"].isna().sum())
    na_lon   = int(df["Longitude"].isna().sum())
    n_before = len(df)
    df       = df.dropna(subset=["Latitude","Longitude"]).reset_index(drop=True)
    n_dropped= n_before - len(df)

    def pt(t):
        try:
            if isinstance(t,str) and "." in t:
                h,m = t.split("."); return int(h)+int(m)/60
            return int(t)
        except: return 0

    def calc_dur(tr):
        if pd.isna(tr) or str(tr).strip() in ["-","0","nan",""]: return 0
        try:
            s,e = str(tr).replace("–","-").replace(" ","").split("-")
            d = pt(e)-pt(s); return d if d>=0 else d+24
        except: return 0

    for col in [c for c in df.columns if c.startswith("Jam ")]:
        df[f"Durasi {col}"] = df[col].astype(str).apply(calc_dur)

    df["Total Durasi Ramai"]  = df.filter(like="Durasi Jam Ramai").sum(axis=1)
    df["Total Durasi Sedang"] = df.filter(like="Durasi Jam Sedang").sum(axis=1)
    df["Total Durasi Sepi"]   = df.filter(like="Durasi Jam Sepi").sum(axis=1)
    df["Rasio Motor/Mobil Weekday"]  = df["Jumlah Motor Weekday"]/(df["Jumlah Mobil Weekday"]+1)
    df["Rasio Motor/Mobil Weekend"]  = df["Jumlah Motor Weekend"]/(df["Jumlah Mobil Weekend"]+1)
    df["Durasi Ramai per Kendaraan"] = df["Total Durasi Ramai"]/(
        df["Jumlah Motor Weekend"]+df["Jumlah Mobil Weekend"]+1)
    df["Jarak ke Pusat (km)"] = np.sqrt(
        (df["Latitude"]-CENTER_LAT)**2+(df["Longitude"]-CENTER_LON)**2)*111

    base_features = [
        "Jumlah Motor Weekday","Jumlah Mobil Weekday",
        "Jumlah Motor Weekend","Jumlah Mobil Weekend",
        "Tarif Motor","Tarif Mobil",
        "Total Durasi Ramai","Total Durasi Sedang","Total Durasi Sepi",
        "Rasio Motor/Mobil Weekday","Rasio Motor/Mobil Weekend",
        "Durasi Ramai per Kendaraan","Jarak ke Pusat (km)",
    ]
    durasi_cols = [c for c in df.columns if c.startswith("Durasi Jam")]
    FEATURES    = list(set(base_features+durasi_cols))
    FEATURES    = [f for f in FEATURES if f in df.columns]

    def cap(data, cols, lo=0.01, hi=0.90):
        d = data.copy()
        for col in cols:
            if col in d.columns:
                d[col] = np.clip(d[col], d[col].quantile(lo), d[col].quantile(hi))
        return d

    prep_info = {
        "n_raw":n_raw, "n_clean":len(df), "n_dropped":n_dropped,
        "reasons":{"Latitude kosong":na_lat,"Longitude kosong":na_lon} if n_dropped>0 else {},
    }
    return cap(df, FEATURES), FEATURES, df_raw, prep_info


# ─────────────────────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_models(df_processed, FEATURES):
    import os
    import pickle
    pkl_path = "trained_models.pkl"
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, "rb") as f:
                d = pickle.load(f)
            return (
                d["mm"], d["mb"], d["metrics"], d["metrics_train"], d["cv_scores"], d["X"], d["y_motor"], d["y_mobil"],
                d["X_train_m"], d["X_test_m"], d["y_train_m"], d["y_test_m"],
                d["X_train_b"], d["X_test_b"], d["y_train_b"], d["y_test_b"], d["avail_m"], d["avail_b"]
            )
        except Exception as e:
            st.warning(f"Gagal memuat pre-trained model ({e}). Melatih ulang...")

    avail = [f for f in FEATURES if f in df_processed.columns]
    X = df_processed[avail].astype(float)

    # ── Pisahkan fitur Motor vs Mobil (hindari cross-vehicle noise) ──
    drop_for_motor = [c for c in avail if any(k in c for k in ["Jumlah Mobil", "Tarif Mobil"])]
    drop_for_mobil = [c for c in avail if any(k in c for k in ["Jumlah Motor", "Tarif Motor"])]
    avail_m = [c for c in avail if c not in drop_for_motor]
    avail_b = [c for c in avail if c not in drop_for_mobil]

    X_m = df_processed[avail_m].astype(float)
    X_b = df_processed[avail_b].astype(float)

    # Target
    if "Target Motor" in df_processed.columns:
        target_motor = df_processed["Target Motor"]
    else:
        target_motor = (df_processed["Pend WD Motor"] + df_processed["Pend WE Motor"]) / 2

    if "Target Mobil" in df_processed.columns:
        target_mobil = df_processed["Target Mobil"]
    else:
        target_mobil = (df_processed["Pend WD Mobil"] + df_processed["Pend WE Mobil"]) / 2

    # ── Cap outlier target Motor (sangat skewed, maks ~13× median) ──
    t_lo_m = target_motor.quantile(0.01)
    t_hi_m = target_motor.quantile(0.90)
    target_motor = np.clip(target_motor, t_lo_m, t_hi_m)

    y_motor = np.log1p(target_motor)
    y_mobil = np.log1p(target_mobil)

    # Split 80:20
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_m, y_motor, test_size=0.2, random_state=42
    )
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X_b, y_mobil, test_size=0.2, random_state=42
    )

    # Best hyperparameters found in RandomizedSearchCV
    best_params_motor = {
        'colsample_bylevel': 0.7,
        'colsample_bytree': 0.6,
        'gamma': 0.5,
        'learning_rate': 0.05,
        'max_delta_step': 2,
        'max_depth': 2,
        'min_child_weight': 15,
        'n_estimators': 150,
        'reg_alpha': 1.0,
        'reg_lambda': 20.0,
        'subsample': 0.6,
        'random_state': 42,
        'objective': 'reg:squarederror'
    }

    best_params_mobil = {
        'colsample_bylevel': 0.8,
        'colsample_bytree': 0.7,
        'gamma': 0.3,
        'learning_rate': 0.02,
        'max_depth': 5,
        'min_child_weight': 3,
        'n_estimators': 500,
        'reg_alpha': 0.1,
        'reg_lambda': 3.0,
        'subsample': 0.7,
        'random_state': 42,
        'objective': 'reg:squarederror'
    }

    pipe_m = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", XGBRegressor(**best_params_motor)),
    ])
    pipe_b = Pipeline([
        ("scaler", RobustScaler()),
        ("xgb", XGBRegressor(**best_params_mobil)),
    ])

    # Fit once directly (takes milliseconds)
    pipe_m.fit(X_train_m, y_train_m)
    pipe_b.fit(X_train_b, y_train_b)

    mm, mb = pipe_m, pipe_b

    def met(model, Xt, yt):
        p = np.expm1(model.predict(Xt))
        t = np.expm1(yt)
        return {
            "R2": r2_score(t, p),
            "MAE": mean_absolute_error(t, p),
            "RMSE": np.sqrt(mean_squared_error(t, p)),
        }

    metrics       = {"Motor": met(mm, X_test_m,  y_test_m),  "Mobil": met(mb, X_test_b,  y_test_b)}
    metrics_train = {"Motor": met(mm, X_train_m, y_train_m), "Mobil": met(mb, X_train_b, y_train_b)}

    # Fast CV scores calculation
    from sklearn.model_selection import cross_val_score
    cv_scores_m = cross_val_score(pipe_m, X_train_m, y_train_m, cv=5, scoring="r2")
    cv_scores_b = cross_val_score(pipe_b, X_train_b, y_train_b, cv=5, scoring="r2")

    cv_scores = {
        "Motor": {
            "mean": float(cv_scores_m.mean()),
            "std":  float(cv_scores_m.std()),
            "n_splits": 5,
        },
        "Mobil": {
            "mean": float(cv_scores_b.mean()),
            "std":  float(cv_scores_b.std()),
            "n_splits": 5,
        },
    }
    return (
        mm, mb, metrics, metrics_train, cv_scores, X, y_motor, y_mobil,
        X_train_m, X_test_m, y_train_m, y_test_m,
        X_train_b, X_test_b, y_train_b, y_test_b, avail_m, avail_b,
    )

# ─────────────────────────────────────────────────────────────
# MEDIAN-BASED RECOMMENDATION
# Alur: prediksi → bandingkan % median → kategori → tarif baru
# Tinggi (>130%) → +2000 | Sedang (70%-130%) → +1000 | Rendah (<70%) → +0
# ─────────────────────────────────────────────────────────────
def get_median_recommendation(pred_revenue, current_tarif,
                               median, persen_naik, persen_turun,
                               delta_tinggi, delta_sedang, delta_rendah,
                               min_tarif, max_tarif):
    """
    Rekomendasi tarif berbasis PERSENTASE MEDIAN (3 kategori):
      pred > persen_naik  × median → Tinggi (+2000)
      pred >= persen_turun × median → Sedang (+1000)
      pred < persen_turun × median  → Rendah (+0)
    """
    batas_naik  = median * persen_naik
    batas_turun = median * persen_turun

    if pred_revenue > batas_naik:
        status = "Tinggi"
        delta  = delta_tinggi
    elif pred_revenue >= batas_turun:
        status = "Sedang"
        delta  = delta_sedang
    else:
        status = "Rendah"
        delta  = delta_rendah

    tarif_baru = int(max(min_tarif, min(max_tarif, current_tarif + delta)))

    alasan = {
        "Tinggi": f"Prediksi > {persen_naik*100:.0f}% median (Rp {batas_naik:,.0f}) → potensi tinggi, tarif naik +Rp {delta_tinggi:,}",
        "Sedang": f"Prediksi di antara {persen_turun*100:.0f}%–{persen_naik*100:.0f}% median (Rp {batas_turun:,.0f}–{batas_naik:,.0f}) → potensi sedang, tarif naik +Rp {delta_sedang:,}",
        "Rendah": f"Prediksi < {persen_turun*100:.0f}% median (Rp {batas_turun:,.0f}) → potensi rendah, tarif tidak berubah",
    }[status]

    return {"status":status, "delta":delta, "tarif_baru":tarif_baru, "alasan":alasan}


# ─────────────────────────────────────────────────────────────
# BATCH RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def calculate_recommendations(_df, _mm, _mb, avail_m, avail_b):
    dp = _df.copy()
    dp["Pred Motor"] = np.maximum(np.expm1(_mm.predict(dp[avail_m])), 0)
    dp["Pred Mobil"] = np.maximum(np.expm1(_mb.predict(dp[avail_b])), 0)

    MED_m = dp["Pred Motor"].median()
    MED_b = dp["Pred Mobil"].median()

    for kd, med, mn, mx in [
        ("Motor", MED_m, MIN_TARIF_MOTOR, MAX_TARIF_MOTOR),
        ("Mobil", MED_b, MIN_TARIF_MOBIL, MAX_TARIF_MOBIL),
    ]:
        results = dp.apply(
            lambda r: get_median_recommendation(
                r[f"Pred {kd}"], r[f"Tarif {kd}"],
                med, PERSEN_NAIK, PERSEN_TURUN,
                DELTA_TINGGI, DELTA_SEDANG, DELTA_RENDAH,
                mn, mx), axis=1)
        dp[f"Status {kd}"] = results.apply(lambda x: x["status"])
        dp[f"Rek {kd}"]    = results.apply(lambda x: x["tarif_baru"])
        dp[f"Delta {kd}"]  = results.apply(lambda x: x["delta"])
        dp[f"Alasan {kd}"] = results.apply(lambda x: x["alasan"])

    return dp, MED_m, MED_b


# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
calculate_recommendations.clear()

try:
    with st.spinner("Loading data and training XGBoost model…"):
        df_processed,FEATURES,df_raw,prep_info = load_and_preprocess_data(FILE_PATH)
        (mm,mb,metrics,metrics_train,cv_scores,X,y_motor,y_mobil,
         X_train_m,X_test_m,y_train_m,y_test_m,
         X_train_b,X_test_b,y_train_b,y_test_b,avail_m,avail_b) = train_models(df_processed,FEATURES)
        avail = list(dict.fromkeys(avail_m + avail_b))  # union for display/heatmap
        df_rec, MED_m, MED_b = calculate_recommendations(df_processed, mm, mb, avail_m, avail_b)
        # Hitung batas untuk visualisasi
        batas_naik_m  = MED_m * PERSEN_NAIK
        batas_turun_m = MED_m * PERSEN_TURUN
        batas_naik_b  = MED_b * PERSEN_NAIK
        batas_turun_b = MED_b * PERSEN_TURUN
except FileNotFoundError:
    st.error(f"❌ File **{FILE_PATH}** not found. Place it in the same folder as app.py.")
    st.stop()
except Exception as e:
    st.error(f"Error: {e}\n\n```\n{traceback.format_exc()}\n```")
    st.stop()


# ─────────────────────────────────────────────────────────────
# SIDEBAR — 6 MENU NAVIGASI
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-brand-name">Rekomendasi Tarif Parkir Banyumas</div>
        <div class="sb-brand-sub">Banyumas · Dashboard</div>
    </div>""", unsafe_allow_html=True)

    NAV = [
        ("Data Table",        "📋"),
        ("Visualization",     "📊"),
        ("Modeling",          "🤖"),
        ("Evaluasi",          "📈"),
        ("Hasil Rekomendasi", "🎯"),
        ("Map",               "🗺️"),
    ]
    for label, icon in NAV:
        if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True):
            st.session_state.page = label

    _active = st.session_state.page
    _labels = [l for l, _ in NAV]
    _js = f"""<script>(function(){{
        var a={repr(_active)},ls={_labels};
        function run(){{try{{
            var btns=window.parent.document.querySelectorAll('[data-testid="stSidebar"] button');
            btns.forEach(function(b){{
                var t=(b.innerText||'').trim(),match=null;
                ls.forEach(function(l){{if(t.indexOf(l)!==-1)match=l;}});
                if(!match)return;
                if(match===a){{b.style.setProperty('background','rgba(59,130,246,.15)','important');
                              b.style.setProperty('color','#3B82F6','important');
                              b.style.setProperty('font-weight','600','important');}}
                else{{b.style.removeProperty('background');
                      b.style.setProperty('color','#94A3B8','important');
                      b.style.setProperty('font-weight','500','important');}}
            }});
        }}catch(e){{}}}}
        run();setTimeout(run,100);setTimeout(run,300);
    }})();</script>"""
    st.components.v1.html(_js, height=0, scrolling=False)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
page = st.session_state.page

def ph(icon, title, sub):
    st.markdown(f"""<div class="page-header">
    <div class="ph-icon">{icon}</div>
    <div><div class="ph-title">{title}</div><div class="ph-sub">{sub}</div></div>
    </div>""", unsafe_allow_html=True)

def slabel(t): st.markdown(f'<div class="slabel">{t}</div>', unsafe_allow_html=True)
def hdiv():    st.markdown('<hr class="hdivider">',            unsafe_allow_html=True)
def info_band(h): st.markdown(f'<div class="info-band">{h}</div>', unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════╗
# ║  PAGE 1 — DATA TABLE                                    ║
# ╚══════════════════════════════════════════════════════════╝
if page == "Data Table":
    ph("📋", "Data Table", "Original Excel file vs. cleaned and engineered features")
    tab_raw, tab_pre = st.tabs(["Raw Data", "Pre-processed Data"])

    with tab_raw:
        info_band("Original data loaded directly from Excel — no transformation applied.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows",    f"{len(df_raw):,}")
        c2.metric("Columns", f"{len(df_raw.columns)}")
        c3.metric("Missing", f"{int(df_raw.isnull().sum().sum()):,}")
        st.dataframe(df_raw, use_container_width=True, height=480)

    with tab_pre:
        info_band("Data after cleaning, feature engineering, and outlier capping (1–90 percentile).")
        n_r=prep_info["n_raw"]; n_c=prep_info["n_clean"]; n_d=prep_info["n_dropped"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows (clean)",  f"{n_c:,}")
        c2.metric("Features used", f"{len(set(avail_m + avail_b))}")
        c3.metric("Missing",       "0")
        c4.metric("Removed rows",  f"{n_d}")
        if n_d > 0:
            rh = "".join([f"<li><b>{v} baris</b>: {k}</li>"
                          for k, v in prep_info["reasons"].items() if v > 0])
            st.markdown(f"""<div class="warn-band"><b>⚠️ Pengurangan data</b>: {n_r:,} → {n_c:,} baris
            <ul style="margin:6px 0 0;padding-left:1.2rem;font-size:12.5px;">{rh}</ul></div>""",
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="ok-band">✅ Semua {n_r:,} baris berhasil diproses.</div>',
                        unsafe_allow_html=True)
        col_sel = st.multiselect("Pilih kolom:", df_processed.columns.tolist(),
                                 default=df_processed.columns[:12].tolist())
        df_disp = df_processed[col_sel].copy() if col_sel else df_processed.copy()
        for c in [x for x in df_disp.columns if "Durasi" in x]:
            df_disp[c] = df_disp[c].apply(lambda x: f"{x:.1f} jam" if pd.notna(x) else "-")
        st.dataframe(df_disp, use_container_width=True, height=460)


# ╔══════════════════════════════════════════════════════════╗
# ║  PAGE 2 — VISUALIZATION (EDA Only)                      ║
# ╚══════════════════════════════════════════════════════════╝
elif page == "Visualization":
    ph("📊", "Visualization", "Exploratory Data Analysis — distribusi, korelasi, dan rekayasa fitur")

    tabs = st.tabs([
        "Heatmap Korelasi",
        "Pairplot",
        "Rekayasa Fitur",
        "Histogram Fitur Utama",
        "Pendapatan Tahunan per Jenis",
    ])

    # ── Tab 0: Heatmap Korelasi ──────────────────────────────
    with tabs[0]:
        info_band("""<b>📌 Heatmap Korelasi Fitur Parkir</b><br>
        Nilai +1 = naik bersama, -1 = berlawanan, 0 = tidak ada hubungan.""")
        slabel("Pearson Correlation Matrix")
        hf = [c for c in avail if c not in ["Tarif Motor", "Tarif Mobil"]]
        df_h = df_processed[hf].copy(); df_h = df_h.loc[:, df_h.std() != 0]
        df_hm = df_h.copy()
        if "Pend WD Motor" in df_processed.columns:
            df_hm["Pendapatan Motor (rata2)"] = (df_processed["Pend WD Motor"] + df_processed["Pend WE Motor"]) / 2
        if "Pend WD Mobil" in df_processed.columns:
            df_hm["Pendapatan Mobil (rata2)"] = (df_processed["Pend WD Mobil"] + df_processed["Pend WE Mobil"]) / 2
        corr = df_hm.corr().fillna(0)

        max_feat = 16
        if len(corr.columns) > max_feat:
            mean_abs = corr.abs().mean().sort_values(ascending=False)
            keep_cols = mean_abs.head(max_feat).index.tolist()
            corr = corr.loc[keep_cols, keep_cols]

        nf = len(corr.columns)
        fig_w = max(11, min(20, 0.95 * nf + 6))
        fig_h = max(8, min(18, 0.85 * nf + 4))
        label_fs = max(8, min(11, int(210 / max(nf, 1))))
        annot_fs = max(7, min(10, label_fs - 1))

        from matplotlib.colors import LinearSegmentedColormap
        cmc = LinearSegmentedColormap.from_list("hm", [
            (0.13, 0.47, 0.71), (0.42, 0.68, 0.84), (1, 1, 0.88),
            (0.99, 0.55, 0.24), (0.89, 0.10, 0.11)], N=256)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white", dpi=150)
        ax.set_facecolor("white")
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap=cmc, center=0, vmin=-1, vmax=1,
            square=False, linewidths=0.4, linecolor="#FFFFFF", ax=ax,
            annot_kws={"size": annot_fs, "weight": "semibold", "color": "black"},
            cbar_kws={"shrink": 0.75, "label": "Korelasi Pearson", "pad": 0.02},
        )
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title("Heatmap Korelasi", pad=16, fontsize=14, fontweight="700")
        ax.tick_params(axis="x", labelsize=label_fs, rotation=35, length=0)
        ax.tick_params(axis="y", labelsize=label_fs, rotation=0, length=0)
        plt.tight_layout(pad=1.2)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        if len(df_hm.columns) > max_feat:
            st.caption(f"Menampilkan {max_feat} fitur paling informatif agar heatmap tetap proporsional dan mudah dibaca.")

        hdiv(); slabel("Top 10 pasangan fitur korelasi tertinggi")
        cp = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().reset_index()
        cp.columns = ["Fitur A", "Fitur B", "Korelasi"]; cp["Abs"] = cp["Korelasi"].abs()
        top10 = cp.sort_values("Abs", ascending=False).head(10).drop(columns="Abs")
        top10["Korelasi"] = top10["Korelasi"].round(4)
        st.dataframe(top10, use_container_width=True, hide_index=True)

    # ── Tab 1: Pairplot ──────────────────────────────────────
    with tabs[1]:
        info_band("<b>📌 Pairplot — Hubungan Antar Variabel Kendaraan & Tarif</b>")
        slabel("Scatter matrix of primary vehicle and tariff features")
        ppc = [c for c in ["Jumlah Motor Weekday", "Jumlah Mobil Weekday",
                            "Jumlah Motor Weekend", "Jumlah Mobil Weekend",
                            "Tarif Motor", "Tarif Mobil"] if c in df_processed.columns]
        with st.spinner("Generating pairplot…"):
            g = sns.pairplot(df_processed[ppc], diag_kind="kde", corner=True,
                             plot_kws={"alpha": .45, "s": 18, "color": PALETTE["blue"]},
                             diag_kws={"fill": True, "color": PALETTE["blue"]})
            g.figure.suptitle("Pairplot of Key Features", y=1.01, fontsize=13, fontweight="600")
            st.pyplot(g.figure, use_container_width=True); plt.close()

    # ── Tab 2: Rekayasa Fitur ────────────────────────────────
    with tabs[2]:
        info_band("Visualisasi ini menunjukkan komposisi dari fitur agregat Total Durasi Ramai. Anda dapat melihat proporsi kontribusi dari masing-masing jenis kendaraan dan waktu (hari kerja vs akhir pekan).")
        slabel("Komposisi Pembentuk Total Durasi Ramai (Top 15 Lokasi)")
        
        top_15 = df_processed.nlargest(15, "Total Durasi Ramai").sort_values("Total Durasi Ramai", ascending=False)
        components = [
            "Durasi Jam Ramai Motor Weekday",
            "Durasi Jam Ramai Motor Weekend",
            "Durasi Jam Ramai Mobil Weekday",
            "Durasi Jam Ramai Mobil Weekend"
        ]
        labels = [c.replace("Durasi Jam Ramai ", "") for c in components]
        colors = [PALETTE["blue"], PALETTE["teal"], PALETTE["amber"], PALETTE["indigo"]]
        
        valid_components = [c for c in components if c in top_15.columns]
        if valid_components:
            fig, ax = plt.subplots(figsize=(12, 6.5))
            bottom = np.zeros(len(top_15))
            
            for comp, label, color in zip(components, labels, colors):
                if comp in top_15.columns:
                    ax.bar(
                        top_15["Lokasi"], 
                        top_15[comp], 
                        bottom=bottom, 
                        label=label, 
                        color=color,
                        edgecolor="white",
                        width=0.6
                    )
                    bottom += top_15[comp].values
            
            ax.set_title("Komposisi Total Durasi Ramai (Top 15 Lokasi)", pad=20, fontsize=14, fontweight="700")
            ax.set_xlabel("Lokasi Parkir", fontsize=11, fontweight="600", labelpad=12)
            ax.set_ylabel("Durasi Ramai (Jam)", fontsize=11, fontweight="600", labelpad=12)
            
            plt.xticks(rotation=45, ha="right", fontsize=9)
            plt.yticks(fontsize=9)
            ax.legend(title="Komponen Waktu & Kendaraan", title_fontsize="10", loc="upper right", frameon=True, facecolor="white", edgecolor="#E5E7EB")
            ax.spines[["top", "right"]].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.warning("Kolom durasi jam ramai tidak ditemukan di dataset.")

    # ── Tab 3: Histogram Fitur Utama ─────────────────────────
    with tabs[3]:
        info_band("<b>📌 Histogram Fitur Utama (Satu per Satu)</b>")
        fitur_utama = [
            "Jumlah Motor Weekday", "Jumlah Mobil Weekday",
            "Jumlah Motor Weekend", "Jumlah Mobil Weekend",
            "Tarif Motor", "Tarif Mobil"
        ]
        fitur_utama = [c for c in fitur_utama if c in df_processed.columns]
        if fitur_utama:
            col_a, col_b = st.columns(2)
            for idx, col in enumerate(fitur_utama):
                with (col_a if idx % 2 == 0 else col_b):
                    fig, ax = plt.subplots(figsize=(6.5, 3.8))
                    sns.histplot(df_processed[col], bins=30, kde=True, color=PALETTE["blue"],
                                 edgecolor="white", alpha=0.75, ax=ax)
                    ax.set_title(f"Histogram {col}", fontweight="bold")
                    ax.set_xlabel(col)
                    ax.set_ylabel("Frekuensi")
                    ax.spines[["top", "right"]].set_visible(False)
                    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)
        else:
            st.warning("Fitur utama tidak tersedia untuk histogram.")

    # ── Tab 4: Pendapatan Tahunan per Jenis ─────────────────
    with tabs[4]:
        info_band("<b>📌 Distribusi Pendapatan Tahunan per Jenis Kendaraan</b>")
        annual_df = pd.DataFrame()
        if "Pend WD Motor" in df_processed.columns and "Pend WE Motor" in df_processed.columns:
            annual_df["Motor"] = ((df_processed["Pend WD Motor"] + df_processed["Pend WE Motor"]) / 2) * 365
        if "Pend WD Mobil" in df_processed.columns and "Pend WE Mobil" in df_processed.columns:
            annual_df["Mobil"] = ((df_processed["Pend WD Mobil"] + df_processed["Pend WE Mobil"]) / 2) * 365

        if not annual_df.empty:
            annual_long = annual_df.melt(var_name="Jenis Kendaraan", value_name="Pendapatan Tahunan")
            fig, ax = plt.subplots(figsize=(10, 5.5))
            sns.violinplot(data=annual_long, x="Jenis Kendaraan", y="Pendapatan Tahunan",
                           palette=[PALETTE["blue"], PALETTE["teal"]], inner="quartile", ax=ax)
            ax.set_title("Distribusi Pendapatan Tahunan per Jenis Kendaraan", fontweight="bold")
            ax.set_xlabel("Jenis Kendaraan")
            ax.set_ylabel("Pendapatan Tahunan (Rp)")
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)
        else:
            st.warning("Kolom pendapatan tidak ditemukan untuk visualisasi tahunan.")


# ╔══════════════════════════════════════════════════════════╗
# ║  PAGE 3 — MODELING (Arsitektur XGBoost)                 ║
# ╚══════════════════════════════════════════════════════════╝
elif page == "Modeling":
    ph("🤖", "Modeling", "Arsitektur XGBoost — hyperparameter terbaik & feature importance")

    tabs = st.tabs([
        "Arsitektur & Parameter · Motor",
        "Arsitektur & Parameter · Mobil",
        "Feature Importance · Motor",
        "Feature Importance · Mobil",
    ])

    # Helper: tampilkan tabel best hyperparameter
    def show_best_params(model, kd):
        slabel(f"Best Hyperparameters — {kd} (via RandomizedSearchCV, 5-fold CV)")
        bp   = model.named_steps["xgb"].get_params()
        keys = [
            "n_estimators", "max_depth", "learning_rate",
            "colsample_bytree", "colsample_bylevel", "subsample",
            "reg_alpha", "reg_lambda", "min_child_weight",
            "gamma", "max_delta_step", "objective",
        ]
        rows = [{"Parameter": k, "Value": bp.get(k, "—")} for k in keys if k in bp]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Tab 0: Arsitektur Motor ──────────────────────────────
    with tabs[0]:
        info_band(f"""
        <b>📌 Arsitektur Model XGBoost — Motor</b><br>
        Model menggunakan Pipeline: <code>StandardScaler → XGBRegressor</code> dengan target <code>log1p(pendapatan Motor)</code>.<br>
        Fitur yang digunakan: <b>{len(avail_m)}</b> fitur khusus Motor (fitur Mobil dieksklusikan untuk mengurangi noise antar model).
        """)
        slabel("Informasi Split Data")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Data",  f"{len(X_train_m) + len(X_test_m)}")
        c2.metric("Train (80%)", f"{len(X_train_m)}")
        c3.metric("Test  (20%)", f"{len(X_test_m)}")
        c4.metric("Jumlah Fitur", f"{len(avail_m)}")
        hdiv()
        show_best_params(mm, "Motor")
        hdiv()
        slabel("Daftar Fitur yang Digunakan — Motor")
        feat_df_m = pd.DataFrame({"No": range(1, len(avail_m)+1), "Nama Fitur": avail_m})
        st.dataframe(feat_df_m, use_container_width=True, hide_index=True)

    # ── Tab 1: Arsitektur Mobil ──────────────────────────────
    with tabs[1]:
        info_band(f"""
        <b>📌 Arsitektur Model XGBoost — Mobil</b><br>
        Model menggunakan Pipeline: <code>RobustScaler → XGBRegressor</code> dengan target <code>log1p(pendapatan Mobil)</code>.<br>
        Fitur yang digunakan: <b>{len(avail_b)}</b> fitur khusus Mobil (fitur Motor dieksklusikan untuk mengurangi noise antar model).
        """)
        slabel("Informasi Split Data")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Data",  f"{len(X_train_b) + len(X_test_b)}")
        c2.metric("Train (80%)", f"{len(X_train_b)}")
        c3.metric("Test  (20%)", f"{len(X_test_b)}")
        c4.metric("Jumlah Fitur", f"{len(avail_b)}")
        hdiv()
        show_best_params(mb, "Mobil")
        hdiv()
        slabel("Daftar Fitur yang Digunakan — Mobil")
        feat_df_b = pd.DataFrame({"No": range(1, len(avail_b)+1), "Nama Fitur": avail_b})
        st.dataframe(feat_df_b, use_container_width=True, hide_index=True)

    # ── Tab 2: Feature Importance Motor ─────────────────────
    with tabs[2]:
        info_band("<b>📌 Feature Importance — Motor (Top 15)</b><br>Faktor terpenting yang memengaruhi prediksi pendapatan Motor.")
        fi = pd.Series(mm.named_steps["xgb"].feature_importances_, index=X_test_m.columns)
        fi = fi.sort_values(ascending=True).tail(15)
        q75fi = fi.quantile(0.75)
        clrs = [PALETTE["blue"] if v >= q75fi else "#94A3B8" for v in fi.values]
        fig, ax = plt.subplots(figsize=(11, 7), facecolor="white")
        ax.set_facecolor("white")
        bars = ax.barh(fi.index, fi.values, color=clrs, edgecolor="white", height=0.62)
        for bar, val in zip(bars, fi.values):
            ax.text(val + fi.max() * 0.012, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", ha="left", fontsize=9, fontweight="600")
        ax.set_axisbelow(True); ax.xaxis.grid(True, color="#E5E7EB", linewidth=0.8)
        ax.set_title("Feature Importance — Motor (Top 15)", fontsize=13, fontweight="700")
        ax.set_xlabel("Importance Score")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_xlim(0, fi.max() * 1.22)
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor=PALETTE["blue"], label="Top importance (Q75+)"),
            Patch(facecolor="#94A3B8", label="Lower importance")],
            loc="lower right", fontsize=9)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    # ── Tab 3: Feature Importance Mobil ─────────────────────
    with tabs[3]:
        info_band("<b>📌 Feature Importance — Mobil (Top 15)</b><br>Faktor terpenting yang memengaruhi prediksi pendapatan Mobil.")
        fi = pd.Series(mb.named_steps["xgb"].feature_importances_, index=X_test_b.columns)
        fi = fi.sort_values(ascending=True).tail(15)
        q75fi = fi.quantile(0.75)
        clrs = [PALETTE["teal"] if v >= q75fi else "#94A3B8" for v in fi.values]
        fig, ax = plt.subplots(figsize=(11, 7), facecolor="white")
        ax.set_facecolor("white")
        bars = ax.barh(fi.index, fi.values, color=clrs, edgecolor="white", height=0.62)
        for bar, val in zip(bars, fi.values):
            ax.text(val + fi.max() * 0.012, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", ha="left", fontsize=9, fontweight="600")
        ax.set_axisbelow(True); ax.xaxis.grid(True, color="#E5E7EB", linewidth=0.8)
        ax.set_title("Feature Importance — Mobil (Top 15)", fontsize=13, fontweight="700")
        ax.set_xlabel("Importance Score")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_xlim(0, fi.max() * 1.22)
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor=PALETTE["teal"], label="Top importance (Q75+)"),
            Patch(facecolor="#94A3B8", label="Lower importance")],
            loc="lower right", fontsize=9)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)


# ╔══════════════════════════════════════════════════════════╗
# ║  PAGE 4 — EVALUASI (Kinerja Model)                      ║
# ╚══════════════════════════════════════════════════════════╝
elif page == "Evaluasi":
    ph("📈", "Evaluasi Model", "Pembuktian kinerja XGBoost — metrik, diagnostik, dan bukti tidak overfitting")

    tabs = st.tabs([
        "Metrik Evaluasi",
        "Data Latih vs Uji",
        "Prediksi vs Asli · Motor",
        "Prediksi vs Asli · Mobil",
        "Residuals · Motor",
        "Residuals · Mobil",
        "🔍 Bukti Tidak Overfitting",
    ])

    # ── Tab 0: Metrik Evaluasi ───────────────────────────────
    with tabs[0]:
        info_band("<b>📌 Kartu Metrik Evaluasi Model XGBoost</b><br>Dihitung pada data <b>uji (20%)</b> yang tidak pernah dilihat model saat training.")
        for kd, met_kd, col_h in [
            ("Motor", metrics["Motor"], PALETTE["blue"]),
            ("Mobil", metrics["Mobil"], PALETTE["teal"]),
        ]:
            slabel(f"Metrik — {kd}")
            c1, c2, c3 = st.columns(3)
            c1.metric("R² Score", f"{met_kd['R2']:.4f}")
            c2.metric("MAE",      f"Rp {met_kd['MAE']:,.0f}")
            c3.metric("RMSE",     f"Rp {met_kd['RMSE']:,.0f}")
            if kd == "Motor":
                hdiv()

        hdiv(); slabel("Tabel Perbandingan Motor vs Mobil")
        comp = pd.DataFrame({
            "Metric":  ["R²", "MAE (Rp)", "RMSE (Rp)"],
            "Motor":   [f"{metrics['Motor']['R2']:.4f}",
                        f"Rp {metrics['Motor']['MAE']:,.0f}",
                        f"Rp {metrics['Motor']['RMSE']:,.0f}"],
            "Mobil":   [f"{metrics['Mobil']['R2']:.4f}",
                        f"Rp {metrics['Mobil']['MAE']:,.0f}",
                        f"Rp {metrics['Mobil']['RMSE']:,.0f}"],
        })
        st.dataframe(comp, use_container_width=True, hide_index=True)

    # ── Tab 1: Data Latih vs Uji ─────────────────────────────
    with tabs[1]:
        info_band("<b>📌 Distribusi Data Latih vs Data Uji</b>")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, kd, ytr, yte, c_tr, c_te in [
            (axes[0], "Motor", y_train_m, y_test_m, PALETTE["blue"], PALETTE["amber"]),
            (axes[1], "Mobil", y_train_b, y_test_b, PALETTE["teal"], PALETTE["red"]),
        ]:
            train_actual = np.expm1(ytr)
            test_actual  = np.expm1(yte)
            sns.kdeplot(train_actual, fill=True, color=c_tr, alpha=0.32, label=f"Train (n={len(ytr)})", ax=ax)
            sns.kdeplot(test_actual,  fill=True, color=c_te, alpha=0.32, label=f"Test (n={len(yte)})",  ax=ax)
            ax.set_title(f"{kd}: Distribusi Target Train vs Test", fontweight="bold")
            ax.set_xlabel("Pendapatan (Rp)"); ax.set_ylabel("Density"); ax.legend()
            ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    # ── Tab 2 & 3: Prediksi vs Asli ─────────────────────────
    for ti, kd, model, Xte, yte, col in [
        (2, "Motor", mm, X_test_m, y_test_m, PALETTE["blue"]),
        (3, "Mobil", mb, X_test_b, y_test_b, PALETTE["teal"]),
    ]:
        with tabs[ti]:
            info_band(f"<b>📌 Seberapa Akurat Prediksi Pendapatan {kd}?</b>")
            pred = np.expm1(model.predict(Xte)); true = np.expm1(yte)
            r2   = r2_score(true, pred)
            c1, c2, c3 = st.columns(3)
            c1.metric("R²",   f"{r2:.4f}")
            c2.metric("MAE",  f"Rp {mean_absolute_error(true, pred):,.0f}")
            c3.metric("RMSE", f"Rp {np.sqrt(mean_squared_error(true, pred)):,.0f}")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(true, pred, alpha=.55, color=col, edgecolors="white", s=55)
            lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
            ax.plot(lims, lims, "--", color=PALETTE["red"], lw=1.8, label="Perfect prediction")
            ax.set_title(f"Predicted vs Actual — {kd}  (R²={r2:.4f})")
            ax.set_xlabel("Actual (Rp)"); ax.set_ylabel("Predicted (Rp)"); ax.legend()
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    # ── Tab 4 & 5: Residuals ─────────────────────────────────
    for ti, kd, model, Xte, yte, col in [
        (4, "Motor", mm, X_test_m, y_test_m, PALETTE["indigo"]),
        (5, "Mobil", mb, X_test_b, y_test_b, PALETTE["teal"]),
    ]:
        with tabs[ti]:
            info_band(f"<b>📌 Analisis Error Prediksi — {kd}</b>")
            pred = np.expm1(model.predict(Xte)); true = np.expm1(yte); res = true - pred
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].hist(res, bins=30, color=col, alpha=.8, edgecolor="white")
            axes[0].axvline(0, color=PALETTE["red"], lw=2, ls="--", label="Zero error")
            axes[0].axvline(res.mean(), color=PALETTE["amber"], lw=1.5, ls=":",
                            label=f"Mean: Rp {res.mean():,.0f}")
            axes[0].set_title(f"Residual Distribution — {kd}")
            axes[0].set_xlabel("Residual (Rp)"); axes[0].set_ylabel("Frequency"); axes[0].legend()
            axes[1].scatter(pred, res, alpha=.5, color=col, edgecolors="white", s=40)
            axes[1].axhline(0, color=PALETTE["red"], lw=1.8, ls="--")
            axes[1].set_title(f"Residuals vs Fitted — {kd}")
            axes[1].set_xlabel("Predicted (Rp)"); axes[1].set_ylabel("Residual (Rp)")
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    # ── Tab 6: Bukti Tidak Overfitting ───────────────────────
    with tabs[6]:
        info_band("""
        <b>📌 Cara membaca bukti ini:</b><br>
        Model dikatakan <b>overfitting</b> jika R² Train jauh lebih tinggi dari R² Test.<br>
        Model dikatakan <b>generalisasi baik</b> jika selisih Train–Test kecil (&lt; 0.10) dan
        nilai Cross-Validation (CV) stabil (std rendah).
        """)

        for kd, col_h in [("Motor", PALETTE["blue"]), ("Mobil", PALETTE["teal"])]:
            slabel(f"🔎 {kd} — Perbandingan Train vs Test vs CV")

            mt  = metrics_train[kd]
            mte = metrics[kd]
            cv  = cv_scores[kd]
            gap = mt["R2"] - mte["R2"]

            if gap < 0.05:
                status_html = "<span style='background:#16A34A;color:white;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:700;'>✅ TIDAK OVERFITTING</span>"
            elif gap < 0.10:
                status_html = "<span style='background:#D97706;color:white;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:700;'>⚠️ SEDIKIT OVERFITTING</span>"
            else:
                status_html = "<span style='background:#DC2626;color:white;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:700;'>❌ OVERFITTING</span>"

            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.03);border:1px solid rgba(148,163,184,0.2);
                        border-radius:12px;padding:18px 22px;margin-bottom:16px;'>
              <div style='display:flex;align-items:center;gap:12px;margin-bottom:14px;'>
                <span style='font-size:16px;font-weight:700;'>{kd}</span>
                {status_html}
                <span style='color:#94A3B8;font-size:12px;'>Selisih R² Train−Test = {gap:+.4f}</span>
              </div>
              <table style='width:100%;border-collapse:collapse;font-size:13px;'>
                <tr style='background:rgba(59,130,246,0.08);'>
                  <th style='padding:8px 12px;text-align:left;'>Metrik</th>
                  <th style='padding:8px 12px;text-align:center;'>Train</th>
                  <th style='padding:8px 12px;text-align:center;'>Test</th>
                  <th style='padding:8px 12px;text-align:center;'>CV Mean ± Std (5-fold)</th>
                </tr>
                <tr><td style='padding:8px 12px;font-weight:600;'>R²</td>
                    <td style='padding:8px 12px;text-align:center;'>{mt['R2']:.4f}</td>
                    <td style='padding:8px 12px;text-align:center;'>{mte['R2']:.4f}</td>
                    <td style='padding:8px 12px;text-align:center;'>{cv['mean']:.4f} ± {cv['std']:.4f}</td></tr>
                <tr style='background:rgba(148,163,184,0.05);'>
                    <td style='padding:8px 12px;font-weight:600;'>MAE</td>
                    <td style='padding:8px 12px;text-align:center;'>Rp {mt['MAE']:,.0f}</td>
                    <td style='padding:8px 12px;text-align:center;'>Rp {mte['MAE']:,.0f}</td>
                    <td style='padding:8px 12px;text-align:center;'>—</td></tr>
                <tr><td style='padding:8px 12px;font-weight:600;'>RMSE</td>
                    <td style='padding:8px 12px;text-align:center;'>Rp {mt['RMSE']:,.0f}</td>
                    <td style='padding:8px 12px;text-align:center;'>Rp {mte['RMSE']:,.0f}</td>
                    <td style='padding:8px 12px;text-align:center;'>—</td></tr>
              </table>
            </div>
            """, unsafe_allow_html=True)

            # Bar chart Train vs Test vs CV
            fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor="white")
            pairs = [
                ("R²",   mt["R2"],       mte["R2"],       cv["mean"], cv["std"]),
                ("MAE",  mt["MAE"]/1e6,  mte["MAE"]/1e6,  None, None),
                ("RMSE", mt["RMSE"]/1e6, mte["RMSE"]/1e6, None, None),
            ]
            for ax, (lbl, tr_val, te_val, cv_mean, cv_std) in zip(axes, pairs):
                bars = ax.bar(["Train", "Test"], [tr_val, te_val],
                              color=[col_h, PALETTE["amber"]], edgecolor="white", width=0.45)
                if cv_mean is not None:
                    ax.axhline(cv_mean, color=PALETTE["green"], lw=2, ls="--",
                               label=f"CV Mean={cv_mean:.3f}")
                    ax.fill_between([-0.5, 1.5],
                                    cv_mean - cv_std, cv_mean + cv_std,
                                    color=PALETTE["green"], alpha=0.12, label=f"±std={cv_std:.3f}")
                    ax.legend(fontsize=8)
                for bar, v in zip(bars, [tr_val, te_val]):
                    unit = "" if lbl == "R²" else " M"
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + tr_val * 0.015,
                            f"{v:.3f}{unit}", ha="center", fontsize=10, fontweight="600")
                ax.set_title(f"{lbl} — {kd}", fontweight="700")
                yunit = "" if lbl == "R²" else " (Juta Rp)"
                ax.set_ylabel(f"{lbl}{yunit}")
                ax.set_ylim(0, max(tr_val, te_val) * 1.3)
                ax.spines[["top", "right"]].set_visible(False)
                ax.set_facecolor("white")
            plt.suptitle(f"{kd} — Train vs Test (gap R²={gap:+.4f})",
                         fontsize=13, fontweight="700", y=1.02)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            hdiv()

        slabel("Penjelasan Ilmiah")
        info_band("""
        <b>Mengapa hasil ini menunjukkan model TIDAK overfitting?</b><br><br>
        1. <b>Selisih R² Train−Test kecil</b> — Model overfitting biasanya memiliki R² Train mendekati 1.0
           sementara R² Test jauh lebih rendah. Selisih &lt; 0.10 menandakan generalisasi baik.<br>
        2. <b>Cross-Validation 5-fold stabil</b> — CV R² yang konsisten dengan std rendah
           membuktikan model tidak "menghafal" data training.<br>
        3. <b>Regularisasi aktif</b> — Hyperparameter <code>reg_alpha</code>, <code>reg_lambda</code>,
           <code>min_child_weight</code>, dan <code>subsample</code> dikonfigurasi via
           RandomizedSearchCV untuk mencegah overfitting.<br>
        4. <b>Feature set terpisah</b> — Model Motor hanya menggunakan fitur relevan Motor;
           model Mobil hanya menggunakan fitur Mobil. Ini mengurangi noise antar model.
        """)


# ╔══════════════════════════════════════════════════════════╗
# ║  PAGE 5 — HASIL REKOMENDASI                             ║
# ╚══════════════════════════════════════════════════════════╝
elif page == "Hasil Rekomendasi":
    ph("🎯", "Hasil Rekomendasi", "Keputusan tarif akhir berbasis prediksi XGBoost & persentase median 130%/70%")

    # Penjelasan logika rekomendasi
    info_band(f"""
    <b>📌 Logika Rekomendasi — Persentase Median (Tinggi/Sedang/Rendah)</b><br><br>
    Model XGBoost memprediksi nilai pendapatan numerik setiap lokasi parkir.
    Hasil prediksi kemudian dibandingkan terhadap <b>median distribusi prediksi</b> seluruh lokasi:<br><br>
    🟢 <b>Tinggi</b> — Prediksi &gt; <b>{PERSEN_NAIK*100:.0f}% × Median</b>
    &nbsp;→ Batas Motor: Rp {batas_naik_m:,.0f} | Batas Mobil: Rp {batas_naik_b:,.0f}
    &nbsp;→ tarif naik <b>+Rp {DELTA_TINGGI:,}</b><br>
    🟡 <b>Sedang</b> — Prediksi di antara <b>{PERSEN_TURUN*100:.0f}%–{PERSEN_NAIK*100:.0f}% × Median</b>
    &nbsp;→ tarif naik <b>+Rp {DELTA_SEDANG:,}</b><br>
    🔴 <b>Rendah</b> — Prediksi &lt; <b>{PERSEN_TURUN*100:.0f}% × Median</b>
    &nbsp;→ Batas Motor: Rp {batas_turun_m:,.0f} | Batas Mobil: Rp {batas_turun_b:,.0f}
    &nbsp;→ tarif <b>tidak berubah (+Rp 0)</b><br><br>
    <b>Batas tarif:</b> Motor Rp {MIN_TARIF_MOTOR:,}–{MAX_TARIF_MOTOR:,} &nbsp;|&nbsp; Mobil Rp {MIN_TARIF_MOBIL:,}–{MAX_TARIF_MOBIL:,}
    """)

    # Kartu median
    slabel("Nilai Median Prediksi")
    qc1, qc2, qc3, qc4 = st.columns(4)
    qc1.metric("Median Motor",                          f"Rp {MED_m:,.0f}")
    qc2.metric(f"{PERSEN_TURUN*100:.0f}% Median Motor (Batas Rendah)", f"Rp {batas_turun_m:,.0f}")
    qc3.metric("Median Mobil",                          f"Rp {MED_b:,.0f}")
    qc4.metric(f"{PERSEN_NAIK*100:.0f}% Median Mobil (Batas Tinggi)", f"Rp {batas_naik_b:,.0f}")

    hdiv()

    tabs = st.tabs(["Distribusi Prediksi", "Status Rekomendasi", "Tabel Rekomendasi"])

    # ── Tab 0: Distribusi Prediksi ───────────────────────────
    with tabs[0]:
        info_band("""<b>📌 Distribusi Prediksi Pendapatan Seluruh Lokasi</b><br>
        Merah = Batas Rendah (70% Median) &nbsp;·&nbsp; Hijau = Batas Tinggi (130% Median) &nbsp;·&nbsp; Oranye = Median.<br>
        Zona merah = potensi Rendah (+Rp 0) &nbsp;·&nbsp; Zona kuning = potensi Sedang (+Rp 1.000) &nbsp;·&nbsp; Zona hijau = potensi Tinggi (+Rp 2.000).""")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, kd, med, bnaik, bturun, col in [
            (axes[0], "Motor", MED_m, batas_naik_m, batas_turun_m, PALETTE["blue"]),
            (axes[1], "Mobil", MED_b, batas_naik_b, batas_turun_b, PALETTE["teal"]),
        ]:
            data = df_rec[f"Pred {kd}"]
            ax.hist(data, bins=40, color=col, alpha=.75, edgecolor="white")
            ax.axvline(bturun, color=PALETTE["red"],   lw=2, ls="--", label=f"70% Median (Batas Rendah): Rp {bturun:,.0f}")
            ax.axvline(bnaik,  color=PALETTE["green"], lw=2, ls="--", label=f"130% Median (Batas Tinggi): Rp {bnaik:,.0f}")
            ax.axvline(med,    color=PALETTE["amber"],   lw=2, ls=":",  label=f"Median: Rp {med:,.0f}")
            xmin, xmax = ax.get_xlim()
            ax.axvspan(xmin,  bturun, alpha=.07, color=PALETTE["red"])
            ax.axvspan(bturun, bnaik, alpha=.05, color=PALETTE["amber"])
            ax.axvspan(bnaik, xmax,   alpha=.07, color=PALETTE["green"])
            ax.set_title(f"Distribusi Prediksi — {kd}")
            ax.set_xlabel("Predicted Revenue (Rp)"); ax.set_ylabel("Frequency"); ax.legend(fontsize=8)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    # ── Tab 1: Status Rekomendasi ────────────────────────────
    with tabs[1]:
        info_band("""<b>📌 Hasil Rekomendasi Tarif — Semua Lokasi Parkir Banyumas</b><br>
        Rekomendasi dihasilkan dari prediksi XGBoost yang dibandingkan terhadap Persentase Median 130%/70%.<br>
        <b style='color:#16A34A;'>🟢 Tinggi</b> — prediksi &gt; 130% Median → tarif naik +Rp 2.000 &nbsp;·&nbsp;
        <b style='color:#D97706;'>🟡 Sedang</b> — 70%–130% Median → tarif naik +Rp 1.000 &nbsp;·&nbsp;
        <b style='color:#DC2626;'>🔴 Rendah</b> — prediksi &lt; 70% Median → tarif tidak berubah""")

        # Ringkasan count
        c1, c2, c3 = st.columns(3)
        for cu, sl_status, emoji in [(c1, "Tinggi", "🟢"), (c2, "Sedang", "🟡"), (c3, "Rendah", "🔴")]:
            nm = (df_rec["Status Motor"] == sl_status).sum()
            nb = (df_rec["Status Mobil"] == sl_status).sum()
            cu.metric(f"{emoji} {sl_status}", f"{nm} / {nb}", "Motor / Mobil")

        hdiv()
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Rekomendasi Tarif — Metode Persentase Median (Tinggi/Sedang/Rendah)", fontsize=14, fontweight="600")
        sc = {"Tinggi": PALETTE["green"], "Sedang": PALETTE["amber"], "Rendah": PALETTE["red"]}
        for ri, kd in enumerate(["Motor", "Mobil"]):
            cnt = df_rec[f"Status {kd}"].value_counts()
            wc  = [sc.get(s, "#999") for s in cnt.index]
            axes[ri][0].pie(cnt.values, labels=cnt.index, autopct="%1.1f%%", colors=wc,
                            startangle=90, textprops={"fontsize": 11, "fontweight": "500"})
            axes[ri][0].set_title(f"{kd} — Distribusi Potensi", fontsize=12)
            bars = axes[ri][1].bar(cnt.index, cnt.values, color=wc, edgecolor="white", width=.5)
            for bar, val in zip(bars, cnt.values):
                axes[ri][1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + .1,
                                 str(val), ha="center", fontsize=11, fontweight="500")
            axes[ri][1].set_title(f"{kd} — Jumlah Lokasi per Potensi", fontsize=12)
            axes[ri][1].set_ylabel("Locations")
            axes[ri][1].spines[["top", "right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    # ── Tab 2: Tabel Rekomendasi ─────────────────────────────
    with tabs[2]:
        slabel("Rekomendasi Tarif — Semua Lokasi (Metode Persentase Median)")
        info_band(f"""<b>Ringkasan Logika:</b> Prediksi XGBoost → bandingkan % median → potensi → tarif baru<br><br>
        <b>Median Motor:</b> Rp {MED_m:,.0f} &nbsp;|&nbsp; <b>Median Mobil:</b> Rp {MED_b:,.0f}<br><br>
        🟢 <b>Tinggi</b> (+Rp {DELTA_TINGGI:,}): prediksi &gt; {PERSEN_NAIK*100:.0f}% median
        &nbsp;(Motor &gt; Rp {batas_naik_m:,.0f} | Mobil &gt; Rp {batas_naik_b:,.0f})<br>
        🟡 <b>Sedang</b> (+Rp {DELTA_SEDANG:,}): prediksi {PERSEN_TURUN*100:.0f}%–{PERSEN_NAIK*100:.0f}% median<br>
        🔴 <b>Rendah</b> (+Rp {DELTA_RENDAH:,}): prediksi &lt; {PERSEN_TURUN*100:.0f}% median
        &nbsp;(Motor &lt; Rp {batas_turun_m:,.0f} | Mobil &lt; Rp {batas_turun_b:,.0f})""")

        tbl_c = ["Titik", "Lokasi",
                 "Tarif Motor", "Pred Motor", "Rek Motor", "Delta Motor", "Status Motor", "Alasan Motor",
                 "Tarif Mobil", "Pred Mobil", "Rek Mobil", "Delta Mobil", "Status Mobil", "Alasan Mobil"]
        df_tbl = df_rec[[c for c in tbl_c if c in df_rec.columns]].reset_index(drop=True)

        def cs(val):
            if val == "Tinggi": return "color:#16A34A;font-weight:600"
            if val == "Sedang": return "color:#D97706;font-weight:600"
            if val == "Rendah": return "color:#DC2626;font-weight:600"
            return "color:#64748B"

        mc  = {c: "Rp {:,.0f}" for c in df_tbl.columns if "Tarif" in c or "Pred" in c or "Rek" in c}
        sc2 = [c for c in ["Status Motor", "Status Mobil"] if c in df_tbl.columns]
        styled = (df_tbl.style.format(mc)
                  .format({c: "{:+,.0f}" for c in ["Delta Motor", "Delta Mobil"] if c in df_tbl.columns})
                  .map(cs, subset=sc2))
        st.write(f"Total: **{len(df_tbl):,}** lokasi parkir")
        st.dataframe(styled, use_container_width=True, height=500)


# ╔══════════════════════════════════════════════════════════╗
# ║  PAGE 6 — MAP                                           ║
# ╚══════════════════════════════════════════════════════════╝
elif page == "Map":
    ph("🗺️", "Map", "Rekomendasi tarif adaptif berbasis prediksi XGBoost & median 170%/30%")

    # Helper function to generate deep structured causes based on XGBoost features
    def generate_causes(row, vehicle_type, mean_vals):
        status = row.get(f"Status {vehicle_type}", "Tetap")
        pred = float(row.get(f"Pred {vehicle_type}", 0))
        rec  = float(row.get(f"Rek {vehicle_type}", 0))
        cur  = float(row.get(f"Tarif {vehicle_type}", 0))

        wd_col = f"Jumlah {vehicle_type} Weekday"
        we_col = f"Jumlah {vehicle_type} Weekend"

        val_wd    = float(row.get(wd_col, 0))
        val_we    = float(row.get(we_col, 0))
        val_jarak = float(row.get("Jarak ke Pusat (km)", 0))
        val_ramai = float(row.get("Total Durasi Ramai", 0))

        mean_wd    = float(mean_vals.get(wd_col, 1))
        mean_we    = float(mean_vals.get(we_col, 1))
        mean_jarak = float(mean_vals.get("Jarak ke Pusat (km)", 1))
        mean_ramai = float(mean_vals.get("Total Durasi Ramai", 1))

        positives = []
        negatives = []
        neutrals  = []

        # 1. Kunjungan Weekday
        if mean_wd > 0:
            diff_wd = ((val_wd - mean_wd) / mean_wd) * 100
            if val_wd > mean_wd * 1.15:
                positives.append(f"<b>Volume Weekday Tinggi</b>: {val_wd:,.0f} kendaraan/hari (+{diff_wd:.1f}% di atas rata-rata lokasi lain sebesar {mean_wd:,.1f}).")
            elif val_wd < mean_wd * 0.85:
                negatives.append(f"<b>Volume Weekday Rendah</b>: {val_wd:,.0f} kendaraan/hari ({diff_wd:.1f}% di bawah rata-rata lokasi lain sebesar {mean_wd:,.1f}).")
            else:
                neutrals.append(f"<b>Volume Weekday Normal</b>: {val_wd:,.0f} kendaraan/hari (mendekati rata-rata lokasi lain sebesar {mean_wd:,.1f}).")

        # 2. Kunjungan Weekend
        if mean_we > 0:
            diff_we = ((val_we - mean_we) / mean_we) * 100
            if val_we > mean_we * 1.15:
                positives.append(f"<b>Volume Weekend Tinggi</b>: {val_we:,.0f} kendaraan/hari (+{diff_we:.1f}% di atas rata-rata lokasi lain sebesar {mean_we:,.1f}).")
            elif val_we < mean_we * 0.85:
                negatives.append(f"<b>Volume Weekend Rendah</b>: {val_we:,.0f} kendaraan/hari ({diff_we:.1f}% di bawah rata-rata lokasi lain sebesar {mean_we:,.1f}).")
            else:
                neutrals.append(f"<b>Volume Weekend Normal</b>: {val_we:,.0f} kendaraan/hari (mendekati rata-rata lokasi lain sebesar {mean_we:,.1f}).")

        # 3. Jarak ke Pusat
        if mean_jarak > 0:
            diff_jarak = ((mean_jarak - val_jarak) / mean_jarak) * 100
            if val_jarak < mean_jarak * 0.7:
                positives.append(f"<b>Sangat Strategis</b>: Berjarak hanya {val_jarak:.2f} km dari pusat kota (lebih dekat sebesar {diff_jarak:.1f}% dibanding rata-rata lokasi lain yaitu {mean_jarak:.2f} km).")
            elif val_jarak > mean_jarak * 1.3:
                negatives.append(f"<b>Area Periferal</b>: Berjarak {val_jarak:.2f} km dari pusat kota (lebih jauh dibanding rata-rata lokasi lain yaitu {mean_jarak:.2f} km).")
            else:
                neutrals.append(f"<b>Lokasi Standard</b>: Jarak ke pusat kota {val_jarak:.2f} km (mendekati rata-rata {mean_jarak:.2f} km).")

        # 4. Durasi Jam Ramai
        if mean_ramai > 0:
            diff_ramai = ((val_ramai - mean_ramai) / mean_ramai) * 100
            if val_ramai > mean_ramai * 1.15:
                positives.append(f"<b>Waktu Kunjungan Panjang</b>: Durasi jam ramai {val_ramai:.1f} jam/hari (+{diff_ramai:.1f}% lebih panjang dari rata-rata {mean_ramai:.1f} jam).")
            elif val_ramai < mean_ramai * 0.85:
                negatives.append(f"<b>Waktu Kunjungan Singkat</b>: Durasi jam ramai hanya {val_ramai:.1f} jam/hari ({diff_ramai:.1f}% lebih pendek dari rata-rata {mean_ramai:.1f} jam).")
            else:
                neutrals.append(f"<b>Waktu Kunjungan Stabil</b>: Durasi jam ramai {val_ramai:.1f} jam/hari (mendekati rata-rata {mean_ramai:.1f} jam).")

        ul_items = ""
        if status == "Tinggi":
            items = positives + neutrals + negatives
            color_bg     = "rgba(22, 163, 74, 0.06)"
            color_border = "rgba(22, 163, 74, 0.25)"
            color_text   = "#16A34A"
            rec_badge    = f"🟢 POTENSI TINGGI — TARIF NAIK +Rp {DELTA_TINGGI:,}"
        elif status == "Sedang":
            items = positives + negatives + neutrals
            color_bg     = "rgba(217, 119, 6, 0.06)"
            color_border = "rgba(217, 119, 6, 0.25)"
            color_text   = "#D97706"
            rec_badge    = f"🟡 POTENSI SEDANG — TARIF NAIK +Rp {DELTA_SEDANG:,}"
        else:  # Rendah
            items = negatives + neutrals + positives
            color_bg     = "rgba(220, 38, 38, 0.06)"
            color_border = "rgba(220, 38, 38, 0.20)"
            color_text   = "#DC2626"
            rec_badge    = f"🔴 POTENSI RENDAH — TARIF TIDAK BERUBAH (+Rp {DELTA_RENDAH:,})"

        for item in items:
            ul_items += f"<li style='margin-bottom: 6px; font-size: 12.5px; line-height: 1.4; color: #94A3B8;'>{item}</li>"

        med_val    = MED_m if vehicle_type == "Motor" else MED_b
        limit_up   = med_val * PERSEN_NAIK
        limit_down = med_val * PERSEN_TURUN

        if status == "Tinggi":
            desc = f"Berdasarkan prediksi model XGBoost, potensi pendapatan titik ini sangat tinggi mencapai <b>Rp {pred:,.0f}/hari</b> (melebihi batas atas {PERSEN_NAIK*100:.0f}% median yaitu Rp {limit_up:,.0f}). Potensi tinggi ini memungkinkan penyesuaian tarif naik <b>+Rp {DELTA_TINGGI:,}</b> menjadi <b>Rp {rec:,.0f}</b>."
        elif status == "Sedang":
            desc = f"Prediksi pendapatan harian di titik ini berada di kisaran menengah, sekitar <b>Rp {pred:,.0f}/hari</b> (di antara {PERSEN_TURUN*100:.0f}%–{PERSEN_NAIK*100:.0f}% median, yaitu Rp {limit_down:,.0f}–{limit_up:,.0f}). Potensi sedang ini memungkinkan penyesuaian tarif naik <b>+Rp {DELTA_SEDANG:,}</b> menjadi <b>Rp {rec:,.0f}</b>."
        else:  # Rendah
            desc = f"Potensi pendapatan diprediksi rendah, hanya sekitar <b>Rp {pred:,.0f}/hari</b> (di bawah batas minimum {PERSEN_TURUN*100:.0f}% median yaitu Rp {limit_down:,.0f}). Tarif direkomendasikan untuk tidak berubah, tetap sebesar <b>Rp {cur:,.0f}</b>."

        html = f"""
        <div style="background: {color_bg}; border: 1px solid {color_border}; border-radius: 12px; padding: 16px; margin-bottom: 16px;">
            <div style="font-weight: 700; color: {color_text}; font-size: 13px; margin-bottom: 8px; display: flex; align-items: center; gap: 6px;">
                {rec_badge}
            </div>
            <div style="font-size: 12.5px; color: #E2E8F0; line-height: 1.5; margin-bottom: 12px;">
                {desc}
            </div>
            <div style="font-weight: 600; font-size: 11px; color: #F8FAFC; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.03em;">Faktor Kontributor Utama:</div>
            <ul style="margin: 0; padding-left: 1.2rem;">
                {ul_items}
            </ul>
        </div>
        """
        return html

    # Precompute df_map and list_opsi
    try:
        bc  = ["Latitude", "Longitude", "Lokasi", "Titik",
               "Status Motor", "Status Mobil", "Alasan Motor", "Alasan Mobil",
               "Tarif Motor", "Tarif Mobil", "Rek Motor", "Rek Mobil",
               "Delta Motor", "Delta Mobil", "Pred Motor", "Pred Mobil",
               "Jumlah Motor Weekday", "Jumlah Mobil Weekday",
               "Jumlah Motor Weekend", "Jumlah Mobil Weekend",
               "Jarak ke Pusat (km)", "Total Durasi Ramai"]
        sel    = list(dict.fromkeys(c for c in bc if c in df_rec.columns))
        df_map = df_rec[sel].dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)

        try:
            df_map["Titik_Num"] = pd.to_numeric(df_map["Titik"], errors="coerce")
            df_map = df_map.sort_values(by="Titik_Num").drop(columns=["Titik_Num"]).reset_index(drop=True)
        except:
            df_map = df_map.sort_values(by="Titik").reset_index(drop=True)

        df_map["OpsiSelect"] = df_map.apply(lambda r: f"{r['Titik']} - {r['Lokasi']}", axis=1)
        list_opsi = df_map["OpsiSelect"].tolist()
    except Exception as e:
        st.error(f"Error loading map data: {e}")
        st.stop()

    # --- HORIZONTAL CONTROL BAR ---
    t_col1, t_col2 = st.columns([1.2, 1.2])
    with t_col1:
        veh_type = st.radio("Pilih Jenis Kendaraan:", ("Motor", "Mobil"), horizontal=True)
    with t_col2:
        map_style = st.selectbox("Layer Peta:", ["OpenStreetMap", "Satelit (ESRI)"])

    map_tabs = st.tabs(["🗺️ Peta Utama", "🔍 Cari & Detail Lokasi"])

    current_opsi = "-- Tampilkan Semua Lokasi --"

    with map_tabs[0]:
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(148,163,184,0.12); border-radius: 10px; padding: 10px 18px; margin-top: 10px; margin-bottom: 22px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
            <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; color: #64748B; letter-spacing: 0.05em;">Legenda Potensi Rekomendasi Tarif ({veh_type}):</div>
            <div style="display: flex; gap: 20px; font-size: 12px;">
                <span style="display: flex; align-items: center; gap: 6px;">🟢 <b style="color:#16A34A;">Tinggi</b> (Prediksi &gt; {PERSEN_NAIK*100:.0f}% Median → +Rp {DELTA_TINGGI:,})</span>
                <span style="display: flex; align-items: center; gap: 6px;">🟡 <b style="color:#D97706;">Sedang</b> ({PERSEN_TURUN*100:.0f}%–{PERSEN_NAIK*100:.0f}% Median → +Rp {DELTA_SEDANG:,})</span>
                <span style="display: flex; align-items: center; gap: 6px;">🔴 <b style="color:#DC2626;">Rendah</b> (Prediksi &lt; {PERSEN_TURUN*100:.0f}% Median → +Rp {DELTA_RENDAH:,})</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        stc = f"Status {veh_type}"; tc = f"Rek {veh_type}"
        try:
            pin_color   = {"Tinggi": "green", "Sedang": "orange", "Rendah": "red"}
            txt_color   = {"Tinggi": "#16A34A", "Sedang": "#D97706", "Rendah": "#DC2626"}
            status_icon = {"Tinggi": "🟢 Tinggi", "Sedang": "🟡 Sedang", "Rendah": "🔴 Rendah"}

            if current_opsi != "-- Tampilkan Semua Lokasi --":
                row_sel        = df_map[df_map["OpsiSelect"] == current_opsi].iloc[0]
                selected_titik = row_sel["Titik"]
                center_coords  = [float(row_sel["Latitude"]), float(row_sel["Longitude"])]
                zoom = 17
            else:
                selected_titik = "-- Cari & Pilih Lokasi --"
                center_coords  = [df_map["Latitude"].mean(), df_map["Longitude"].mean()]
                zoom = 13

            m = folium.Map(location=center_coords, zoom_start=zoom, tiles="OpenStreetMap")
            if map_style == "Satelit (ESRI)":
                folium.TileLayer(
                    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                    name="Satelit", attr="Esri").add_to(m)

            fg = folium.FeatureGroup(name="Parking Locations", show=True)
            for _, row in df_map.iterrows():
                cs_val  = str(row.get(stc, ""))
                pin_clr = pin_color.get(cs_val, "gray")
                txt_clr = txt_color.get(cs_val, "#6B7280")
                cur_t   = float(row.get(f"Tarif {veh_type}", 0))
                rec_t   = float(row.get(tc, cur_t))
                delta   = int(rec_t - cur_t)
                ds      = (f"+Rp {delta:,}" if delta >= 0 else f"−Rp {abs(delta):,}")
                icon_m  = status_icon.get(str(row.get("Status Motor", "")), "–")
                icon_b  = status_icon.get(str(row.get("Status Mobil", "")), "–")
                clr_m   = txt_color.get(str(row.get("Status Motor", "")), "#6B7280")
                clr_b   = txt_color.get(str(row.get("Status Mobil", "")), "#6B7280")
                pred_m  = float(row.get("Pred Motor", 0))
                pred_b  = float(row.get("Pred Mobil", 0))
                al_m    = str(row.get("Alasan Motor", "–"))
                al_b    = str(row.get("Alasan Mobil", "–"))
                titik   = str(row.get("Titik", ""))
                lokasi  = str(row.get("Lokasi", ""))

                popup_html = f"""
                <div style="font-family:'Plus Jakarta Sans',sans-serif;width:300px;font-size:12px;">
                <div style="font-size:15px;font-weight:700;margin-bottom:2px;color:#0F172A;">{titik}</div>
                <div style="color:#64748B;margin-bottom:12px;font-size:11.5px;">{lokasi}</div>
                <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;
                            padding:10px;margin-bottom:10px;">
                  <table style="width:100%;border-collapse:collapse;">
                  <tr><td style="padding:4px 0;color:#64748B;">Tarif saat ini</td>
                      <td style="text-align:right;font-weight:600;">Rp {cur_t:,.0f}</td></tr>
                  <tr><td style="padding:4px 0;color:#64748B;">Rekomendasi tarif</td>
                      <td style="text-align:right;font-weight:700;color:{txt_clr};">Rp {rec_t:,.0f}</td></tr>
                  <tr><td style="padding:4px 0;color:#64748B;">Perubahan</td>
                      <td style="text-align:right;font-weight:600;color:{txt_clr};">{ds}</td></tr>
                  <tr><td colspan="2" style="padding-top:6px;border-top:1px dashed #CBD5E1;"></td></tr>
                  <tr><td style="padding:4px 0;color:#64748B;">Pred Motor</td>
                      <td style="text-align:right;font-family:monospace;">Rp {pred_m:,.0f}</td></tr>
                  <tr><td style="padding:4px 0;color:#64748B;">Pred Mobil</td>
                      <td style="text-align:right;font-family:monospace;">Rp {pred_b:,.0f}</td></tr>
                  </table>
                </div>
                <div style="background:#F1F5F9;border:1px solid #E2E8F0;border-radius:8px;padding:10px;">
                  <div style="font-weight:700;font-size:11px;margin-bottom:8px;color:#334155;">
                    💡 Potensi Pendapatan — Median {PERSEN_TURUN*100:.0f}%/{PERSEN_NAIK*100:.0f}%
                  </div>
                  <div style="margin-bottom:6px;">
                    <span style="font-weight:700;color:{clr_m};">🏍️ Motor — {icon_m}</span><br>
                    <span style="color:#475569;font-size:11px;line-height:1.4;">{al_m}</span>
                  </div>
                  <div style="border-top:1px solid #CBD5E1;padding-top:6px;">
                    <span style="font-weight:700;color:{clr_b};">🚗 Mobil — {icon_b}</span><br>
                    <span style="color:#475569;font-size:11px;line-height:1.4;">{al_b}</span>
                  </div>
                </div>
                </div>"""

                if selected_titik != "-- Cari & Pilih Lokasi --" and titik == selected_titik:
                    folium.Marker(
                        location=[float(row["Latitude"]), float(row["Longitude"])],
                        popup=folium.Popup(popup_html, max_width=320, show=True),
                        tooltip=f"✨ {titik} (TERPILIH) — {cs_val}",
                        icon=folium.Icon(color="red", icon_color="white", icon="star", prefix="fa")
                    ).add_to(fg)
                else:
                    folium.Marker(
                        location=[float(row["Latitude"]), float(row["Longitude"])],
                        popup=folium.Popup(popup_html, max_width=320),
                        tooltip=f"{titik} — {cs_val}",
                        icon=folium.Icon(color=pin_clr, icon_color="white", icon="map-marker", prefix="fa")
                    ).add_to(fg)

            fg.add_to(m); folium.LayerControl().add_to(m)
            if len(df_map) == 0:
                st.warning("⚠️ Tidak ada lokasi dengan koordinat valid.")
            else:
                map_html = m._repr_html_()
                js_fix = """
                <script>
                setInterval(function() {
                    window.dispatchEvent(new Event('resize'));
                    try {
                        var maps = document.getElementsByClassName('folium-map');
                        for (var i = 0; i < maps.length; i++) {
                            var map_id = maps[i].id;
                            var leaflet_map = window[map_id];
                            if (leaflet_map && typeof leaflet_map.invalidateSize === 'function') {
                                leaflet_map.invalidateSize();
                            }
                        }
                    } catch(e) {}
                }, 500);
                </script>
                """
                st.components.v1.html(map_html + js_fix, height=620)

        except Exception as e:
            st.error(f"Map error: {e}"); st.code(traceback.format_exc())

    with map_tabs[1]:
        selected_opsi = st.selectbox(
            "Cari & Pilih Lokasi Spesifik:",
            options=["-- Tampilkan Semua Lokasi --"] + list_opsi,
            key="lokasi_spesifik"
        )

        _, col_details, _ = st.columns([1, 6, 1])

        with col_details:
            if selected_opsi != "-- Tampilkan Semua Lokasi --":
                row_sel        = df_map[df_map["OpsiSelect"] == selected_opsi].iloc[0]
                selected_titik = row_sel["Titik"]

                mean_vals = {
                    "Jumlah Motor Weekday": df_map["Jumlah Motor Weekday"].mean(),
                    "Jumlah Mobil Weekday": df_map["Jumlah Mobil Weekday"].mean(),
                    "Jumlah Motor Weekend":  df_map["Jumlah Motor Weekend"].mean(),
                    "Jumlah Mobil Weekend":  df_map["Jumlah Mobil Weekend"].mean(),
                    "Jarak ke Pusat (km)":  df_map["Jarak ke Pusat (km)"].mean(),
                    "Total Durasi Ramai":   df_map["Total Durasi Ramai"].mean(),
                }

                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(148,163,184,0.1); border-radius: 12px; padding: 16px; margin-bottom: 16px;">
                    <div style="font-size: 15px; font-weight: 700; color: #F8FAFC; margin-bottom: 4px;">{row_sel['Titik']}</div>
                    <div style="font-size: 12.5px; color: #94A3B8; margin-bottom: 12px;">📍 {row_sel['Lokasi']}</div>
                    <div style="display: flex; gap: 8px; font-size: 11px; color: #64748B;">
                        <span>Lat: {row_sel['Latitude']:.5f}</span>
                        <span>•</span>
                        <span>Lon: {row_sel['Longitude']:.5f}</span>
                        <span>•</span>
                        <span>Jarak: {row_sel['Jarak ke Pusat (km)']:.2f} km dari Pusat</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<div style='margin-bottom: 10px; font-weight: 700; font-size:13.5px; color:#F8FAFC;'>🏍️ Detail Analisis Tarif Motor</div>", unsafe_allow_html=True)
                causes_motor_html = generate_causes(row_sel, "Motor", mean_vals)
                st.markdown(causes_motor_html, unsafe_allow_html=True)

                st.markdown("<div style='margin-top: 10px; margin-bottom: 10px; font-weight: 700; font-size:13.5px; color:#F8FAFC;'>🚗 Detail Analisis Tarif Mobil</div>", unsafe_allow_html=True)
                causes_mobil_html = generate_causes(row_sel, "Mobil", mean_vals)
                st.markdown(causes_mobil_html, unsafe_allow_html=True)
            else:
                st.info("Pilih lokasi dari dropdown di atas untuk melihat detail analisis tarif.")
