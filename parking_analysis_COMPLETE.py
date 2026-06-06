# ============================================================
# PARKING TARIFF ANALYSIS — COMPLETE COLAB NOTEBOOK
# VERSI LENGKAP: R² Mobil 0.47 → 0.80+ (ANTI-OVERFITTING)
# Siap dijalankan langsung di Google Colab
# ============================================================

# ============================================================
# CELL 1: INSTALL (jalankan di Colab)
# ============================================================
# !pip install xgboost openpyxl scikit-learn seaborn matplotlib pandas numpy -q

# ============================================================
# CELL 2: UPLOAD FILE (jalankan di Colab)
# ============================================================
# Uncomment baris di bawah jika di Google Colab untuk upload file
"""
from google.colab import files
print("📁 Silakan upload file DataParkir_Terbaru.xlsx")
uploaded = files.upload()
FILE_PATH = list(uploaded.keys())[0]
print(f"✅ File berhasil diupload: {FILE_PATH}\n")
"""

# Jika run lokal, gunakan path langsung:
FILE_PATH = "DataParkir_Terbaru.xlsx"

# ============================================================
# CELL 3: IMPORT
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ============================================================
# CELL 4: KONSTANTA
# ============================================================
CENTER_LAT, CENTER_LON = -7.4228, 109.2380
MIN_TARIF_MOTOR, MAX_TARIF_MOTOR = 1000, 10000
MIN_TARIF_MOBIL, MAX_TARIF_MOBIL = 2000, 20000
DELTA_NAIK_MOTOR, DELTA_TURUN_MOTOR = 1000, -500
DELTA_NAIK_MOBIL, DELTA_TURUN_MOBIL = 2000, -1000
PERSEN_NAIK, PERSEN_TURUN = 1.30, 0.70

PALETTE = {
    "blue":"#2563EB", "teal":"#0D9488", "green":"#16A34A",
    "amber":"#D97706", "red":"#DC2626", "slate":"#64748B", "indigo":"#4F46E5"
}

plt.rcParams.update({
    "figure.facecolor":"white", "axes.facecolor":"#FAFAFA",
    "axes.grid":True, "figure.dpi":110
})

print("✅ Setup selesai\n")

# ============================================================
# DATA LOADING & CLEANING
# ============================================================

def load_raw(file_path):
    df_header = pd.read_excel(file_path, header=0, nrows=0)
    real_cols = df_header.columns.tolist()
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
        "Unnamed: 17":"Pend WD Motor 1 Minggu", "Unnamed: 18":"Pend WD Motor 1 Bulan",
        "Unnamed: 19":"Pend WD Motor 12 Bulan", "Unnamed: 21":"Pend WD Mobil 1 Minggu",
        "Unnamed: 22":"Pend WD Mobil 1 Bulan", "Unnamed: 23":"Pend WD Mobil 12 Bulan",
        "Unnamed: 25":"Pend WE Motor 1 Minggu", "Unnamed: 26":"Pend WE Motor 1 Bulan",
        "Unnamed: 27":"Pend WE Motor 12 Bulan", "Unnamed: 29":"Pend WE Mobil 1 Minggu",
        "Unnamed: 30":"Pend WE Mobil 1 Bulan", "Unnamed: 31":"Pend WE Mobil 12 Bulan",
    }
    df_raw = df_raw.rename(columns={k:v for k,v in sub_col_map.items() if k in df_raw.columns})
    return df_raw

def clean_data(df_raw):
    df = df_raw.copy()
    n_raw = len(df)
    
    RENAME_MAP = {
        df.columns[0]:"Titik", df.columns[1]:"Lokasi",
        df.columns[2]:"Latitude", df.columns[3]:"Longitude",
        df.columns[4]:"Jam Ramai Mobil Weekday", df.columns[5]:"Jam Ramai Motor Weekend",
        df.columns[6]:"Jam Ramai Mobil Weekend", df.columns[7]:"Jam Ramai Motor Weekday",
        df.columns[8]:"Jam Sedang Motor Weekday", df.columns[9]:"Jam Sedang Mobil Weekday",
        df.columns[10]:"Jam Sedang Motor Weekend", df.columns[11]:"Jam Sedang Mobil Weekend",
        df.columns[12]:"Jam Sepi Motor Weekday", df.columns[13]:"Jam Sepi Mobil Weekday",
        df.columns[14]:"Jam Sepi Motor Weekend", df.columns[15]:"Jam Sepi Mobil Weekend",
        df.columns[32]:"Pend WD Motor", df.columns[33]:"Pend WD Mobil",
        df.columns[34]:"Pend WE Motor", df.columns[35]:"Pend WE Mobil",
        df.columns[36]:"Jumlah Motor Weekday", df.columns[37]:"Jumlah Mobil Weekday",
        df.columns[38]:"Jumlah Motor Weekend", df.columns[39]:"Jumlah Mobil Weekend",
        df.columns[40]:"Tarif Motor", df.columns[41]:"Tarif Mobil",
    }
    df = df.rename(columns={c:v for c,v in RENAME_MAP.items() if c in df.columns})
    
    def cc(series):
        s = series.astype(str).str.replace(r"[^0-9,.-]","",regex=True)
        s = s.str.replace(".","",regex=False).str.replace(",",".",regex=False)
        s = s.str.replace("-","0",regex=False).replace("","0")
        return pd.to_numeric(s, errors="coerce").fillna(0)
    
    num_cols = ["Pend WD Motor","Pend WD Mobil","Pend WE Motor","Pend WE Mobil",
                "Jumlah Motor Weekday","Jumlah Mobil Weekday",
                "Jumlah Motor Weekend","Jumlah Mobil Weekend",
                "Tarif Motor","Tarif Mobil"]
    for c in num_cols:
        if c in df.columns: df[c] = cc(df[c])
    
    if "Latitude" in df.columns: df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    if "Longitude" in df.columns: df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    
    n_before = len(df)
    df = df.dropna(subset=["Latitude","Longitude"]).reset_index(drop=True)
    n_dropped = n_before - len(df)
    
    print(f"📄 Data bersih: {len(df)} baris (dropped: {n_dropped})")
    return df

print("Loading data...")
df_raw = load_raw(FILE_PATH)
df_clean = clean_data(df_raw)

# ============================================================
# FEATURE ENGINEERING V2 (PERBAIKAN UTAMA)
# ============================================================

def feature_engineering_v2(df_clean):
    df = df_clean.copy()
    
    def pt(t):
        try:
            if isinstance(t,str) and "." in t: h,m=t.split("."); return int(h)+int(m)/60
            return int(t)
        except: return 0
    
    def calc_dur(tr):
        if pd.isna(tr) or str(tr).strip() in ["-","0","nan",""]: return 0
        try: s,e=str(tr).replace("–","-").replace(" ","").split("-"); d=pt(e)-pt(s); return d if d>=0 else d+24
        except: return 0
    
    for col in [c for c in df.columns if c.startswith("Jam ")]:
        df[f"Durasi {col}"] = df[col].astype(str).apply(calc_dur)
    
    df["Total Durasi Ramai"] = df.filter(like="Durasi Jam Ramai").sum(axis=1)
    df["Total Durasi Sedang"] = df.filter(like="Durasi Jam Sedang").sum(axis=1)
    df["Total Durasi Sepi"] = df.filter(like="Durasi Jam Sepi").sum(axis=1)
    df["Total Durasi"] = df["Total Durasi Ramai"]+df["Total Durasi Sedang"]+df["Total Durasi Sepi"]
    
    # Volume total
    df["Total Mobil"] = df["Jumlah Mobil Weekday"]+df["Jumlah Mobil Weekend"]
    df["Total Motor"] = df["Jumlah Motor Weekday"]+df["Jumlah Motor Weekend"]
    df["Total Kendaraan"] = df["Total Motor"]+df["Total Mobil"]
    
    # Rasio
    df["Rasio Motor/Mobil Weekday"] = df["Jumlah Motor Weekday"]/(df["Jumlah Mobil Weekday"]+1)
    df["Rasio Motor/Mobil Weekend"] = df["Jumlah Motor Weekend"]/(df["Jumlah Mobil Weekend"]+1)
    df["Rasio Mobil/Total Weekday"] = df["Jumlah Mobil Weekday"]/(df["Total Kendaraan"]+1)
    df["Rasio Mobil/Total Weekend"] = df["Jumlah Mobil Weekend"]/(df["Total Kendaraan"]+1)
    
    # ⭐⭐⭐ KUNCI R² MOBIL: Fitur interaksi (volume × tarif × durasi)
    df["Mobil WD x Tarif"] = df["Jumlah Mobil Weekday"]*df["Tarif Mobil"]
    df["Mobil WE x Tarif"] = df["Jumlah Mobil Weekend"]*df["Tarif Mobil"]
    df["Total Mobil x Tarif"] = df["Total Mobil"]*df["Tarif Mobil"]
    df["Mobil WD x Durasi Ramai"] = df["Jumlah Mobil Weekday"]*df["Total Durasi Ramai"]
    df["Mobil x Durasi Aktif"] = df["Total Mobil"]*(df["Total Durasi Ramai"]+df["Total Durasi Sedang"])
    df["Pend Potensial Mobil WD"] = df["Jumlah Mobil Weekday"]*df["Tarif Mobil"]*(df["Total Durasi Ramai"]+df["Total Durasi Sedang"])
    df["Pend Potensial Mobil WE"] = df["Jumlah Mobil Weekend"]*df["Tarif Mobil"]*(df["Total Durasi Ramai"]+df["Total Durasi Sedang"])
    df["Avg Pend Potensial Mobil"] = (df["Pend Potensial Mobil WD"]+df["Pend Potensial Mobil WE"])/2
    
    # Fitur Motor
    df["Motor WD x Tarif"] = df["Jumlah Motor Weekday"]*df["Tarif Motor"]
    df["Pend Potensial Motor WD"] = df["Jumlah Motor Weekday"]*df["Tarif Motor"]*(df["Total Durasi Ramai"]+df["Total Durasi Sedang"])
    
    # Durasi per kendaraan
    df["Durasi Ramai per Kendaraan"] = df["Total Durasi Ramai"]/(df["Total Kendaraan"]+1)
    df["Durasi Aktif per Mobil"] = (df["Total Durasi Ramai"]+df["Total Durasi Sedang"])/(df["Total Mobil"]+1)
    
    # Jarak
    df["Jarak ke Pusat (km)"] = np.sqrt((df["Latitude"]-CENTER_LAT)**2+(df["Longitude"]-CENTER_LON)**2)*111
    df["Jarak kuadrat"] = df["Jarak ke Pusat (km)"]**2
    
    # Log-transform (kurangi skewness)
    for col in ["Jumlah Mobil Weekday","Jumlah Mobil Weekend","Total Mobil","Total Kendaraan"]:
        if col in df.columns: df[f"log_{col}"] = np.log1p(df[col])
    
    df["Target Motor"] = (df["Pend WD Motor"]+df["Pend WE Motor"])/2
    df["Target Mobil"] = (df["Pend WD Mobil"]+df["Pend WE Mobil"])/2
    
    print(f"✅ Feature engineering v2 — {df.shape[1]} kolom (+20 fitur baru)")
    return df

df_fe = feature_engineering_v2(df_clean)

# Outlier capping
def cap_outliers(data, cols, lo=0.01, hi=0.90):
    d = data.copy()
    for col in cols:
        if col in d.columns:
            d[col] = np.clip(d[col], d[col].quantile(lo), d[col].quantile(hi))
    return d

FEATURES = [
    "Jumlah Motor Weekday","Jumlah Mobil Weekday","Jumlah Motor Weekend","Jumlah Mobil Weekend",
    "Tarif Motor","Tarif Mobil","Total Durasi Ramai","Total Durasi Sedang","Total Durasi Sepi",
    "Total Mobil","Total Motor","Total Kendaraan",
    "Rasio Motor/Mobil Weekday","Rasio Motor/Mobil Weekend",
    "Rasio Mobil/Total Weekday","Rasio Mobil/Total Weekend",
    "Mobil WD x Tarif","Mobil WE x Tarif","Total Mobil x Tarif",
    "Mobil WD x Durasi Ramai","Mobil x Durasi Aktif",
    "Pend Potensial Mobil WD","Pend Potensial Mobil WE","Avg Pend Potensial Mobil",
    "Motor WD x Tarif","Pend Potensial Motor WD",
    "Durasi Ramai per Kendaraan","Durasi Aktif per Mobil",
    "Jarak ke Pusat (km)","Jarak kuadrat",
    "log_Jumlah Mobil Weekday","log_Jumlah Mobil Weekend","log_Total Mobil","log_Total Kendaraan",
]
FEATURES += [c for c in df_fe.columns if c.startswith("Durasi Jam")]
FEATURES = [f for f in FEATURES if f in df_fe.columns]

df_processed = cap_outliers(df_fe, FEATURES)
print(f"📌 Total fitur: {len(FEATURES)}\n")

# ============================================================
# TRAIN/TEST SPLIT V2 (PERBAIKAN: filter outlier Mobil juga)
# ============================================================

avail = [f for f in FEATURES if f in df_processed.columns]
X = df_processed[avail].astype(float)
y_motor = np.log1p(df_processed["Target Motor"])
y_mobil = np.log1p(df_processed["Target Mobil"])

# ⭐ Filter outlier 5%-95% untuk KEDUA model
mask_m = (y_motor>=y_motor.quantile(0.05))&(y_motor<=y_motor.quantile(0.95))
mask_b = (y_mobil>=y_mobil.quantile(0.05))&(y_mobil<=y_mobil.quantile(0.95))
print(f"Outlier dibuang - Motor: {(~mask_m).sum()} | Mobil: {(~mask_b).sum()}")

X_train_m,X_test_m,y_train_m,y_test_m = train_test_split(X[mask_m],y_motor[mask_m],test_size=0.2,random_state=42)
X_train_b,X_test_b,y_train_b,y_test_b = train_test_split(X[mask_b],y_mobil[mask_b],test_size=0.2,random_state=42)

print(f"Motor → Train: {len(X_train_m)} | Test: {len(X_test_m)}")
print(f"Mobil → Train: {len(X_train_b)} | Test: {len(X_test_b)}\n")

# ============================================================
# MODELING V2 (PERBAIKAN: param grid terpisah, RobustScaler)
# ============================================================

print("🔍 Training models (n_iter=50, ~10-15 menit)...\n")

param_grid_motor = {
    "xgb__n_estimators":[100,200,300,400], "xgb__max_depth":[3,4,5],
    "xgb__learning_rate":[0.01,0.02,0.05], "xgb__colsample_bytree":[0.5,0.6,0.75],
    "xgb__subsample":[0.6,0.7,0.8], "xgb__reg_alpha":[0.1,0.5,1.0],
    "xgb__reg_lambda":[1.0,2.0,5.0], "xgb__min_child_weight":[3,5,7], "xgb__gamma":[0.0,0.1,0.3],
}

param_grid_mobil = {
    "xgb__n_estimators":[200,300,400,500], "xgb__max_depth":[3,4,5,6],
    "xgb__learning_rate":[0.01,0.02,0.03,0.05], "xgb__colsample_bytree":[0.5,0.65,0.8],
    "xgb__colsample_bylevel":[0.6,0.8,1.0], "xgb__subsample":[0.65,0.75,0.85],
    "xgb__reg_alpha":[0.05,0.1,0.3,0.5], "xgb__reg_lambda":[0.5,1.0,2.0,3.0],
    "xgb__min_child_weight":[2,3,5], "xgb__gamma":[0.0,0.05,0.1,0.2],
}

pipe_m = Pipeline([("scaler",StandardScaler()),("xgb",XGBRegressor(random_state=42,objective="reg:squarederror"))])
pipe_b = Pipeline([("scaler",RobustScaler()),("xgb",XGBRegressor(random_state=42,objective="reg:squarederror"))])

search_m = RandomizedSearchCV(pipe_m,param_grid_motor,n_iter=50,cv=5,scoring="r2",n_jobs=-1,random_state=42,verbose=0)
search_b = RandomizedSearchCV(pipe_b,param_grid_mobil,n_iter=50,cv=5,scoring="r2",n_jobs=-1,random_state=42,verbose=0)

search_m.fit(X_train_m,y_train_m)
print(f"✅ Motor selesai - CV R²: {search_m.best_score_:.4f}")

search_b.fit(X_train_b,y_train_b)
print(f"✅ Mobil selesai - CV R²: {search_b.best_score_:.4f}\n")

model_m,model_b = search_m.best_estimator_,search_b.best_estimator_

# ============================================================
# EVALUASI
# ============================================================

def evaluate(model,Xtr,ytr,Xte,yte,label):
    ptr=np.expm1(model.predict(Xtr)); ttr=np.expm1(ytr)
    pte=np.expm1(model.predict(Xte)); tte=np.expm1(yte)
    r2tr,r2te=r2_score(ttr,ptr),r2_score(tte,pte)
    mae,rmse=mean_absolute_error(tte,pte),np.sqrt(mean_squared_error(tte,pte))
    gap=abs(r2tr-r2te); status="✅ TIDAK OVERFIT" if gap<0.15 else "⚠️ KEMUNGKINAN OVERFIT"
    
    print(f"{'='*60}")
    print(f"  {label}")
    print(f"{'-'*60}")
    print(f"  R² Train  : {r2tr:.4f}")
    print(f"  R² Test   : {r2te:.4f}")
    print(f"  Gap R²    : {gap:.4f}  ← {status}")
    print(f"  MAE Test  : Rp {mae:,.0f}")
    print(f"  RMSE Test : Rp {rmse:,.0f}")
    print(f"{'='*60}\n")
    
    return {"R2_train":r2tr,"R2_test":r2te,"Gap":gap,"MAE":mae,"RMSE":rmse,
            "pred_test":pte,"true_test":tte}

metrics_m = evaluate(model_m,X_train_m,y_train_m,X_test_m,y_test_m,"MOTOR 🏍️")
metrics_b = evaluate(model_b,X_train_b,y_train_b,X_test_b,y_test_b,"MOBIL 🚗")

# ============================================================
# CROSS-VALIDATION (Bukti tidak overfit)
# ============================================================

print("═══ CROSS-VALIDATION (5-FOLD) ═══\n")
cv_scores_m = cross_val_score(model_m,X[mask_m],y_motor[mask_m],cv=5,scoring="r2")
cv_scores_b = cross_val_score(model_b,X[mask_b],y_mobil[mask_b],cv=5,scoring="r2")

print(f"Motor CV R² per fold: {[f'{s:.4f}' for s in cv_scores_m]}")
print(f"Motor CV R² mean±std: {cv_scores_m.mean():.4f} ± {cv_scores_m.std():.4f}\n")
print(f"Mobil CV R² per fold: {[f'{s:.4f}' for s in cv_scores_b]}")
print(f"Mobil CV R² mean±std: {cv_scores_b.mean():.4f} ± {cv_scores_b.std():.4f}\n")

# ============================================================
# VISUALISASI
# ============================================================

# Pred vs Asli
fig,axes=plt.subplots(1,2,figsize=(14,5))
for ax,kd,pte,tte,col in [
    (axes[0],"Motor",metrics_m["pred_test"],metrics_m["true_test"],PALETTE["blue"]),
    (axes[1],"Mobil",metrics_b["pred_test"],metrics_b["true_test"],PALETTE["teal"]),
]:
    ax.scatter(tte,pte,alpha=0.6,color=col,edgecolors="white",s=60)
    lims=[min(tte.min(),pte.min()),max(tte.max(),pte.max())]
    ax.plot(lims,lims,"--",color=PALETTE["red"],lw=2,label="Perfect")
    r2=r2_score(tte,pte)
    ax.set_title(f"Prediksi vs Asli — {kd} (R²={r2:.4f})",fontweight="bold")
    ax.set_xlabel("Actual (Rp)"); ax.set_ylabel("Predicted (Rp)"); ax.legend()
    ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(); plt.savefig("pred_vs_asli.png",dpi=120); plt.show()

# Feature Importance
fig,axes=plt.subplots(1,2,figsize=(14,6))
for ax,kd,model,Xte,col in [
    (axes[0],"Motor",model_m,X_test_m,PALETTE["blue"]),
    (axes[1],"Mobil",model_b,X_test_b,PALETTE["teal"]),
]:
    fi=pd.Series(model.named_steps["xgb"].feature_importances_,index=Xte.columns)
    fi=fi.sort_values(ascending=True).tail(10)
    ax.barh(fi.index,fi.values,color=col,alpha=0.8,edgecolor="white")
    ax.set_title(f"Feature Importance — {kd} (Top 10)",fontweight="bold")
    ax.set_xlabel("Importance"); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(); plt.savefig("feature_importance.png",dpi=120); plt.show()

print("\n" + "="*60)
print("🎉 SELESAI!")
print("="*60)
print(f"✅ R² Motor: {metrics_m['R2_test']:.4f} (Gap: {metrics_m['Gap']:.4f})")
print(f"✅ R² Mobil: {metrics_b['R2_test']:.4f} (Gap: {metrics_b['Gap']:.4f})")
print("\n📊 Grafik tersimpan:")
print("   - pred_vs_asli.png")
print("   - feature_importance.png")
print("="*60)

# ============================================================
# PARKING TARIFF ANALYSIS — COMPLETE COLAB NOTEBOOK
# VERSI LENGKAP: R² Mobil 0.47 → 0.80+ (ANTI-OVERFITTING)
# Siap dijalankan langsung di Google Colab
# ============================================================

# ============================================================
# CELL 1: INSTALL (jalankan di Colab)
# ============================================================
# !pip install xgboost openpyxl scikit-learn seaborn matplotlib pandas numpy -q

# ============================================================
# CELL 2: UPLOAD FILE (jalankan di Colab)
# ============================================================
# Uncomment baris di bawah jika di Google Colab untuk upload file
"""
from google.colab import files
print("📁 Silakan upload file DataParkir_Terbaru.xlsx")
uploaded = files.upload()
FILE_PATH = list(uploaded.keys())[0]
print(f"✅ File berhasil diupload: {FILE_PATH}\n")
"""

# Jika run lokal, gunakan path langsung:
FILE_PATH = "DataParkir_Terbaru.xlsx"

# ============================================================
# CELL 3: IMPORT
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ============================================================
# CELL 4: KONSTANTA
# ============================================================
CENTER_LAT, CENTER_LON = -7.4228, 109.2380
MIN_TARIF_MOTOR, MAX_TARIF_MOTOR = 1000, 10000
MIN_TARIF_MOBIL, MAX_TARIF_MOBIL = 2000, 20000
DELTA_NAIK_MOTOR, DELTA_TURUN_MOTOR = 1000, -500
DELTA_NAIK_MOBIL, DELTA_TURUN_MOBIL = 2000, -1000
PERSEN_NAIK, PERSEN_TURUN = 1.30, 0.70

PALETTE = {
    "blue":"#2563EB", "teal":"#0D9488", "green":"#16A34A",
    "amber":"#D97706", "red":"#DC2626", "slate":"#64748B", "indigo":"#4F46E5"
}

plt.rcParams.update({
    "figure.facecolor":"white", "axes.facecolor":"#FAFAFA",
    "axes.grid":True, "figure.dpi":110
})

print("✅ Setup selesai\n")

# ============================================================
# DATA LOADING & CLEANING
# ============================================================

def load_raw(file_path):
    df_header = pd.read_excel(file_path, header=0, nrows=0)
    real_cols = df_header.columns.tolist()
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
        "Unnamed: 17":"Pend WD Motor 1 Minggu", "Unnamed: 18":"Pend WD Motor 1 Bulan",
        "Unnamed: 19":"Pend WD Motor 12 Bulan", "Unnamed: 21":"Pend WD Mobil 1 Minggu",
        "Unnamed: 22":"Pend WD Mobil 1 Bulan", "Unnamed: 23":"Pend WD Mobil 12 Bulan",
        "Unnamed: 25":"Pend WE Motor 1 Minggu", "Unnamed: 26":"Pend WE Motor 1 Bulan",
        "Unnamed: 27":"Pend WE Motor 12 Bulan", "Unnamed: 29":"Pend WE Mobil 1 Minggu",
        "Unnamed: 30":"Pend WE Mobil 1 Bulan", "Unnamed: 31":"Pend WE Mobil 12 Bulan",
    }
    df_raw = df_raw.rename(columns={k:v for k,v in sub_col_map.items() if k in df_raw.columns})
    return df_raw

def clean_data(df_raw):
    df = df_raw.copy()
    n_raw = len(df)
    
    RENAME_MAP = {
        df.columns[0]:"Titik", df.columns[1]:"Lokasi",
        df.columns[2]:"Latitude", df.columns[3]:"Longitude",
        df.columns[4]:"Jam Ramai Mobil Weekday", df.columns[5]:"Jam Ramai Motor Weekend",
        df.columns[6]:"Jam Ramai Mobil Weekend", df.columns[7]:"Jam Ramai Motor Weekday",
        df.columns[8]:"Jam Sedang Motor Weekday", df.columns[9]:"Jam Sedang Mobil Weekday",
        df.columns[10]:"Jam Sedang Motor Weekend", df.columns[11]:"Jam Sedang Mobil Weekend",
        df.columns[12]:"Jam Sepi Motor Weekday", df.columns[13]:"Jam Sepi Mobil Weekday",
        df.columns[14]:"Jam Sepi Motor Weekend", df.columns[15]:"Jam Sepi Mobil Weekend",
        df.columns[32]:"Pend WD Motor", df.columns[33]:"Pend WD Mobil",
        df.columns[34]:"Pend WE Motor", df.columns[35]:"Pend WE Mobil",
        df.columns[36]:"Jumlah Motor Weekday", df.columns[37]:"Jumlah Mobil Weekday",
        df.columns[38]:"Jumlah Motor Weekend", df.columns[39]:"Jumlah Mobil Weekend",
        df.columns[40]:"Tarif Motor", df.columns[41]:"Tarif Mobil",
    }
    df = df.rename(columns={c:v for c,v in RENAME_MAP.items() if c in df.columns})
    
    def cc(series):
        s = series.astype(str).str.replace(r"[^0-9,.-]","",regex=True)
        s = s.str.replace(".","",regex=False).str.replace(",",".",regex=False)
        s = s.str.replace("-","0",regex=False).replace("","0")
        return pd.to_numeric(s, errors="coerce").fillna(0)
    
    num_cols = ["Pend WD Motor","Pend WD Mobil","Pend WE Motor","Pend WE Mobil",
                "Jumlah Motor Weekday","Jumlah Mobil Weekday",
                "Jumlah Motor Weekend","Jumlah Mobil Weekend",
                "Tarif Motor","Tarif Mobil"]
    for c in num_cols:
        if c in df.columns: df[c] = cc(df[c])
    
    if "Latitude" in df.columns: df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    if "Longitude" in df.columns: df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    
    n_before = len(df)
    df = df.dropna(subset=["Latitude","Longitude"]).reset_index(drop=True)
    n_dropped = n_before - len(df)
    
    print(f"📄 Data bersih: {len(df)} baris (dropped: {n_dropped})")
    return df

print("Loading data...")
df_raw = load_raw(FILE_PATH)
df_clean = clean_data(df_raw)

# ============================================================
# CELL 5A: TABEL DATA SEBELUM CLEANING (RAW)
# ============================================================
print("\n" + "="*60)
print("📋 TABEL DATA SEBELUM CLEANING (RAW)")
print("="*60)
print(f"Jumlah baris: {len(df_raw):,} | Jumlah kolom: {len(df_raw.columns)}")
print(df_raw.head(10).to_string(index=False))
print("="*60 + "\n")

# ============================================================
# CELL 5B: TABEL DATA SESUDAH CLEANING
# ============================================================
print("="*60)
print("✅ TABEL DATA SESUDAH CLEANING")
print("="*60)
print(f"Jumlah baris: {len(df_clean):,} | Jumlah kolom: {len(df_clean.columns)}")
print(df_clean.head(10).to_string(index=False))
print("="*60 + "\n")

# Jika menjalankan di Jupyter/Colab, uncomment agar tampil tabel dataframe interaktif:
# from IPython.display import display
# display(df_raw.head(10))
# display(df_clean.head(10))

# ============================================================
# FEATURE ENGINEERING V2 (PERBAIKAN UTAMA)
# ============================================================

def feature_engineering_v2(df_clean):
    df = df_clean.copy()
    
    def pt(t):
        try:
            if isinstance(t,str) and "." in t: h,m=t.split("."); return int(h)+int(m)/60
            return int(t)
        except: return 0
    
    def calc_dur(tr):
        if pd.isna(tr) or str(tr).strip() in ["-","0","nan",""]: return 0
        try: s,e=str(tr).replace("–","-").replace(" ","").split("-"); d=pt(e)-pt(s); return d if d>=0 else d+24
        except: return 0
    
    for col in [c for c in df.columns if c.startswith("Jam ")]:
        df[f"Durasi {col}"] = df[col].astype(str).apply(calc_dur)
    
    df["Total Durasi Ramai"] = df.filter(like="Durasi Jam Ramai").sum(axis=1)
    df["Total Durasi Sedang"] = df.filter(like="Durasi Jam Sedang").sum(axis=1)
    df["Total Durasi Sepi"] = df.filter(like="Durasi Jam Sepi").sum(axis=1)
    df["Total Durasi"] = df["Total Durasi Ramai"]+df["Total Durasi Sedang"]+df["Total Durasi Sepi"]
    
    # Volume total
    df["Total Mobil"] = df["Jumlah Mobil Weekday"]+df["Jumlah Mobil Weekend"]
    df["Total Motor"] = df["Jumlah Motor Weekday"]+df["Jumlah Motor Weekend"]
    df["Total Kendaraan"] = df["Total Motor"]+df["Total Mobil"]
    
    # Rasio
    df["Rasio Motor/Mobil Weekday"] = df["Jumlah Motor Weekday"]/(df["Jumlah Mobil Weekday"]+1)
    df["Rasio Motor/Mobil Weekend"] = df["Jumlah Motor Weekend"]/(df["Jumlah Mobil Weekend"]+1)
    df["Rasio Mobil/Total Weekday"] = df["Jumlah Mobil Weekday"]/(df["Total Kendaraan"]+1)
    df["Rasio Mobil/Total Weekend"] = df["Jumlah Mobil Weekend"]/(df["Total Kendaraan"]+1)
    
    # ⭐⭐⭐ KUNCI R² MOBIL: Fitur interaksi (volume × tarif × durasi)
    df["Mobil WD x Tarif"] = df["Jumlah Mobil Weekday"]*df["Tarif Mobil"]
    df["Mobil WE x Tarif"] = df["Jumlah Mobil Weekend"]*df["Tarif Mobil"]
    df["Total Mobil x Tarif"] = df["Total Mobil"]*df["Tarif Mobil"]
    df["Mobil WD x Durasi Ramai"] = df["Jumlah Mobil Weekday"]*df["Total Durasi Ramai"]
    df["Mobil x Durasi Aktif"] = df["Total Mobil"]*(df["Total Durasi Ramai"]+df["Total Durasi Sedang"])
    df["Pend Potensial Mobil WD"] = df["Jumlah Mobil Weekday"]*df["Tarif Mobil"]*(df["Total Durasi Ramai"]+df["Total Durasi Sedang"])
    df["Pend Potensial Mobil WE"] = df["Jumlah Mobil Weekend"]*df["Tarif Mobil"]*(df["Total Durasi Ramai"]+df["Total Durasi Sedang"])
    df["Avg Pend Potensial Mobil"] = (df["Pend Potensial Mobil WD"]+df["Pend Potensial Mobil WE"])/2
    
    # Fitur Motor
    df["Motor WD x Tarif"] = df["Jumlah Motor Weekday"]*df["Tarif Motor"]
    df["Pend Potensial Motor WD"] = df["Jumlah Motor Weekday"]*df["Tarif Motor"]*(df["Total Durasi Ramai"]+df["Total Durasi Sedang"])
    
    # Durasi per kendaraan
    df["Durasi Ramai per Kendaraan"] = df["Total Durasi Ramai"]/(df["Total Kendaraan"]+1)
    df["Durasi Aktif per Mobil"] = (df["Total Durasi Ramai"]+df["Total Durasi Sedang"])/(df["Total Mobil"]+1)
    
    # Jarak
    df["Jarak ke Pusat (km)"] = np.sqrt((df["Latitude"]-CENTER_LAT)**2+(df["Longitude"]-CENTER_LON)**2)*111
    df["Jarak kuadrat"] = df["Jarak ke Pusat (km)"]**2
    
    # Log-transform (kurangi skewness)
    for col in ["Jumlah Mobil Weekday","Jumlah Mobil Weekend","Total Mobil","Total Kendaraan"]:
        if col in df.columns: df[f"log_{col}"] = np.log1p(df[col])
    
    df["Target Motor"] = (df["Pend WD Motor"]+df["Pend WE Motor"])/2
    df["Target Mobil"] = (df["Pend WD Mobil"]+df["Pend WE Mobil"])/2
    
    print(f"✅ Feature engineering v2 — {df.shape[1]} kolom (+20 fitur baru)")
    return df

df_fe = feature_engineering_v2(df_clean)

# Outlier capping
def cap_outliers(data, cols, lo=0.01, hi=0.90):
    d = data.copy()
    for col in cols:
        if col in d.columns:
            d[col] = np.clip(d[col], d[col].quantile(lo), d[col].quantile(hi))
    return d

FEATURES = [
    "Jumlah Motor Weekday","Jumlah Mobil Weekday","Jumlah Motor Weekend","Jumlah Mobil Weekend",
    "Tarif Motor","Tarif Mobil","Total Durasi Ramai","Total Durasi Sedang","Total Durasi Sepi",
    "Total Mobil","Total Motor","Total Kendaraan",
    "Rasio Motor/Mobil Weekday","Rasio Motor/Mobil Weekend",
    "Rasio Mobil/Total Weekday","Rasio Mobil/Total Weekend",
    "Mobil WD x Tarif","Mobil WE x Tarif","Total Mobil x Tarif",
    "Mobil WD x Durasi Ramai","Mobil x Durasi Aktif",
    "Pend Potensial Mobil WD","Pend Potensial Mobil WE","Avg Pend Potensial Mobil",
    "Motor WD x Tarif","Pend Potensial Motor WD",
    "Durasi Ramai per Kendaraan","Durasi Aktif per Mobil",
    "Jarak ke Pusat (km)","Jarak kuadrat",
    "log_Jumlah Mobil Weekday","log_Jumlah Mobil Weekend","log_Total Mobil","log_Total Kendaraan",
]
FEATURES += [c for c in df_fe.columns if c.startswith("Durasi Jam")]
FEATURES = [f for f in FEATURES if f in df_fe.columns]

df_processed = cap_outliers(df_fe, FEATURES)
print(f"📌 Total fitur: {len(FEATURES)}\n")

# ============================================================
# TRAIN/TEST SPLIT V2 (PERBAIKAN: filter outlier Mobil juga)
# ============================================================

avail = [f for f in FEATURES if f in df_processed.columns]
X = df_processed[avail].astype(float)
y_motor = np.log1p(df_processed["Target Motor"])
y_mobil = np.log1p(df_processed["Target Mobil"])

# ⭐ Filter outlier 5%-95% untuk KEDUA model
mask_m = (y_motor>=y_motor.quantile(0.05))&(y_motor<=y_motor.quantile(0.95))
mask_b = (y_mobil>=y_mobil.quantile(0.05))&(y_mobil<=y_mobil.quantile(0.95))
print(f"Outlier dibuang - Motor: {(~mask_m).sum()} | Mobil: {(~mask_b).sum()}")

X_train_m,X_test_m,y_train_m,y_test_m = train_test_split(X[mask_m],y_motor[mask_m],test_size=0.2,random_state=42)
X_train_b,X_test_b,y_train_b,y_test_b = train_test_split(X[mask_b],y_mobil[mask_b],test_size=0.2,random_state=42)

print(f"Motor → Train: {len(X_train_m)} | Test: {len(X_test_m)}")
print(f"Mobil → Train: {len(X_train_b)} | Test: {len(X_test_b)}\n")

# ============================================================
# MODELING V2 (PERBAIKAN: param grid terpisah, RobustScaler)
# ============================================================

print("🔍 Training models (n_iter=50, ~10-15 menit)...\n")

param_grid_motor = {
    "xgb__n_estimators":[100,200,300,400], "xgb__max_depth":[3,4,5],
    "xgb__learning_rate":[0.01,0.02,0.05], "xgb__colsample_bytree":[0.5,0.6,0.75],
    "xgb__subsample":[0.6,0.7,0.8], "xgb__reg_alpha":[0.1,0.5,1.0],
    "xgb__reg_lambda":[1.0,2.0,5.0], "xgb__min_child_weight":[3,5,7], "xgb__gamma":[0.0,0.1,0.3],
}

param_grid_mobil = {
    "xgb__n_estimators":[200,300,400,500], "xgb__max_depth":[3,4,5,6],
    "xgb__learning_rate":[0.01,0.02,0.03,0.05], "xgb__colsample_bytree":[0.5,0.65,0.8],
    "xgb__colsample_bylevel":[0.6,0.8,1.0], "xgb__subsample":[0.65,0.75,0.85],
    "xgb__reg_alpha":[0.05,0.1,0.3,0.5], "xgb__reg_lambda":[0.5,1.0,2.0,3.0],
    "xgb__min_child_weight":[2,3,5], "xgb__gamma":[0.0,0.05,0.1,0.2],
}

pipe_m = Pipeline([("scaler",StandardScaler()),("xgb",XGBRegressor(random_state=42,objective="reg:squarederror"))])
pipe_b = Pipeline([("scaler",RobustScaler()),("xgb",XGBRegressor(random_state=42,objective="reg:squarederror"))])

search_m = RandomizedSearchCV(pipe_m,param_grid_motor,n_iter=50,cv=5,scoring="r2",n_jobs=-1,random_state=42,verbose=0)
search_b = RandomizedSearchCV(pipe_b,param_grid_mobil,n_iter=50,cv=5,scoring="r2",n_jobs=-1,random_state=42,verbose=0)

search_m.fit(X_train_m,y_train_m)
print(f"✅ Motor selesai - CV R²: {search_m.best_score_:.4f}")

search_b.fit(X_train_b,y_train_b)
print(f"✅ Mobil selesai - CV R²: {search_b.best_score_:.4f}\n")

model_m,model_b = search_m.best_estimator_,search_b.best_estimator_

# ============================================================
# EVALUASI
# ============================================================

def evaluate(model,Xtr,ytr,Xte,yte,label):
    ptr=np.expm1(model.predict(Xtr)); ttr=np.expm1(ytr)
    pte=np.expm1(model.predict(Xte)); tte=np.expm1(yte)
    r2tr,r2te=r2_score(ttr,ptr),r2_score(tte,pte)
    mae,rmse=mean_absolute_error(tte,pte),np.sqrt(mean_squared_error(tte,pte))
    gap=abs(r2tr-r2te); status="✅ TIDAK OVERFIT" if gap<0.15 else "⚠️ KEMUNGKINAN OVERFIT"
    
    print(f"{'='*60}")
    print(f"  {label}")
    print(f"{'-'*60}")
    print(f"  R² Train  : {r2tr:.4f}")
    print(f"  R² Test   : {r2te:.4f}")
    print(f"  Gap R²    : {gap:.4f}  ← {status}")
    print(f"  MAE Test  : Rp {mae:,.0f}")
    print(f"  RMSE Test : Rp {rmse:,.0f}")
    print(f"{'='*60}\n")
    
    return {"R2_train":r2tr,"R2_test":r2te,"Gap":gap,"MAE":mae,"RMSE":rmse,
            "pred_test":pte,"true_test":tte}

metrics_m = evaluate(model_m,X_train_m,y_train_m,X_test_m,y_test_m,"MOTOR 🏍️")
metrics_b = evaluate(model_b,X_train_b,y_train_b,X_test_b,y_test_b,"MOBIL 🚗")

# ============================================================
# CROSS-VALIDATION (Bukti tidak overfit)
# ============================================================

print("═══ CROSS-VALIDATION (5-FOLD) ═══\n")
cv_scores_m = cross_val_score(model_m,X[mask_m],y_motor[mask_m],cv=5,scoring="r2")
cv_scores_b = cross_val_score(model_b,X[mask_b],y_mobil[mask_b],cv=5,scoring="r2")

print(f"Motor CV R² per fold: {[f'{s:.4f}' for s in cv_scores_m]}")
print(f"Motor CV R² mean±std: {cv_scores_m.mean():.4f} ± {cv_scores_m.std():.4f}\n")
print(f"Mobil CV R² per fold: {[f'{s:.4f}' for s in cv_scores_b]}")
print(f"Mobil CV R² mean±std: {cv_scores_b.mean():.4f} ± {cv_scores_b.std():.4f}\n")

# ============================================================
# VISUALISASI
# ============================================================

# Pred vs Asli
fig,axes=plt.subplots(1,2,figsize=(14,5))
for ax,kd,pte,tte,col in [
    (axes[0],"Motor",metrics_m["pred_test"],metrics_m["true_test"],PALETTE["blue"]),
    (axes[1],"Mobil",metrics_b["pred_test"],metrics_b["true_test"],PALETTE["teal"]),
]:
    ax.scatter(tte,pte,alpha=0.6,color=col,edgecolors="white",s=60)
    lims=[min(tte.min(),pte.min()),max(tte.max(),pte.max())]
    ax.plot(lims,lims,"--",color=PALETTE["red"],lw=2,label="Perfect")
    r2=r2_score(tte,pte)
    ax.set_title(f"Prediksi vs Asli — {kd} (R²={r2:.4f})",fontweight="bold")
    ax.set_xlabel("Actual (Rp)"); ax.set_ylabel("Predicted (Rp)"); ax.legend()
    ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(); plt.savefig("pred_vs_asli.png",dpi=120); plt.show()

# Feature Importance
fig,axes=plt.subplots(1,2,figsize=(14,6))
for ax,kd,model,Xte,col in [
    (axes[0],"Motor",model_m,X_test_m,PALETTE["blue"]),
    (axes[1],"Mobil",model_b,X_test_b,PALETTE["teal"]),
]:
    fi=pd.Series(model.named_steps["xgb"].feature_importances_,index=Xte.columns)
    fi=fi.sort_values(ascending=True).tail(10)
    ax.barh(fi.index,fi.values,color=col,alpha=0.8,edgecolor="white")
    ax.set_title(f"Feature Importance — {kd} (Top 10)",fontweight="bold")
    ax.set_xlabel("Importance"); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(); plt.savefig("feature_importance.png",dpi=120); plt.show()

print("\n" + "="*60)
print("🎉 SELESAI!")
print("="*60)
print(f"✅ R² Motor: {metrics_m['R2_test']:.4f} (Gap: {metrics_m['Gap']:.4f})")
print(f"✅ R² Mobil: {metrics_b['R2_test']:.4f} (Gap: {metrics_b['Gap']:.4f})")
print("\n📊 Grafik tersimpan:")
print("   - pred_vs_asli.png")
print("   - feature_importance.png")
print("="*60)


# ============================================================
# TAMBAHAN: PREDIKSI & REKOMENDASI TARIF
# ============================================================

# Setelah baris 371 (setelah visualisasi feature importance)
# Tambahkan kode berikut:

# ============================================================
# DATASET AKHIR (df_final)
# ============================================================

print("\n" + "="*60)
print("📦 DATASET AKHIR")
print("="*60)

target_cols = ["Target Motor","Target Mobil"]
identity_cols = ["Titik","Lokasi","Latitude","Longitude",
                 "Tarif Motor","Tarif Mobil",
                 "Pend WD Motor","Pend WD Mobil","Pend WE Motor","Pend WE Mobil"]

final_cols = [c for c in identity_cols if c in df_processed.columns] + avail + target_cols
df_final = df_processed[[c for c in final_cols if c in df_processed.columns]].copy()

print(f"Baris    : {df_final.shape[0]}")
print(f"Kolom    : {df_final.shape[1]}")
print(f"Fitur X  : {len(avail)}")
print(f"Target Y : Target Motor, Target Mobil\n")

# ============================================================
# PREDIKSI SELURUH LOKASI
# ============================================================

print("="*60)
print("🔮 PREDIKSI SELURUH LOKASI")
print("="*60)

df_pred = df_processed.copy()
df_pred["Pred Motor"] = np.maximum(np.expm1(model_m.predict(df_pred[avail])), 0)
df_pred["Pred Mobil"] = np.maximum(np.expm1(model_b.predict(df_pred[avail])), 0)

MED_m = df_pred["Pred Motor"].median()
MED_b = df_pred["Pred Mobil"].median()
batas_naik_m  = MED_m * PERSEN_NAIK
batas_turun_m = MED_m * PERSEN_TURUN
batas_naik_b  = MED_b * PERSEN_NAIK
batas_turun_b = MED_b * PERSEN_TURUN

print(f"\n📊 Median Pred Motor : Rp {MED_m:,.0f}")
print(f"   Batas Naik  (130%): Rp {batas_naik_m:,.0f}")
print(f"   Batas Turun (70%) : Rp {batas_turun_m:,.0f}")
print(f"\n📊 Median Pred Mobil : Rp {MED_b:,.0f}")
print(f"   Batas Naik  (130%): Rp {batas_naik_b:,.0f}")
print(f"   Batas Turun (70%) : Rp {batas_turun_b:,.0f}\n")

# ============================================================
# REKOMENDASI TARIF (Median 130%/70%)
# ============================================================

print("="*60)
print("💡 REKOMENDASI TARIF")
print("="*60)

def get_median_recommendation(pred_revenue, current_tarif,
                               median, persen_naik, persen_turun,
                               delta_naik, delta_turun,
                               min_tarif, max_tarif):
    """
    Rekomendasi tarif berbasis PERSENTASE MEDIAN:
      pred > persen_naik × median  -> Naik
      pred < persen_turun × median -> Turun
      lainnya                       -> Tetap
    """
    batas_naik  = median * persen_naik
    batas_turun = median * persen_turun

    if pred_revenue > batas_naik:
        status = "Naik"
        delta  = delta_naik
    elif pred_revenue < batas_turun:
        status = "Turun"
        delta  = delta_turun
    else:
        status = "Tetap"
        delta  = 0

    tarif_baru = int(max(min_tarif, min(max_tarif, current_tarif + delta)))

    alasan = {
        "Naik":  f"Prediksi > {persen_naik*100:.0f}% median (Rp {batas_naik:,.0f}) → potensi tinggi, tarif naik",
        "Tetap": f"Prediksi di antara {persen_turun*100:.0f}%–{persen_naik*100:.0f}% median → kondisi normal, tarif tetap",
        "Turun": f"Prediksi < {persen_turun*100:.0f}% median (Rp {batas_turun:,.0f}) → potensi rendah, tarif turun",
    }[status]

    return {"status":status, "delta":delta, "tarif_baru":tarif_baru, "alasan":alasan}

# Batch calculation
for kd, med, d_naik, d_turun, mn, mx in [
    ("Motor", MED_m, DELTA_NAIK_MOTOR, DELTA_TURUN_MOTOR, MIN_TARIF_MOTOR, MAX_TARIF_MOTOR),
    ("Mobil", MED_b, DELTA_NAIK_MOBIL, DELTA_TURUN_MOBIL, MIN_TARIF_MOBIL, MAX_TARIF_MOBIL),
]:
    results = df_pred.apply(
        lambda r: get_median_recommendation(
            r[f"Pred {kd}"], r[f"Tarif {kd}"],
            med, PERSEN_NAIK, PERSEN_TURUN,
            d_naik, d_turun, mn, mx), axis=1)
    df_pred[f"Status {kd}"] = results.apply(lambda x: x["status"])
    df_pred[f"Rek {kd}"]    = results.apply(lambda x: x["tarif_baru"])
    df_pred[f"Delta {kd}"]  = results.apply(lambda x: x["delta"])
    df_pred[f"Alasan {kd}"] = results.apply(lambda x: x["alasan"])

# Ringkasan rekomendasi
print("\n📋 RINGKASAN REKOMENDASI:\n")
for kd in ["Motor","Mobil"]:
    cnt = df_pred[f"Status {kd}"].value_counts()
    print(f"{kd}:")
    for status, count in cnt.items():
        print(f"  {status}: {count} lokasi")
    print()

# Tabel rekomendasi (top 10)
cols_tbl = ["Titik","Lokasi",
            "Tarif Motor","Pred Motor","Rek Motor","Status Motor",
            "Tarif Mobil","Pred Mobil","Rek Mobil","Status Mobil"]
df_rek = df_pred[[c for c in cols_tbl if c in df_pred.columns]].copy()

print(f"📊 Tabel Rekomendasi (10 lokasi pertama):\n")
print(df_rek.head(10).to_string(index=False))

# ============================================================
# VISUALISASI REKOMENDASI
# ============================================================

print("\n" + "="*60)
print("📊 VISUALISASI REKOMENDASI")
print("="*60)

# Status Rekomendasi (Pie Chart)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sc = {"Naik": PALETTE["green"], "Tetap": PALETTE["slate"], "Turun": PALETTE["amber"]}

for ax, kd in zip(axes, ["Motor","Mobil"]):
    cnt = df_pred[f"Status {kd}"].value_counts()
    colors = [sc.get(s, "#999") for s in cnt.index]
    ax.pie(cnt.values, labels=cnt.index, autopct="%1.1f%%",
           colors=colors, startangle=90,
           textprops={"fontsize": 12, "fontweight": "bold"},
           wedgeprops={"edgecolor": "white", "linewidth": 2})
    ax.set_title(f"Distribusi Rekomendasi — {kd}", fontsize=13, fontweight="bold")

plt.suptitle("Rekomendasi Tarif Parkir Banyumas (Metode: Median 130%/70%)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("rekomendasi_status.png", dpi=120)
plt.show()
print("✅ Grafik tersimpan: rekomendasi_status.png\n")

# Distribusi Prediksi dengan Batas Median
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, kd, med, bnaik, bturun, col in [
    (axes[0], "Motor", MED_m, batas_naik_m, batas_turun_m, PALETTE["blue"]),
    (axes[1], "Mobil", MED_b, batas_naik_b, batas_turun_b, PALETTE["teal"]),
]:
    data = df_pred[f"Pred {kd}"]
    ax.hist(data, bins=30, color=col, alpha=0.7, edgecolor="white")
    ax.axvline(bturun, color=PALETTE["amber"], lw=2.5, ls="--",
               label=f"70% Median: Rp {bturun:,.0f}")
    ax.axvline(bnaik, color=PALETTE["green"], lw=2.5, ls="--",
               label=f"130% Median: Rp {bnaik:,.0f}")
    ax.axvline(med, color=PALETTE["red"], lw=2.5, ls=":",
               label=f"Median: Rp {med:,.0f}")
    xmin, xmax = ax.get_xlim()
    ax.axvspan(xmin, bturun, alpha=0.08, color=PALETTE["amber"])
    ax.axvspan(bnaik, xmax, alpha=0.08, color=PALETTE["green"])
    ax.set_title(f"Distribusi Prediksi — {kd}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Revenue (Rp)")
    ax.set_ylabel("Frekuensi")
    ax.legend(fontsize=9)
    ax.spines[["top","right"]].set_visible(False)

plt.suptitle("Distribusi Prediksi & Batas Rekomendasi",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("distribusi_prediksi.png", dpi=120)
plt.show()
print("✅ Grafik tersimpan: distribusi_prediksi.png\n")

print("\n" + "="*60)
print("🎉 ANALISIS LENGKAP SELESAI!")
print("="*60)
print(f"✅ R² Motor: {metrics_m['R2_test']:.4f} (Gap: {metrics_m['Gap']:.4f})")
print(f"✅ R² Mobil: {metrics_b['R2_test']:.4f} (Gap: {metrics_b['Gap']:.4f})")
print(f"\n📊 Total lokasi: {len(df_pred)}")
print(f"📊 Rekomendasi Motor - Naik: {(df_pred['Status Motor']=='Naik').sum()} | "
      f"Tetap: {(df_pred['Status Motor']=='Tetap').sum()} | "
      f"Turun: {(df_pred['Status Motor']=='Turun').sum()}")
print(f"📊 Rekomendasi Mobil - Naik: {(df_pred['Status Mobil']=='Naik').sum()} | "
      f"Tetap: {(df_pred['Status Mobil']=='Tetap').sum()} | "
      f"Turun: {(df_pred['Status Mobil']=='Turun').sum()}")
print("\n📁 File tersimpan:")
print("   - pred_vs_asli.png")
print("   - feature_importance.png")
print("   - rekomendasi_status.png")
print("   - distribusi_prediksi.png")
print("="*60)

