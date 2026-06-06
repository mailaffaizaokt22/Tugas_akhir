# ============================================================
# VISUALISASI LENGKAP (10+ GRAFIK)
# Tambahkan setelah Cross-Validation dan sebelum Pred vs Asli
# ============================================================

print("\n" + "="*60)
print("📊 VISUALISASI LENGKAP")
print("="*60 + "\n")

# ============================================================
# 1. HEATMAP KORELASI
# ============================================================
print("1/10 Membuat Heatmap Korelasi...")

hf = [c for c in avail if c not in ["Tarif Motor","Tarif Mobil"]]
df_h = df_processed[hf].copy()
df_h = df_h.loc[:, df_h.std() != 0]
df_hm = df_h.copy()
if "Target Motor" in df_processed.columns:
    df_hm["Pendapatan Motor (avg)"] = df_processed["Target Motor"]
if "Target Mobil" in df_processed.columns:
    df_hm["Pendapatan Mobil (avg)"] = df_processed["Target Mobil"]

corr = df_hm.corr().fillna(0)
nf = len(corr.columns)
cpx = max(0.7, min(1.1, 24/nf))
fa = max(6, min(10, int(180/nf)))

cmc = LinearSegmentedColormap.from_list("hm", [
    (0.13,0.47,0.71),(0.42,0.68,0.84),(1,1,0.88),
    (0.99,0.55,0.24),(0.89,0.10,0.11)], N=256)

fig, ax = plt.subplots(figsize=(nf*cpx+4, nf*cpx*0.85+3), facecolor="white", dpi=120)
ax.set_facecolor("white")
sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmc, center=0, vmin=-1, vmax=1,
            square=True, linewidths=0, ax=ax,
            annot_kws={"size": fa, "weight": "bold", "color": "black"},
            cbar_kws={"shrink": 0.5, "label": "Pearson r", "pad": 0.02})
for s in ax.spines.values(): s.set_visible(False)
ax.set_title("Heatmap Korelasi Fitur Parkir Banyumas",
             pad=14, fontsize=14, fontweight="bold")
ax.tick_params(axis="x", labelsize=fa, rotation=45, length=0)
ax.tick_params(axis="y", labelsize=fa, rotation=0, length=0)
plt.tight_layout(pad=1.5)
plt.savefig("heatmap_korelasi.png", bbox_inches="tight", dpi=130)
plt.show()
print("✅ heatmap_korelasi.png tersimpan\n")

# ============================================================
# 2. PAIRPLOT
# ============================================================
print("2/10 Membuat Pairplot...")

ppc = [c for c in ["Jumlah Motor Weekday","Jumlah Mobil Weekday",
                    "Jumlah Motor Weekend","Jumlah Mobil Weekend",
                    "Tarif Motor","Tarif Mobil"] if c in df_processed.columns]
g = sns.pairplot(df_processed[ppc], diag_kind="kde", corner=True,
                 plot_kws={"alpha": 0.45, "s": 18, "color": PALETTE["blue"]},
                 diag_kws={"fill": True, "color": PALETTE["blue"]})
g.figure.suptitle("Pairplot Variabel Utama Kendaraan & Tarif",
                   y=1.01, fontsize=13, fontweight="bold")
g.figure.savefig("pairplot.png", bbox_inches="tight", dpi=110)
plt.show()
print("✅ pairplot.png tersimpan\n")

# ============================================================
# 3. VISUALISASI REKAYASA FITUR
# ============================================================
print("3/10 Membuat Visualisasi Rekayasa Fitur...")

engineered_candidates = [
    "Rasio Motor/Mobil Weekday", "Rasio Motor/Mobil Weekend",
    "Total Kendaraan Weekday", "Total Kendaraan Weekend",
    "Delta Weekend-Weekday Motor", "Delta Weekend-Weekday Mobil",
    "Interaksi Tarif Motor x Volume", "Interaksi Tarif Mobil x Volume"
]
engineered_cols = [c for c in engineered_candidates if c in df_processed.columns]

if engineered_cols:
    ncols = 2
    nrows = int(np.ceil(len(engineered_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4*nrows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(engineered_cols):
        ax = axes[i]
        sns.boxplot(y=df_processed[col], ax=ax, color=PALETTE["indigo"], fliersize=2)
        ax.set_title(f"Distribusi Fitur Rekayasa: {col}", fontweight="bold")
        ax.set_ylabel(col)
        ax.spines[["top", "right"]].set_visible(False)

    for j in range(len(engineered_cols), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig("rekayasa_fitur_boxplot.png", dpi=120)
    plt.show()
    print("✅ rekayasa_fitur_boxplot.png tersimpan\n")
else:
    print("⚠️ Kolom rekayasa fitur tidak ditemukan, lewati visualisasi ini.\n")

# ============================================================
# 4. DISTRIBUSI DATA LATIH VS DATA UJI
# ============================================================
print("4/10 Membuat Visualisasi Data Latih vs Data Uji...")

# Gunakan target set test dari metrik, dan estimasi train dari prediksi full data
if "Target Motor" in df_processed.columns:
    train_est_motor = df_processed["Target Motor"]
    test_motor = pd.Series(metrics_m["true_test"]).reset_index(drop=True)

    plt.figure(figsize=(11, 5))
    sns.kdeplot(train_est_motor, fill=True, color=PALETTE["blue"], alpha=0.35, label="Train (aproksimasi)")
    sns.kdeplot(test_motor, fill=True, color=PALETTE["amber"], alpha=0.35, label="Test")
    plt.title("Distribusi Target Motor: Data Latih vs Data Uji", fontweight="bold")
    plt.xlabel("Pendapatan Motor (Rp)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig("train_test_motor.png", dpi=120)
    plt.show()
    print("✅ train_test_motor.png tersimpan")

if "Target Mobil" in df_processed.columns:
    train_est_mobil = df_processed["Target Mobil"]
    test_mobil = pd.Series(metrics_b["true_test"]).reset_index(drop=True)

    plt.figure(figsize=(11, 5))
    sns.kdeplot(train_est_mobil, fill=True, color=PALETTE["teal"], alpha=0.35, label="Train (aproksimasi)")
    sns.kdeplot(test_mobil, fill=True, color=PALETTE["red"], alpha=0.30, label="Test")
    plt.title("Distribusi Target Mobil: Data Latih vs Data Uji", fontweight="bold")
    plt.xlabel("Pendapatan Mobil (Rp)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig("train_test_mobil.png", dpi=120)
    plt.show()
    print("✅ train_test_mobil.png tersimpan\n")

# ============================================================
# 5. HISTOGRAM FITUR UTAMA (SATU-SATU)
# ============================================================
print("5/10 Membuat Histogram Fitur Utama satu per satu...")

fitur_utama = [
    "Jumlah Motor Weekday", "Jumlah Mobil Weekday",
    "Jumlah Motor Weekend", "Jumlah Mobil Weekend",
    "Tarif Motor", "Tarif Mobil"
]
fitur_utama = [c for c in fitur_utama if c in df_processed.columns]

for col in fitur_utama:
    plt.figure(figsize=(10, 5))
    sns.histplot(df_processed[col], bins=30, kde=True, color=PALETTE["blue"], edgecolor="white", alpha=0.75)
    plt.title(f"Histogram Distribusi {col}", fontweight="bold")
    plt.xlabel(col)
    plt.ylabel("Frekuensi")
    plt.tight_layout()
    safe_name = col.lower().replace(" ", "_").replace("/", "_")
    plt.savefig(f"hist_{safe_name}.png", dpi=120)
    plt.show()
    print(f"✅ hist_{safe_name}.png tersimpan")
print()

# ============================================================
# 6. DISTRIBUSI PENDAPATAN TAHUNAN PER JENIS KENDARAAN
# ============================================================
print("6/10 Membuat Distribusi Pendapatan Tahunan per Jenis Kendaraan...")

annual_df = pd.DataFrame()
if "Target Motor" in df_processed.columns:
    annual_df["Motor"] = df_processed["Target Motor"] * 365
if "Target Mobil" in df_processed.columns:
    annual_df["Mobil"] = df_processed["Target Mobil"] * 365

if not annual_df.empty:
    annual_long = annual_df.melt(var_name="Jenis Kendaraan", value_name="Pendapatan Tahunan")
    plt.figure(figsize=(11, 6))
    sns.violinplot(data=annual_long, x="Jenis Kendaraan", y="Pendapatan Tahunan",
                   palette=[PALETTE["blue"], PALETTE["teal"]], inner="quartile")
    plt.title("Distribusi Pendapatan Tahunan per Jenis Kendaraan", fontweight="bold")
    plt.xlabel("Jenis Kendaraan")
    plt.ylabel("Pendapatan Tahunan (Rp)")
    plt.tight_layout()
    plt.savefig("pendapatan_tahunan_per_jenis.png", dpi=120)
    plt.show()
    print("✅ pendapatan_tahunan_per_jenis.png tersimpan\n")
else:
    print("⚠️ Kolom target pendapatan tidak ditemukan, lewati visualisasi ini.\n")

# ============================================================
# 7 & 8. PREDIKSI VS ASLI (Motor & Mobil)
# ============================================================
print("7/10 Membuat Prediksi vs Asli — Motor...")
print("8/10 Membuat Prediksi vs Asli — Mobil...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, kd, pte, tte, col in [
    (axes[0], "Motor", metrics_m["pred_test"], metrics_m["true_test"], PALETTE["blue"]),
    (axes[1], "Mobil", metrics_b["pred_test"], metrics_b["true_test"], PALETTE["teal"]),
]:
    ax.scatter(tte, pte, alpha=0.6, color=col, edgecolors="white", s=60)
    lims = [min(tte.min(), pte.min()), max(tte.max(), pte.max())]
    ax.plot(lims, lims, "--", color=PALETTE["red"], lw=2, label="Perfect")
    r2 = r2_score(tte, pte)
    ax.set_title(f"Prediksi vs Asli — {kd} (R²={r2:.4f})", fontweight="bold")
    ax.set_xlabel("Actual (Rp)")
    ax.set_ylabel("Predicted (Rp)")
    ax.legend()
    ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("pred_vs_asli.png", dpi=120)
plt.show()
print("✅ pred_vs_asli.png tersimpan\n")

# ============================================================
# 9 & 10. RESIDUALS (Motor & Mobil)
# ============================================================
print("9/10 Membuat Residuals — Motor...")

res_m = metrics_m["true_test"] - metrics_m["pred_test"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(res_m, bins=25, color=PALETTE["indigo"], alpha=0.8, edgecolor="white")
axes[0].axvline(0, color=PALETTE["red"], lw=2.5, ls="--", label="Zero error")
axes[0].axvline(res_m.mean(), color=PALETTE["amber"], lw=2, ls=":",
                label=f"Mean: Rp {res_m.mean():,.0f}")
axes[0].set_title("Distribusi Residual — Motor", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Residual (Rp)")
axes[0].set_ylabel("Frekuensi")
axes[0].legend()

axes[1].scatter(metrics_m["pred_test"], res_m, alpha=0.55, color=PALETTE["indigo"],
                edgecolors="white", s=45)
axes[1].axhline(0, color=PALETTE["red"], lw=2, ls="--")
axes[1].set_title("Residuals vs Fitted — Motor", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Predicted (Rp)")
axes[1].set_ylabel("Residual (Rp)")
for ax in axes: ax.spines[["top","right"]].set_visible(False)

plt.suptitle("Analisis Residual Motor — Idealnya Menyebar Acak di Sekitar 0",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("residuals_motor.png", dpi=120)
plt.show()
print("✅ residuals_motor.png tersimpan\n")

print("10/10 Membuat Residuals — Mobil...")

res_b = metrics_b["true_test"] - metrics_b["pred_test"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(res_b, bins=25, color=PALETTE["teal"], alpha=0.8, edgecolor="white")
axes[0].axvline(0, color=PALETTE["red"], lw=2.5, ls="--", label="Zero error")
axes[0].axvline(res_b.mean(), color=PALETTE["amber"], lw=2, ls=":",
                label=f"Mean: Rp {res_b.mean():,.0f}")
axes[0].set_title("Distribusi Residual — Mobil", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Residual (Rp)")
axes[0].set_ylabel("Frekuensi")
axes[0].legend()

axes[1].scatter(metrics_b["pred_test"], res_b, alpha=0.55, color=PALETTE["teal"],
                edgecolors="white", s=45)
axes[1].axhline(0, color=PALETTE["red"], lw=2, ls="--")
axes[1].set_title("Residuals vs Fitted — Mobil", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Predicted (Rp)")
axes[1].set_ylabel("Residual (Rp)")
for ax in axes: ax.spines[["top","right"]].set_visible(False)

plt.suptitle("Analisis Residual Mobil — Idealnya Menyebar Acak di Sekitar 0",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("residuals_mobil.png", dpi=120)
plt.show()
print("✅ residuals_mobil.png tersimpan\n")

# ============================================================
# 11 & 12. FEATURE IMPORTANCE (Motor & Mobil)
# ============================================================
print("11/12 Membuat Feature Importance — Motor...")
print("12/12 Membuat Feature Importance — Mobil...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, kd, model, Xte, col in [
    (axes[0], "Motor", model_m, X_test_m, PALETTE["blue"]),
    (axes[1], "Mobil", model_b, X_test_b, PALETTE["teal"]),
]:
    fi = pd.Series(model.named_steps["xgb"].feature_importances_, index=Xte.columns)
    fi = fi.sort_values(ascending=True).tail(10)
    ax.barh(fi.index, fi.values, color=col, alpha=0.8, edgecolor="white")
    ax.set_title(f"Feature Importance — {kd} (Top 10)", fontweight="bold")
    ax.set_xlabel("Importance")
    ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=120)
plt.show()
print("✅ feature_importance.png tersimpan\n")

# ============================================================
# 13. DISTRIBUSI PREDIKSI
# ============================================================
print("13/13 Membuat Distribusi Prediksi...")

# Hitung median dan batas (akan digunakan untuk rekomendasi nanti)
df_pred_temp = df_processed.copy()
df_pred_temp["Pred Motor"] = np.maximum(np.expm1(model_m.predict(df_pred_temp[avail])), 0)
df_pred_temp["Pred Mobil"] = np.maximum(np.expm1(model_b.predict(df_pred_temp[avail])), 0)

MED_m_temp = df_pred_temp["Pred Motor"].median()
MED_b_temp = df_pred_temp["Pred Mobil"].median()
batas_naik_m_temp = MED_m_temp * PERSEN_NAIK
batas_turun_m_temp = MED_m_temp * PERSEN_TURUN
batas_naik_b_temp = MED_b_temp * PERSEN_NAIK
batas_turun_b_temp = MED_b_temp * PERSEN_TURUN

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, kd, med, bnaik, bturun, col in [
    (axes[0], "Motor", MED_m_temp, batas_naik_m_temp, batas_turun_m_temp, PALETTE["blue"]),
    (axes[1], "Mobil", MED_b_temp, batas_naik_b_temp, batas_turun_b_temp, PALETTE["teal"]),
]:
    data = df_pred_temp[f"Pred {kd}"]
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

plt.suptitle("Distribusi Prediksi & Batas Rekomendasi (70%/130% Median)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("distribusi_prediksi.png", dpi=120)
plt.show()
print("✅ distribusi_prediksi.png tersimpan\n")

# ============================================================
# 14. STATUS REKOMENDASI (akan dibuat setelah rekomendasi)
# ============================================================
# Grafik ini akan dibuat setelah bagian rekomendasi tarif

print("="*60)
print("✅ 13+ visualisasi selesai!")
print("   (Status Rekomendasi akan dibuat setelah perhitungan rekomendasi)")
print("="*60 + "\n")
