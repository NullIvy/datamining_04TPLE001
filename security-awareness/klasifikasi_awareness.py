"""
Klasifikasi tingkat security awareness perusahaan dengan Decision Tree.

Alur: load data mentah -> buang duplikat -> pisahkan fitur (X) & target (y) ->
split train/test -> pipeline (imputasi + one-hot encoding + Decision Tree) ->
evaluasi -> simpan grafik & pohon ke folder outputs/.

Jalankan:  python klasifikasi_awareness.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree

BASE = Path(__file__).parent
DATA = BASE / "data" / "security_awareness_indonesia.csv"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

TARGET = "tingkat_awareness"
KELAS = ["Rendah", "Sedang", "Tinggi"]
# Kolom yang TIDAK boleh jadi fitur:
# - identitas/label teks unik  -> model cuma menghafal
# - skor_awareness             -> label versi angka (target leakage)
DIBUANG = ["id_perusahaan", "nama_perusahaan", "kode_pos", "tanggal_survei", "skor_awareness", TARGET]
SEED = 42

BIRU, ABU = "#2a78d6", "#52514e"
plt.rcParams.update({
    "figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c9c8c3", "axes.labelcolor": ABU, "xtick.color": ABU, "ytick.color": ABU,
    "axes.titleweight": "bold", "font.size": 10,
})


def garis(judul):
    print(f"\n{'=' * 70}\n{judul}\n{'=' * 70}")


# ------------------------------------------------------------------ 1. load
df = pd.read_csv(DATA, dtype={"kode_pos": "string"})
garis("1. DATA")
print(f"Baris mentah        : {len(df):,}")
print(f"Duplikat dibuang    : {df.duplicated().sum():,}")
df = df.drop_duplicates().reset_index(drop=True)
print(f"Baris dipakai       : {len(df):,}")
print(f"Sel null            : {int(df.isna().sum().sum()):,} (ditangani di pipeline)")
print("\nDistribusi target:")
dist = df[TARGET].value_counts().reindex(KELAS)
print(pd.DataFrame({"jumlah": dist, "persen": (dist / len(df) * 100).round(1)}).to_string())

# ------------------------------------------------------------------ 2. X & y
X = df.drop(columns=DIBUANG)
y = df[TARGET]
fitur_num = X.select_dtypes(include="number").columns.tolist()
fitur_kat = X.select_dtypes(exclude="number").columns.tolist()
garis("2. FITUR (X) & TARGET (y)")
print(f"Fitur numerik       : {len(fitur_num)}")
print(f"Fitur kategorikal   : {len(fitur_kat)}")
print(f"Target              : {TARGET} -> {KELAS}")

# ------------------------------------------------------------------ 3. split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
garis("3. TRAIN / TEST")
print(f"Train : {len(X_train):,} baris (80%)")
print(f"Test  : {len(X_test):,} baris (20%)")

# ------------------------------------------------------------------ 4. pipeline
preprocess = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), fitur_num),
    ("kat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), fitur_kat),
])
model = Pipeline([
    ("preprocess", preprocess),
    ("tree", DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=SEED)),
])
model.fit(X_train, y_train)
pred = model.predict(X_test)

# ------------------------------------------------------------------ 5. evaluasi
garis("4. EVALUASI")
acc = accuracy_score(y_test, pred)
baseline = (y_test == y_train.mode().iat[0]).mean()
print(f"Accuracy model           : {acc:.3f} ({(pred == y_test).sum()} dari {len(y_test)} benar)")
print(f"Baseline (tebak mayoritas): {baseline:.3f}")
laporan = classification_report(y_test, pred, labels=KELAS, digits=3, zero_division=0)
print("\n" + laporan)

cm = confusion_matrix(y_test, pred, labels=KELAS)
print("Confusion matrix (baris = aktual, kolom = prediksi):")
print(pd.DataFrame(cm, index=[f"aktual {k}" for k in KELAS], columns=[f"prediksi {k}" for k in KELAS]).to_string())

(OUT / "laporan_evaluasi.txt").write_text(
    f"Accuracy: {acc:.3f}\nBaseline: {baseline:.3f}\n\n{laporan}\n"
    f"Confusion matrix (baris = aktual, kolom = prediksi, urutan {KELAS}):\n{cm}\n", encoding="utf-8")

# ------------------------------------------------------------------ 6. feature importance
nama_ohe = model.named_steps["preprocess"].get_feature_names_out()
imp = pd.Series(model.named_steps["tree"].feature_importances_, index=nama_ohe)


def kolom_asal(nama):
    jenis, sisa = nama.split("__", 1)
    if jenis == "num":
        return sisa
    return next(k for k in sorted(fitur_kat, key=len, reverse=True) if sisa.startswith(k + "_"))


imp_kolom = imp.groupby(imp.index.map(kolom_asal)).sum().sort_values(ascending=False)
garis("5. FITUR PALING BERPENGARUH (top 10)")
print((imp_kolom.head(10) * 100).round(1).astype(str).add(" %").to_string())

# ------------------------------------------------------------------ 7. contoh prediksi
garis("6. CONTOH PREDIKSI (1 perusahaan dari data test)")
contoh = X_test.iloc[[0]]
proba = model.predict_proba(contoh)[0]
tampil = ["sektor", "skala_usaha", "phishing_click_rate_persen", "skor_indeks_kami",
          "persen_karyawan_terlatih", "rata2_hari_patching", "penerapan_mfa"]
print(contoh[tampil].T.to_string(header=False))
print(f"\nPrediksi : {model.predict(contoh)[0]}   |   Aktual : {y_test.iloc[0]}")
print("Peluang  : " + ", ".join(f"{k} {p:.0%}" for k, p in zip(model.classes_, proba)))

# ------------------------------------------------------------------ 8. grafik
# confusion matrix
fig, ax = plt.subplots(figsize=(5.2, 4.4))
ax.imshow(cm, cmap="Blues")
for i in range(len(KELAS)):
    for j in range(len(KELAS)):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12,
                color="white" if cm[i, j] > cm.max() / 2 else "#1a1a19")
ax.set_xticks(range(len(KELAS)), KELAS)
ax.set_yticks(range(len(KELAS)), KELAS)
ax.set_xlabel("Prediksi")
ax.set_ylabel("Aktual")
ax.set_title(f"Confusion matrix (accuracy {acc:.1%})")
ax.spines[:].set_visible(False)
plt.tight_layout()
plt.savefig(OUT / "confusion_matrix.png")
plt.close()

# feature importance
top = imp_kolom.head(10).sort_values()
fig, ax = plt.subplots(figsize=(7.5, 4.4))
ax.barh(top.index, top.values * 100, color=BIRU, height=0.6)
for yy, v in enumerate(top.values * 100):
    ax.text(v + 0.5, yy, f"{v:.1f}%", va="center", color=ABU, fontsize=9)
ax.set_xlabel("Kontribusi ke keputusan pohon (%)")
ax.set_title("10 fitur paling berpengaruh")
ax.grid(axis="x", color="#ecebe7")
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(OUT / "feature_importance.png")
plt.close()

# pohon (3 level teratas supaya terbaca)
fig, ax = plt.subplots(figsize=(22, 9))
plot_tree(model.named_steps["tree"], feature_names=nama_ohe, class_names=list(model.classes_),
          max_depth=3, filled=True, rounded=True, impurity=False, fontsize=8, ax=ax)
ax.set_title("Decision Tree (3 level teratas)", fontsize=14)
plt.tight_layout()
plt.savefig(OUT / "decision_tree.png")
plt.close()

export_graphviz(model.named_steps["tree"], out_file=str(OUT / "decision_tree.dot"),
                feature_names=nama_ohe, class_names=list(model.classes_), filled=True, rounded=True)

garis("SELESAI")
for f in sorted(OUT.iterdir()):
    print(f"  outputs/{f.name}")
