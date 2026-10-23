"""
Eksplorasi & klasifikasi dataset Iris dengan Decision Tree.

Alur: load data -> cek kualitas (null, duplikat) -> statistik per spesies ->
split train/test -> Decision Tree -> evaluasi -> simpan grafik ke outputs/.

Jalankan:  python iris_klasifikasi.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text, plot_tree

BASE = Path(__file__).parent
DATA = BASE / "data" / "Iris.csv"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

FITUR = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
TARGET = "Species"
SEED = 42
WARNA = {"Iris-setosa": "#eda100", "Iris-versicolor": "#2a78d6", "Iris-virginica": "#e34948"}

ABU = "#52514e"
plt.rcParams.update({
    "figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c9c8c3", "axes.labelcolor": ABU, "xtick.color": ABU, "ytick.color": ABU,
    "axes.titleweight": "bold", "font.size": 10,
})


def garis(judul):
    print(f"\n{'=' * 60}\n{judul}\n{'=' * 60}")


# ------------------------------------------------------------------ 1. load & kualitas
df = pd.read_csv(DATA)
garis("1. DATA")
print(f"Ukuran              : {df.shape[0]} baris x {df.shape[1]} kolom")
print(f"Kolom numerik       : {df.select_dtypes(include='number').columns.tolist()}")
print(f"Kolom kategorikal   : {df.select_dtypes(exclude='number').columns.tolist()}")
print(f"Null                : {int(df.isna().sum().sum())}")
print(f"Zero                : {int((df[FITUR] == 0).sum().sum())}")

# Kolom Id selalu unik, jadi duplikat dicek tanpa Id
dup = df.drop(columns="Id").duplicated()
print(f"Duplikat (tanpa Id) : {int(dup.sum())} -> Id {df.loc[dup, 'Id'].tolist()}")
df = df.loc[~dup].reset_index(drop=True)
print(f"Baris dipakai       : {len(df)}")

# ------------------------------------------------------------------ 2. statistik
garis("2. RATA-RATA PER SPESIES (cm)")
print(df.groupby(TARGET)[FITUR].mean().round(2).to_string())
garis("3. KORELASI ANTAR FITUR")
print(df[FITUR].corr().round(2).to_string())

# ------------------------------------------------------------------ 3. split & model
X, y = df[FITUR], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
garis("4. TRAIN / TEST")
print(f"Train : {len(X_train)} baris | Test : {len(X_test)} baris")

model = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(X_train, y_train)
pred = model.predict(X_test)

# ------------------------------------------------------------------ 4. evaluasi
garis("5. EVALUASI")
acc = accuracy_score(y_test, pred)
kelas = list(model.classes_)
print(f"Accuracy : {acc:.3f} ({(pred == y_test).sum()} dari {len(y_test)} benar)\n")
laporan = classification_report(y_test, pred, digits=3)
print(laporan)
cm = confusion_matrix(y_test, pred, labels=kelas)
print(pd.DataFrame(cm, index=[f"aktual {k[5:]}" for k in kelas], columns=[f"pred {k[5:]}" for k in kelas]).to_string())

garis("6. ATURAN POHON")
aturan = export_text(model, feature_names=FITUR)
print(aturan)
imp = pd.Series(model.feature_importances_, index=FITUR).sort_values(ascending=False)
print("Feature importance:")
print((imp * 100).round(1).astype(str).add(" %").to_string())

(OUT / "laporan_evaluasi.txt").write_text(
    f"Accuracy: {acc:.3f}\n\n{laporan}\nConfusion matrix ({kelas}):\n{cm}\n\nAturan pohon:\n{aturan}", encoding="utf-8")

garis("7. CONTOH PREDIKSI")
baru = pd.DataFrame([[5.0, 3.4, 1.5, 0.2], [6.0, 2.9, 4.5, 1.5], [6.9, 3.1, 5.8, 2.2]], columns=FITUR)
for (_, r), p in zip(baru.iterrows(), model.predict(baru)):
    print(f"  petal {r.PetalLengthCm} x {r.PetalWidthCm} cm  ->  {p}")

# ------------------------------------------------------------------ 5. grafik
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
ax = axes[0]
for sp, warna in WARNA.items():
    d = df[df[TARGET] == sp]
    ax.scatter(d.PetalLengthCm, d.PetalWidthCm, s=28, color=warna, alpha=0.8,
               edgecolor="white", linewidth=0.6, label=sp.replace("Iris-", ""))
ax.axvline(2.45, color="#8a8984", ls="--", lw=1)
ax.text(2.55, 2.45, "batas 2,45 cm", color=ABU, fontsize=8)
ax.set_xlabel("Petal length (cm)")
ax.set_ylabel("Petal width (cm)")
ax.set_title("Petal memisahkan spesies")
ax.legend(frameon=False)

ax = axes[1]
rata = df.groupby(TARGET)[FITUR].mean()
lebar = 0.26
x = np.arange(len(FITUR))
for i, (sp, warna) in enumerate(WARNA.items()):
    ax.bar(x + (i - 1) * lebar, rata.loc[sp], width=lebar - 0.03, color=warna, label=sp.replace("Iris-", ""))
ax.set_xticks(x, [f.replace("Cm", "") for f in FITUR])
ax.set_ylabel("Rata-rata (cm)")
ax.set_title("Rata-rata ukuran per spesies")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(OUT / "ringkasan_iris.png")
plt.close()

fig, ax = plt.subplots(figsize=(5.2, 4.4))
ax.imshow(cm, cmap="Blues")
for i in range(len(kelas)):
    for j in range(len(kelas)):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12,
                color="white" if cm[i, j] > cm.max() / 2 else "#1a1a19")
label = [k.replace("Iris-", "") for k in kelas]
ax.set_xticks(range(3), label)
ax.set_yticks(range(3), label)
ax.set_xlabel("Prediksi")
ax.set_ylabel("Aktual")
ax.set_title(f"Confusion matrix (accuracy {acc:.1%})")
ax.spines[:].set_visible(False)
plt.tight_layout()
plt.savefig(OUT / "confusion_matrix.png")
plt.close()

fig, ax = plt.subplots(figsize=(12, 6))
plot_tree(model, feature_names=FITUR, class_names=label, filled=True, rounded=True, fontsize=9, ax=ax)
ax.set_title("Decision Tree Iris (max_depth = 3)", fontsize=13)
plt.tight_layout()
plt.savefig(OUT / "decision_tree.png")
plt.close()

export_graphviz(model, out_file=str(OUT / "decision_tree.dot"), feature_names=FITUR,
                class_names=label, filled=True, rounded=True)

garis("SELESAI")
for f in sorted(OUT.iterdir()):
    print(f"  outputs/{f.name}")
