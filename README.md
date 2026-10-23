# Data Mining

Tugas mata kuliah Data Mining (semester 4). Repo ini berisi dua proyek klasifikasi dengan alur yang sama.
Kenali data dulu, rapikan, baru latih model.

| Proyek | Pertanyaan | Data | Model |
|---|---|---|---|
| [`iris/`](iris/) | Bisa nggak spesies bunga ditebak dari ukuran kelopaknya? | 150 bunga, 4 fitur | Decision Tree |
| [`security-awareness/`](security-awareness/) | Perusahaan mana yang security awareness-nya rendah, dan kenapa? | 5.120 perusahaan (sintetis), 55 kolom | Decision Tree |

Iris dipakai buat belajar alurnya di data kecil yang bersih. Security awareness dipakai buat praktik di data yang lebih mirip dunia nyata: banyak kolom, ada duplikat, ada sel kosong, ada nilai nol yang punya arti.

## Struktur

```
datamining_unpam/
├── iris/
│   ├── data/Iris.csv
│   ├── iris_klasifikasi.py
│   ├── outputs/
│   └── README.md
├── security-awareness/
│   ├── data/
│   │   ├── security_awareness_indonesia.csv
│   │   └── security_awareness_bersih.csv
│   ├── 01_eksplorasi_data.ipynb
│   ├── 02_pengumpulan_data.ipynb
│   ├── klasifikasi_awareness.py
│   ├── sa_cli.py
│   ├── generate_dataset.py
│   ├── outputs/
│   └── README.md
├── requirements.txt
└── README.md
```

## Mulai cepat

Butuh Python 3.11 atau lebih baru (pandas 3 tidak jalan di versi lama).

```bash
git clone https://github.com/NullIvy/datamining_unpam.git
cd datamining_unpam
pip install -r requirements.txt

python iris/iris_klasifikasi.py
python security-awareness/klasifikasi_awareness.py
```

Hasil grafik dan laporan evaluasi masuk ke folder `outputs/` di masing-masing proyek. Script bisa dijalankan dari folder mana pun karena path-nya relatif terhadap lokasi file.

Notebook di `security-awareness/` paling enak dibuka di VS Code (extension **Python** + **Jupyter**), lalu klik **Run All**.

## Hasil singkat

| | Iris | Security Awareness |
|---|---|---|
| Baris dipakai (setelah buang duplikat) | 147 | 5.000 |
| Train / test | 117 / 30 | 4.000 / 1.000 |
| Accuracy | 96,7% | 72,7% (baseline tebak mayoritas 41,4%) |
| Fitur paling berpengaruh | Panjang & lebar petal | Skor Indeks KAMI, phishing click rate |

Detail cara baca angka-angka ini ada di README masing-masing proyek.

## Catatan soal data

- **Iris** adalah dataset publik klasik (Fisher, 1936) dari UCI Machine Learning Repository via Kaggle.
- **Security awareness** adalah data **sintetis** buatan sendiri. Struktur dan ambang nilainya mengacu ke regulasi resmi (UU PDP, Indeks KAMI BSSN, PP 7/2021), tapi angkanya bukan data perusahaan nyata. Jangan dikutip sebagai statistik Indonesia.

## Kontributor

Muhammad Rizki Firmansyah Bahtiar | 241011450073 | Teknik Informatika Universitas Pamulang
