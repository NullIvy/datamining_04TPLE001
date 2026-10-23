"""
Generator dataset DUMMY v2: Security Awareness Perusahaan di Indonesia.

Satu baris = satu perusahaan. Semua baris FIKTIF (nama perusahaan, angka, insiden).
Kategori & ambang nilai mengikuti regulasi/laporan resmi (lihat README.md), lalu
data sengaja "dikotori" (null, duplikat) agar bisa dipakai latihan eksplorasi data.

Jalankan:  python generate_dataset.py
Output  :  data/security_awareness_indonesia.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_UNIK = 5000          # perusahaan unik
N_DUPLIKAT = 120       # baris duplikat persis yang disisipkan
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- master data
SEKTOR = {
    # sektor: (bobot, rata2 kematangan, peluang skala besar, peluang sistem "Strategis")
    "Perbankan":               (0.10, 0.85, 0.55, 0.70),
    "Asuransi & Multifinance": (0.07, 0.40, 0.35, 0.30),
    "Fintech":                 (0.08, 0.45, 0.20, 0.35),
    "Telekomunikasi":          (0.05, 0.70, 0.55, 0.60),
    "Teknologi Informasi":     (0.10, 0.50, 0.25, 0.25),
    "Kesehatan":               (0.09, -0.20, 0.30, 0.30),
    "Pendidikan":              (0.09, -0.35, 0.20, 0.10),
    "Manufaktur":              (0.11, -0.10, 0.35, 0.15),
    "Ritel & E-commerce":      (0.10, 0.10, 0.30, 0.20),
    "Logistik & Transportasi": (0.07, -0.15, 0.30, 0.25),
    "Energi & Pertambangan":   (0.05, 0.30, 0.60, 0.65),
    "Media & Kreatif":         (0.04, -0.30, 0.15, 0.05),
    "Pemerintahan & BUMN Jasa": (0.05, 0.05, 0.50, 0.55),
}

# kota: (latitude, longitude, prefix kode pos)
KOTA = {
    "Jakarta Selatan": (-6.261, 106.810, "12"), "Jakarta Pusat": (-6.186, 106.834, "10"),
    "Jakarta Barat": (-6.168, 106.759, "11"), "Jakarta Utara": (-6.138, 106.864, "14"),
    "Jakarta Timur": (-6.225, 106.900, "13"), "Bandung": (-6.917, 107.619, "40"),
    "Bekasi": (-6.238, 106.976, "17"), "Bogor": (-6.595, 106.817, "16"), "Depok": (-6.402, 106.794, "16"),
    "Cirebon": (-6.732, 108.552, "45"), "Surabaya": (-7.257, 112.752, "60"), "Malang": (-7.978, 112.630, "65"),
    "Sidoarjo": (-7.447, 112.718, "61"), "Gresik": (-7.157, 112.656, "61"), "Tangerang": (-6.178, 106.630, "15"),
    "Tangerang Selatan": (-6.288, 106.718, "15"), "Serang": (-6.120, 106.150, "42"), "Cilegon": (-6.002, 106.011, "42"),
    "Semarang": (-6.966, 110.417, "50"), "Surakarta": (-7.575, 110.824, "57"), "Kudus": (-6.805, 110.841, "59"),
    "Medan": (3.595, 98.672, "20"), "Deli Serdang": (3.531, 98.698, "20"), "Yogyakarta": (-7.797, 110.370, "55"),
    "Sleman": (-7.716, 110.355, "55"), "Makassar": (-5.147, 119.432, "90"), "Gowa": (-5.208, 119.452, "92"),
    "Denpasar": (-8.670, 115.212, "80"), "Badung": (-8.582, 115.177, "80"), "Balikpapan": (-1.239, 116.853, "76"),
    "Samarinda": (-0.502, 117.154, "75"), "Pekanbaru": (0.507, 101.448, "28"), "Dumai": (1.665, 101.447, "28"),
    "Batam": (1.045, 104.031, "29"), "Tanjungpinang": (0.918, 104.446, "29"), "Palembang": (-2.976, 104.775, "30"),
    "Jayapura": (-2.533, 140.718, "99"), "Bandar Lampung": (-5.397, 105.266, "35"), "Padang": (-0.947, 100.417, "25"),
    "Pontianak": (-0.026, 109.342, "78"), "Manado": (1.474, 124.842, "95"), "Mataram": (-8.583, 116.116, "83"),
    "Banjarmasin": (-3.319, 114.590, "70"),
}
WILAYAH = {
    "DKI Jakarta": (0.27, ["Jakarta Selatan", "Jakarta Pusat", "Jakarta Barat", "Jakarta Utara", "Jakarta Timur"]),
    "Jawa Barat": (0.16, ["Bandung", "Bekasi", "Bogor", "Depok", "Cirebon"]),
    "Jawa Timur": (0.12, ["Surabaya", "Malang", "Sidoarjo", "Gresik"]),
    "Banten": (0.08, ["Tangerang", "Tangerang Selatan", "Serang", "Cilegon"]),
    "Jawa Tengah": (0.07, ["Semarang", "Surakarta", "Kudus"]),
    "Sumatera Utara": (0.05, ["Medan", "Deli Serdang"]),
    "DI Yogyakarta": (0.03, ["Yogyakarta", "Sleman"]),
    "Sulawesi Selatan": (0.03, ["Makassar", "Gowa"]),
    "Bali": (0.03, ["Denpasar", "Badung"]),
    "Kalimantan Timur": (0.025, ["Balikpapan", "Samarinda"]),
    "Riau": (0.02, ["Pekanbaru", "Dumai"]),
    "Kepulauan Riau": (0.02, ["Batam", "Tanjungpinang"]),
    "Sumatera Selatan": (0.02, ["Palembang"]),
    "Lampung": (0.015, ["Bandar Lampung"]),
    "Sumatera Barat": (0.015, ["Padang"]),
    "Kalimantan Barat": (0.01, ["Pontianak"]),
    "Kalimantan Selatan": (0.01, ["Banjarmasin"]),
    "Sulawesi Utara": (0.01, ["Manado"]),
    "Nusa Tenggara Barat": (0.01, ["Mataram"]),
    "Papua": (0.01, ["Jayapura"]),
}

KATA_A = ["Arta", "Bumi", "Cipta", "Dwi", "Eka", "Graha", "Inti", "Jaya", "Karya", "Lintas",
          "Mitra", "Nusa", "Prima", "Sinar", "Tirta", "Wahana", "Mega", "Surya", "Bina", "Kreasi",
          "Adhi", "Cakra", "Dharma", "Gita", "Harapan", "Indah", "Kencana", "Laksana", "Mulia", "Panca",
          "Rajawali", "Samudra", "Tunas", "Wira", "Yasa", "Bahtera"]
KATA_B = ["Digital", "Data", "Solusi", "Teknologi", "Logistik", "Medika", "Finansial", "Energi",
          "Niaga", "Industri", "Sarana", "Media", "Informatika", "Sentosa", "Edukasi", "Retail",
          "Cipta", "Daya", "Kapital", "Trans"]
KATA_C = ["Abadi", "Mandiri", "Sejahtera", "Utama", "Persada", "Nusantara", "Global", "Perkasa",
          "Gemilang", "Lestari", "Makmur", "Raya", "Sentosa", "Jaya", "Indonesia"]

# Ambang Indeks KAMI (Perban BSSN No. 8 & 9 Tahun 2021) per kategori sistem elektronik
AMBANG_KAMI = {  # (min "Pemenuhan Kerangka Kerja Dasar", min "Cukup Baik", min "Baik")
    "Strategis": (334, 536, 610),
    "Tinggi": (273, 456, 584),
    "Rendah": (175, 313, 536),
}

# Motif penyerang & vektor awal, bobot berbeda per sektor (data sintetis)
MOTIF = ["Finansial", "Pencurian Data/Spionase", "Hacktivism", "Insider/Balas Dendam", "Iseng/Uji Kemampuan"]
MOTIF_BOBOT = {
    "default": [0.50, 0.20, 0.08, 0.10, 0.12],
    "Perbankan": [0.70, 0.15, 0.03, 0.08, 0.04], "Fintech": [0.68, 0.17, 0.03, 0.07, 0.05],
    "Asuransi & Multifinance": [0.62, 0.20, 0.03, 0.10, 0.05],
    "Pemerintahan & BUMN Jasa": [0.20, 0.30, 0.30, 0.08, 0.12], "Pendidikan": [0.25, 0.20, 0.15, 0.10, 0.30],
    "Media & Kreatif": [0.25, 0.15, 0.30, 0.10, 0.20], "Kesehatan": [0.45, 0.35, 0.05, 0.10, 0.05],
    "Energi & Pertambangan": [0.30, 0.40, 0.18, 0.08, 0.04], "Telekomunikasi": [0.40, 0.35, 0.10, 0.10, 0.05],
}
VEKTOR = ["Email Phishing", "Kredensial Bocor", "Eksploitasi Kerentanan", "Pihak Ketiga/Vendor", "Perangkat USB/Fisik"]


def pilih(pilihan, bobot, n=None):
    bobot = np.asarray(bobot, dtype=float)
    return rng.choice(pilihan, size=n, p=bobot / bobot.sum())


def kategori_kami(skor, kategori_se):
    dasar, cukup, baik = AMBANG_KAMI[kategori_se]
    if skor >= baik:
        return "Baik"
    if skor >= cukup:
        return "Cukup Baik"
    if skor >= dasar:
        return "Pemenuhan Kerangka Kerja Dasar"
    return "Tidak Layak"


# ---------------------------------------------------------------- generate
nama_semua = [f"PT {a} {b} {c}" for a in KATA_A for b in KATA_B for c in KATA_C if b != c]
nama = rng.choice(nama_semua, size=N_UNIK, replace=False)

sektor_list = list(SEKTOR)
sektor = pilih(sektor_list, [SEKTOR[s][0] for s in sektor_list], N_UNIK)
prov_list = list(WILAYAH)
provinsi = pilih(prov_list, [WILAYAH[p][0] for p in prov_list], N_UNIK)

rows = []
for i in range(N_UNIK):
    s, p = sektor[i], provinsi[i]
    _, mat_sektor, p_besar, p_strategis = SEKTOR[s]
    kota = rng.choice(WILAYAH[p][1])
    lat0, lon0, pos = KOTA[kota]
    lat = round(lat0 + rng.normal(0, 0.03), 5)
    lon = round(lon0 + rng.normal(0, 0.03), 5)
    kode_pos = f"{pos}{rng.integers(100, 999):03d}"

    # Skala usaha (kriteria omzet tahunan mengacu PP 7/2021)
    p_kecil = max(0.10, 0.45 - p_besar)
    skala = pilih(["Kecil", "Menengah", "Besar"], [p_kecil, 1 - p_kecil - p_besar, p_besar])
    if skala == "Kecil":
        karyawan = int(rng.integers(5, 50))
        omzet = round(float(rng.uniform(2, 15)), 2)
        cabang = int(rng.poisson(0.2))
    elif skala == "Menengah":
        karyawan = int(rng.integers(50, 250))
        omzet = round(float(rng.uniform(15, 50)), 2)
        cabang = int(rng.poisson(1.5))
    else:
        karyawan = int(min(25000, max(250, rng.lognormal(6.6, 0.9))))
        omzet = round(float(min(80000, 50 + rng.lognormal(5.0, 1.4))), 2)
        cabang = int(rng.poisson(12))

    kepemilikan = pilih(["Swasta Nasional", "PMA", "BUMN/BUMD", "Terbuka (Tbk)"],
                        [0.58, 0.18, 0.10, 0.14] if skala == "Besar" else [0.80, 0.15, 0.03, 0.02])
    if s == "Pemerintahan & BUMN Jasa":
        kepemilikan = "BUMN/BUMD"
    tahun_min = 2010 if s == "Fintech" else 1970
    tahun_berdiri = int(rng.integers(tahun_min, 2024))

    kategori_se = pilih(["Strategis", "Tinggi", "Rendah"],
                        [p_strategis, 0.45, max(0.05, 0.55 - p_strategis)])

    # Variabel laten: kematangan keamanan (tidak ikut dikeluarkan)
    bonus_skala = {"Kecil": -0.6, "Menengah": 0.0, "Besar": 0.6}[skala]
    bonus_milik = 0.3 if kepemilikan in ("PMA", "Terbuka (Tbk)") else 0.0
    m = mat_sektor + bonus_skala + bonus_milik + rng.normal(0, 0.7)
    pm = 1 / (1 + np.exp(-m))  # 0..1

    # --- tata kelola
    tim = "Ya" if rng.random() < 0.15 + 0.8 * pm else "Tidak"
    staf = 0 if tim == "Tidak" else int(max(1, round(karyawan * rng.uniform(0.003, 0.02) + rng.integers(1, 4))))
    soc = pilih(["Internal", "Outsource (MSSP)", "Tidak Ada"],
                [pm ** 2 * (1.5 if skala == "Besar" else 0.4), 0.25 + 0.3 * pm, (1 - pm) * 1.3])
    iso = pilih(["Tersertifikasi", "Dalam Proses", "Belum"], [0.6 * pm ** 2, 0.25, (1 - pm) * 1.5])
    kebijakan = "Ya" if rng.random() < 0.25 + 0.7 * pm else "Tidak"
    pdp = pilih(["Patuh", "Sebagian", "Belum"], [pm * 0.9, 0.45, (1 - pm) * 0.8])
    dpo = "Ya" if rng.random() < 0.05 + 0.75 * pm * (1.0 if skala != "Kecil" else 0.4) else "Tidak"
    anggaran = round(float(np.clip(rng.normal(3 + 10 * pm, 2.0), 0.5, 25)), 1)
    asuransi = "Ya" if rng.random() < 0.02 + 0.35 * pm ** 2 * (1.5 if skala == "Besar" else 0.6) else "Tidak"
    bug_bounty = "Ya" if (s in ("Fintech", "Teknologi Informasi", "Perbankan", "Ritel & E-commerce", "Telekomunikasi")
                          and rng.random() < 0.6 * pm ** 2) else "Tidak"

    # --- lingkungan TI
    cloud = pilih(["On-Premise", "Hybrid", "Cloud-First"], [(1 - pm) + 0.2, 0.45, 0.15 + 0.3 * pm])
    remote = round(float(np.clip(rng.normal(30 if s in ("Teknologi Informasi", "Fintech", "Media & Kreatif") else 8, 15), 0, 90)), 1)
    if rng.random() < 0.30:
        remote = 0.0
    byod = round(float(np.clip(rng.normal(45 - 30 * pm, 15), 0, 100)), 1)
    vendor = int(rng.poisson({"Kecil": 1.5, "Menengah": 5, "Besar": 18}[skala]))
    admin = int(max(1, rng.poisson(max(1, karyawan * 0.01 * (1.4 - pm)))))
    patching = int(np.clip(rng.lognormal(np.log(10 + 80 * (1 - pm)), 0.5), 1, 365))
    backup = pilih(["Harian", "Mingguan", "Bulanan", "Tidak Ada"], [pm, 0.35, 0.25, (1 - pm) * 0.35])

    # --- kesadaran & perilaku
    pelatihan = int(np.clip(rng.poisson(0.5 + 6 * pm), 0, 12))
    if pelatihan == 0:
        metode, terlatih = "Tidak Ada", 0.0
    else:
        metode = pilih(["Tatap Muka", "E-Learning", "Hybrid", "Gamifikasi"], [0.28, 0.38, 0.26, 0.08])
        terlatih = round(float(np.clip(rng.normal(35 + 60 * pm, 12), 5, 100)), 1)
    simulasi = 0 if pelatihan == 0 else int(np.clip(rng.poisson(1 + 5 * pm), 0, 12))
    click_rate = round(float(np.clip(rng.normal(32 - 25 * pm - 0.08 * terlatih, 5), 1, 60)), 1)
    report_rate = round(float(np.clip(rng.normal(8 + 55 * pm, 8), 0, 95)), 1)
    laporan_phishing = int(rng.poisson(karyawan * report_rate / 100 * 0.3))
    mfa = pilih(["Penuh", "Sebagian", "Tidak"], [pm ** 1.5, 0.35, (1 - pm)])
    password = pilih(["Kuat", "Sedang", "Lemah"], [pm, 0.4, (1 - pm) * 0.8])
    pwd_bocor = round(float(np.clip(rng.normal(12 - 10 * pm, 4), 0, 40)), 1)

    skor_kami = int(np.clip(rng.normal(120 + 480 * pm, 55), 0, 645))
    kat_kami = kategori_kami(skor_kami, kategori_se)

    skor_awareness = round(float(np.clip(
        30 + 52 * pm + 0.15 * terlatih - 0.4 * click_rate - 0.15 * pwd_bocor + rng.normal(0, 7), 5, 99)), 1)
    tingkat = "Tinggi" if skor_awareness >= 80 else ("Sedang" if skor_awareness >= 60 else "Rendah")

    # --- insiden
    lam = (0.4 + {"Kecil": 0.3, "Menengah": 1.0, "Besar": 2.2}[skala]) * (1.6 - pm) * (1.3 - skor_awareness / 100)
    insiden = int(rng.poisson(max(0.05, lam * 1.5)))
    if insiden == 0:
        jenis = penyebab = motif = vektor = "Tidak Ada"
        deteksi, kerugian, downtime = np.nan, 0.0, 0.0
    else:
        jenis = pilih(["Phishing", "Malware", "Ransomware", "Kebocoran Data",
                       "Social Engineering", "DDoS", "Web Defacement"],
                      [0.30, 0.18, 0.12, 0.14, 0.12, 0.07, 0.07])
        # Proporsi faktor manusia ~60% & pihak ketiga ~30% (Verizon DBIR 2025), disederhanakan
        penyebab = pilih(["Kesalahan Manusia", "Kredensial Dicuri", "Kerentanan Sistem", "Pihak Ketiga"],
                         [0.42, 0.18, 0.22, 0.18])
        motif = pilih(MOTIF, MOTIF_BOBOT.get(s, MOTIF_BOBOT["default"]))
        if jenis == "Web Defacement":
            motif = pilih(MOTIF, [0.05, 0.05, 0.55, 0.05, 0.30])
        vektor = {"Kesalahan Manusia": pilih(VEKTOR, [0.70, 0.05, 0.05, 0.05, 0.15]),
                  "Kredensial Dicuri": pilih(VEKTOR, [0.35, 0.60, 0.00, 0.05, 0.00]),
                  "Kerentanan Sistem": pilih(VEKTOR, [0.00, 0.05, 0.90, 0.05, 0.00]),
                  "Pihak Ketiga": pilih(VEKTOR, [0.05, 0.10, 0.10, 0.75, 0.00])}[penyebab]
        deteksi = round(float(rng.lognormal(np.log(24 + 300 * (1 - pm)), 0.8)), 1)
        skala_rugi = {"Kecil": 15, "Menengah": 120, "Besar": 900}[skala]
        kerugian = round(float(skala_rugi * insiden * rng.lognormal(0, 0.9) * (1.5 - pm)), 1)
        downtime = round(float(np.clip(rng.exponential(2 + 30 * (1 - pm)) * (2.5 if jenis == "Ransomware" else 1), 0, 720)), 1)
        if jenis in ("Phishing", "Social Engineering") and rng.random() < 0.5:
            downtime = 0.0

    tgl = pd.Timestamp("2026-01-05") + pd.Timedelta(days=int(rng.integers(0, 268)))

    rows.append({
        # identitas & lokasi
        "id_perusahaan": f"SA-{i + 1:05d}",
        "nama_perusahaan": nama[i],
        "sektor": s,
        "status_kepemilikan": kepemilikan,
        "provinsi": p,
        "kota": kota,
        "kode_pos": kode_pos,
        "latitude": lat,
        "longitude": lon,
        # profil perusahaan
        "skala_usaha": skala,
        "jumlah_karyawan": karyawan,
        "omzet_tahunan_miliar_rp": omzet,
        "tahun_berdiri": tahun_berdiri,
        "jumlah_cabang": cabang,
        "kategori_sistem_elektronik": kategori_se,
        # tata kelola keamanan
        "memiliki_tim_keamanan": tim,
        "jumlah_staf_keamanan_it": staf,
        "memiliki_soc": soc,
        "status_iso27001": iso,
        "kebijakan_keamanan_tertulis": kebijakan,
        "kepatuhan_uu_pdp": pdp,
        "memiliki_dpo": dpo,
        "anggaran_keamanan_persen_it": anggaran,
        "asuransi_siber": asuransi,
        "program_bug_bounty": bug_bounty,
        # lingkungan TI
        "adopsi_cloud": cloud,
        "persen_kerja_remote": remote,
        "persen_perangkat_byod": byod,
        "jumlah_vendor_ti": vendor,
        "jumlah_akun_admin": admin,
        "rata2_hari_patching": patching,
        "frekuensi_backup": backup,
        # kesadaran & perilaku
        "frekuensi_pelatihan_per_tahun": pelatihan,
        "metode_pelatihan": metode,
        "persen_karyawan_terlatih": terlatih,
        "simulasi_phishing_per_tahun": simulasi,
        "phishing_click_rate_persen": click_rate,
        "phishing_report_rate_persen": report_rate,
        "jumlah_laporan_phishing_karyawan": laporan_phishing,
        "penerapan_mfa": mfa,
        "kebijakan_password": password,
        "persen_password_bocor": pwd_bocor,
        "skor_indeks_kami": skor_kami,
        "kategori_indeks_kami": kat_kami,
        # insiden
        "jumlah_insiden_12_bulan": insiden,
        "jenis_insiden_utama": jenis,
        "motif_serangan_utama": motif,
        "vektor_serangan_awal": vektor,
        "penyebab_utama_insiden": penyebab,
        "waktu_deteksi_jam": deteksi,
        "lama_downtime_jam": downtime,
        "estimasi_kerugian_juta_rp": kerugian,
        # label
        "skor_awareness": skor_awareness,
        "tingkat_awareness": tingkat,
        "tanggal_survei": tgl.strftime("%Y-%m-%d"),
    })

df = pd.DataFrame(rows)

# ---------------------------------------------------- kotor-kan data (sengaja)
# Missing value acak (MCAR) pada beberapa kolom, seolah responden tidak mengisi
NULL_RATE = {
    "kota": 0.02,
    "kode_pos": 0.03,
    "latitude": 0.03,
    "jumlah_karyawan": 0.03,
    "omzet_tahunan_miliar_rp": 0.05,
    "kepatuhan_uu_pdp": 0.04,
    "anggaran_keamanan_persen_it": 0.06,
    "persen_kerja_remote": 0.04,
    "persen_perangkat_byod": 0.05,
    "rata2_hari_patching": 0.07,
    "frekuensi_backup": 0.03,
    "metode_pelatihan": 0.03,
    "persen_karyawan_terlatih": 0.04,
    "phishing_click_rate_persen": 0.05,
    "penerapan_mfa": 0.02,
    "persen_password_bocor": 0.06,
    "estimasi_kerugian_juta_rp": 0.03,
}
for kolom, rate in NULL_RATE.items():
    idx = rng.choice(df.index, size=int(round(rate * len(df))), replace=False)
    df.loc[idx, kolom] = np.nan
# latitude & longitude hilang berpasangan (GPS tidak terekam)
df.loc[df["latitude"].isna(), "longitude"] = np.nan

# Baris duplikat persis (seolah data terinput dua kali)
dup = df.sample(n=N_DUPLIKAT, random_state=SEED)
df = pd.concat([df, dup], ignore_index=True)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

out = Path(__file__).parent / "data" / "security_awareness_indonesia.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f"Tersimpan: {out} ({len(df):,} baris x {df.shape[1]} kolom)")
