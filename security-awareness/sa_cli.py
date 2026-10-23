"""
sa_cli.py — CLI untuk memeriksa, merapikan, dan "memanggil" (summon) data
Security Awareness Perusahaan Indonesia.

Contoh pemakaian:
  python sa_cli.py profil
  python sa_cli.py rapikan
  python sa_cli.py ringkas --by sektor
  python sa_cli.py ringkas --by motif_serangan_utama --metrik estimasi_kerugian_juta_rp
  python sa_cli.py cari --where "sektor == 'Perbankan' and tingkat_awareness == 'Rendah'"
  python sa_cli.py cari --where "jumlah_insiden_12_bulan >= 5" --kolom nama_perusahaan,sektor,jumlah_insiden_12_bulan --urut jumlah_insiden_12_bulan

Semua perintah bisa diberi --input <file.csv>. Default:
  data mentah  : data/security_awareness_indonesia.csv
  data bersih  : data/security_awareness_bersih.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
RAW = BASE / "data" / "security_awareness_indonesia.csv"
CLEAN = BASE / "data" / "security_awareness_bersih.csv"

# kode_pos berisi angka tapi sebenarnya label -> wajib dibaca sebagai teks
DTYPE_BACA = {"kode_pos": "string"}

# Kategori yang punya urutan (ordinal) — urut dari rendah ke tinggi
ORDINAL = {
    "skala_usaha": ["Kecil", "Menengah", "Besar"],
    "kategori_sistem_elektronik": ["Rendah", "Tinggi", "Strategis"],
    "status_iso27001": ["Belum", "Dalam Proses", "Tersertifikasi"],
    "kepatuhan_uu_pdp": ["Belum", "Sebagian", "Patuh"],
    "frekuensi_backup": ["Tidak Ada", "Bulanan", "Mingguan", "Harian"],
    "penerapan_mfa": ["Tidak", "Sebagian", "Penuh"],
    "kebijakan_password": ["Lemah", "Sedang", "Kuat"],
    "kategori_indeks_kami": ["Tidak Layak", "Pemenuhan Kerangka Kerja Dasar", "Cukup Baik", "Baik"],
    "tingkat_awareness": ["Rendah", "Sedang", "Tinggi"],
}
NOMINAL = ["sektor", "status_kepemilikan", "provinsi", "kota", "memiliki_soc", "adopsi_cloud",
           "metode_pelatihan", "jenis_insiden_utama", "motif_serangan_utama", "vektor_serangan_awal",
           "penyebab_utama_insiden"]
BINER = ["memiliki_tim_keamanan", "kebijakan_keamanan_tertulis", "memiliki_dpo", "asuransi_siber",
         "program_bug_bounty"]
ID_TEKS = ["id_perusahaan", "nama_perusahaan", "kode_pos"]


# ------------------------------------------------------------------ helpers
def baca(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"File tidak ditemukan: {path}. Jalankan dulu 'python sa_cli.py rapikan' atau cek --input.")
    df = pd.read_csv(path, dtype=DTYPE_BACA)
    if "tanggal_survei" in df:
        df["tanggal_survei"] = pd.to_datetime(df["tanggal_survei"])
    return df


def tampil(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown()
    except ImportError:  # tabulate belum terpasang
        return df.to_string()


def jenis_kolom(df: pd.DataFrame):
    numerik = df.select_dtypes(include="number").columns.tolist()
    kategorikal = [c for c in df.columns if c not in numerik and c != "tanggal_survei"]
    return numerik, kategorikal


def tabel_kualitas(df: pd.DataFrame) -> pd.DataFrame:
    """Jumlah & persen null dan zero per kolom (hanya kolom yang punya salah satunya)."""
    n = len(df)
    numerik = df.select_dtypes(include="number").columns
    t = pd.DataFrame({
        "tipe": df.dtypes.astype(str),
        "jumlah_null": df.isna().sum(),
        "jumlah_zero": pd.Series({c: int((df[c] == 0).sum()) if c in numerik else 0 for c in df.columns}),
    })
    t["persen_null"] = (t["jumlah_null"] / n * 100).round(2)
    t["persen_zero"] = (t["jumlah_zero"] / n * 100).round(2)
    t = t[(t["jumlah_null"] > 0) | (t["jumlah_zero"] > 0)]
    return t[["tipe", "jumlah_null", "persen_null", "jumlah_zero", "persen_zero"]].sort_values(
        ["jumlah_null", "jumlah_zero"], ascending=False)


# ------------------------------------------------------------------ rapikan
def rapikan_df(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    log = []
    df = df.copy()
    awal = len(df)

    # 1) Duplikat persis
    df = df.drop_duplicates().reset_index(drop=True)
    log.append(f"Hapus duplikat: {awal:,} -> {len(df):,} baris ({awal - len(df)} dibuang)")

    # 2) Isi null berbasis aturan logis (bukan tebakan)
    tanpa_latih = df["frekuensi_pelatihan_per_tahun"].eq(0)
    tanpa_insiden = df["jumlah_insiden_12_bulan"].eq(0)
    aturan = [
        ("metode_pelatihan", tanpa_latih, "Tidak Ada"),
        ("persen_karyawan_terlatih", tanpa_latih, 0.0),
        ("estimasi_kerugian_juta_rp", tanpa_insiden, 0.0),
    ]
    n_aturan = 0
    for kolom, kondisi, nilai in aturan:
        m = df[kolom].isna() & kondisi
        df.loc[m, kolom] = nilai
        n_aturan += int(m.sum())
    log.append(f"Isi berbasis aturan: {n_aturan} sel (tidak ada pelatihan/insiden -> 'Tidak Ada'/0)")

    # 3) Isi null numerik dengan median kelompoknya.
    #    Median dihitung hanya dari baris yang "sebanding" (mis. kerugian hanya dari
    #    perusahaan yang memang kena insiden), supaya angka 0 tidak menarik median turun.
    median_grup = {
        # kolom: (kolom kelompok, baris acuan median)
        "jumlah_karyawan": ("skala_usaha", None),
        "omzet_tahunan_miliar_rp": ("skala_usaha", None),
        "anggaran_keamanan_persen_it": ("sektor", None),
        "persen_kerja_remote": ("sektor", None),
        "persen_perangkat_byod": ("sektor", None),
        "rata2_hari_patching": ("skala_usaha", None),
        "persen_karyawan_terlatih": ("sektor", ~tanpa_latih),
        "phishing_click_rate_persen": ("sektor", None),
        "persen_password_bocor": ("sektor", None),
        "estimasi_kerugian_juta_rp": ("skala_usaha", ~tanpa_insiden),
    }
    for kolom, (grup, acuan) in median_grup.items():
        k = int(df[kolom].isna().sum())
        sumber = df if acuan is None else df[acuan]
        median = sumber.groupby(grup)[kolom].median()
        df[kolom] = df[kolom].fillna(df[grup].map(median))
        if k:
            log.append(f"Median per {grup}: {kolom} ({k} sel)")
    for kolom in ["jumlah_karyawan", "rata2_hari_patching"]:
        df[kolom] = df[kolom].round().astype("int64")

    # 4) Koordinat GPS kosong -> titik tengah kota (median koordinat perusahaan lain di kota itu)
    k = int(df["latitude"].isna().sum())
    for kolom in ["latitude", "longitude"]:
        df[kolom] = df[kolom].fillna(df.groupby("kota")[kolom].transform("median"))
        df[kolom] = df[kolom].fillna(df.groupby("provinsi")[kolom].transform("median"))
    if k:
        log.append(f"Median koordinat per kota/provinsi: latitude & longitude ({k} baris)")

    # 5) Isi null kategorikal
    for kolom in ["kepatuhan_uu_pdp", "penerapan_mfa", "metode_pelatihan", "frekuensi_backup"]:
        k = int(df[kolom].isna().sum())
        modus = df.groupby("sektor")[kolom].transform(lambda s: s.mode().iat[0])
        df[kolom] = df[kolom].fillna(modus)
        if k:
            log.append(f"Modus per sektor: {kolom} ({k} sel)")
    for kolom in ["kota", "kode_pos"]:
        k = int(df[kolom].isna().sum())
        df[kolom] = df[kolom].fillna("Tidak Diketahui")
        if k:
            log.append(f"Label 'Tidak Diketahui': {kolom} ({k} sel)")
    log.append("Dibiarkan kosong (struktural): waktu_deteksi_jam = tidak ada insiden")

    # 6) Tipe data yang tepat
    df["tanggal_survei"] = pd.to_datetime(df["tanggal_survei"])
    for kolom in BINER:
        df[kolom] = df[kolom].map({"Ya": True, "Tidak": False})
    for kolom, urutan in ORDINAL.items():
        df[kolom] = pd.Categorical(df[kolom], categories=urutan, ordered=True)
    for kolom in NOMINAL:
        df[kolom] = df[kolom].astype("category")
    log.append("Tipe data: ordinal -> category(ordered), nominal -> category, Ya/Tidak -> bool, tanggal -> datetime")

    if verbose:
        print("\n".join(f"  • {x}" for x in log))
    return df


# ------------------------------------------------------------------ commands
def cmd_profil(a):
    path = Path(a.input) if a.input else RAW
    df = baca(path)
    numerik, kategorikal = jenis_kolom(df)
    print(f"\nFile        : {path}")
    print(f"Ukuran      : {df.shape[0]:,} baris x {df.shape[1]} kolom")
    print(f"Duplikat    : {df.duplicated().sum():,} baris")
    print(f"Sel null    : {int(df.isna().sum().sum()):,} dari {df.size:,} ({df.isna().sum().sum() / df.size:.2%})")
    print(f"Sel zero    : {int((df[numerik] == 0).sum().sum()):,} (kolom numerik bernilai 0)")
    print(f"Numerik     : {len(numerik)} kolom  -> {', '.join(numerik)}")
    print(f"Kategorikal : {len(kategorikal)} kolom -> {', '.join(kategorikal)}\n")
    print(tampil(tabel_kualitas(df)))


def cmd_rapikan(a):
    src = Path(a.input) if a.input else RAW
    dst = Path(a.output) if a.output else CLEAN
    print(f"\nMerapikan {src} ...")
    df = rapikan_df(baca(src))
    df.to_csv(dst, index=False)
    print(f"\nTersimpan: {dst}  ({df.shape[0]:,} baris x {df.shape[1]} kolom, "
          f"sisa null = {int(df.isna().sum().sum())} sel struktural)")


def cmd_ringkas(a):
    df = baca(Path(a.input) if a.input else CLEAN)
    if a.by not in df:
        sys.exit(f"Kolom '{a.by}' tidak ada. Pilihan: {', '.join(df.columns)}")
    hasil = df.groupby(a.by, observed=True).agg(
        jumlah_perusahaan=("id_perusahaan", "count"),
        rata2_metrik=(a.metrik, "mean"),
        median_metrik=(a.metrik, "median"),
        rata2_insiden=("jumlah_insiden_12_bulan", "mean"),
        persen_awareness_rendah=("tingkat_awareness", lambda s: (s == "Rendah").mean() * 100),
    ).round(1).sort_values("rata2_metrik", ascending=a.naik)
    hasil = hasil.rename(columns={"rata2_metrik": f"rata2_{a.metrik}", "median_metrik": f"median_{a.metrik}"})
    print(f"\nRingkasan per '{a.by}' (metrik: {a.metrik})\n")
    print(tampil(hasil))


def cmd_cari(a):
    df = baca(Path(a.input) if a.input else CLEAN)
    hasil = df.query(a.where) if a.where else df
    if a.urut:
        hasil = hasil.sort_values(a.urut, ascending=False)
    if a.kolom:
        hasil = hasil[[c.strip() for c in a.kolom.split(",")]]
    print(f"\n{len(hasil):,} baris cocok\n")
    print(tampil(hasil.head(a.n)))
    if a.simpan:
        hasil.to_csv(a.simpan, index=False)
        print(f"\nTersimpan: {a.simpan}")


def main():
    p = argparse.ArgumentParser(description="CLI data Security Awareness Perusahaan Indonesia")
    sub = p.add_subparsers(dest="perintah", required=True)

    s = sub.add_parser("profil", help="Lihat ukuran, jenis kolom, null, zero, dan duplikat")
    s.add_argument("--input")
    s.set_defaults(func=cmd_profil)

    s = sub.add_parser("rapikan", help="Bersihkan data mentah lalu simpan versi rapi")
    s.add_argument("--input")
    s.add_argument("--output")
    s.set_defaults(func=cmd_rapikan)

    s = sub.add_parser("ringkas", help="Tabel ringkasan per kelompok")
    s.add_argument("--by", default="sektor", help="Kolom pengelompokan (default: sektor)")
    s.add_argument("--metrik", default="skor_awareness", help="Kolom angka yang diringkas")
    s.add_argument("--naik", action="store_true", help="Urutkan dari kecil ke besar")
    s.add_argument("--input")
    s.set_defaults(func=cmd_ringkas)

    s = sub.add_parser("cari", help="Filter data dengan ekspresi pandas query")
    s.add_argument("--where", help="Contoh: \"sektor == 'Fintech' and skor_awareness < 50\"")
    s.add_argument("--kolom", help="Kolom yang ditampilkan, pisahkan koma")
    s.add_argument("--urut", help="Urutkan menurun berdasarkan kolom ini")
    s.add_argument("--n", type=int, default=15, help="Jumlah baris ditampilkan (default 15)")
    s.add_argument("--simpan", help="Simpan hasil filter ke CSV")
    s.add_argument("--input")
    s.set_defaults(func=cmd_cari)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
