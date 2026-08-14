"""
========================================================================================
ETL Script: Konversi File *_updated.xlsx ke Format Parquet (Cross-Platform)
Membaca seluruh file Excel hasil integrasi PDRB dan Transportasi, kemudian
mengekstrak setiap sheet menjadi file .parquet terkompresi di folder data/.
========================================================================================
"""

import os
import sys
import json
import re
import shutil
import pandas as pd
import numpy as np
from pathlib import Path

# Set output encoding ke UTF-8
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Path direktori
TARGET_PROJECT_DIR = Path(r"C:\Users\Eric\Documents\Kuliah - kerja\KP\Project\KorelasiPDRB")
DATA_DIR = TARGET_PROJECT_DIR / "data"
EXCEL_SOURCE_DIR = Path(r"C:\Users\Eric\Documents\Kuliah - kerja\KP\Data")


def clean_label(text):
    """Membersihkan whitespace dan karakter newline dari label kolom/teks."""
    if text is None or pd.isna(text):
        return ""
    text = str(text).strip().replace('\n', ' ')
    return re.sub(r'\s+', ' ', text)


def normalize_province_key(name):
    """Menghasilkan identifier provinsi yang seragam dalam huruf kecil snake_case."""
    name_clean = name.lower().strip()
    if "selatan" in name_clean:
        return "sulawesi_selatan", "Sulawesi Selatan"
    elif "utara" in name_clean:
        return "sulawesi_utara", "Sulawesi Utara"
    elif "tengah" in name_clean:
        return "sulawesi_tengah", "Sulawesi Tengah"
    elif "tenggara" in name_clean:
        return "sulawesi_tenggara", "Sulawesi Tenggara"
    elif "barat" in name_clean:
        return "sulawesi_barat", "Sulawesi Barat"
    elif "gorontalo" in name_clean:
        return "gorontalo", "Gorontalo"
    elif "maluku utara" in name_clean:
        return "maluku_utara", "Maluku Utara"
    elif "maluku" in name_clean:
        return "maluku", "Maluku"
    return re.sub(r'[^a-z0-9]+', '_', name_clean).strip('_'), name


def sanitize_df_for_parquet(df):
    """Ensure all columns have clean, unambiguous dtypes for pyarrow."""
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            # Try to convert to numeric if mostly numeric
            num_series = pd.to_numeric(df_clean[col], errors='coerce')
            if num_series.notna().sum() > len(df_clean) * 0.7:
                df_clean[col] = num_series
            else:
                df_clean[col] = df_clean[col].astype(str).replace({'nan': '', 'None': '', '<NA>': '', 'NaN': ''})
    return df_clean


def parse_pdrb_sheet(raw_df):
    """
    Memecah sheet 'PDRB' menjadi 3 tabel terstruktur:
    1. Data Triwulanan (Q1-Q4)
    2. Matriks Korelasi Lapangan Usaha
    3. Matriks Korelasi Pengeluaran
    """
    # 1. Cari batas Seksi 1: Data Triwulanan
    triwulan_row_idx = None
    for idx, row in raw_df.iterrows():
        val0 = str(row.iloc[0]).strip().lower() if pd.notna(row.iloc[0]) else ""
        if val0 in ['triwulan', 'periode', 'quarter', 'waktu']:
            triwulan_row_idx = idx
            break
        if not triwulan_row_idx:
            for c_i in range(min(3, len(row))):
                cv = str(row.iloc[c_i]).strip().lower() if pd.notna(row.iloc[c_i]) else ""
                if cv == 'triwulan':
                    triwulan_row_idx = idx
                    break

    df_triwulan = None
    curr_idx = 0
    if triwulan_row_idx is not None:
        header_row = raw_df.iloc[triwulan_row_idx]
        data_rows = []
        curr_idx = triwulan_row_idx + 1
        while curr_idx < len(raw_df):
            row = raw_df.iloc[curr_idx]
            val0 = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
            if not val0 or 'matriks' in val0.lower() or 'korelasi' in val0.lower() or '2.' in val0:
                break
            if re.search(r'triwulan|tw|q\d|t\d', val0, re.IGNORECASE) or val0 in ['Triwulan I', 'Triwulan II', 'Triwulan III', 'Triwulan IV']:
                data_rows.append(row)
            curr_idx += 1

        if data_rows:
            df_triwulan = pd.DataFrame(data_rows)
            df_triwulan.columns = [clean_label(c) if pd.notna(c) else f"col_{i}" for i, c in enumerate(header_row)]
            first_col = df_triwulan.columns[0]
            df_triwulan.rename(columns={first_col: 'Triwulan'}, inplace=True)
            for c in df_triwulan.columns:
                if c != 'Triwulan':
                    df_triwulan[c] = pd.to_numeric(df_triwulan[c], errors='coerce')

    # 2. Cari Seksi 2: Matriks Korelasi Lapangan Usaha
    df_correl_lu = None
    lu_correl_start = None
    for idx in range(curr_idx, len(raw_df)):
        row_str = " ".join([str(v) for v in raw_df.iloc[idx] if pd.notna(v)])
        if 'lapangan usaha' in row_str.lower() and ('korelasi' in row_str.lower() or 'tipe pdrb' in row_str.lower()):
            lu_correl_start = idx
            break

    if lu_correl_start is not None:
        lu_header_idx = lu_correl_start
        if 'matriks' in str(raw_df.iloc[lu_correl_start, 0]).lower() or '2.' in str(raw_df.iloc[lu_correl_start, 0]):
            lu_header_idx = lu_correl_start + 1

        lu_headers = [clean_label(c) for c in raw_df.iloc[lu_header_idx] if pd.notna(c)]
        lu_rows = []
        c_idx = lu_header_idx + 1
        while c_idx < len(raw_df):
            row = raw_df.iloc[c_idx]
            val0 = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
            if not val0 or 'pengeluaran' in val0.lower() or '3.' in val0 or 'matriks korelasi pdrb menurut pengeluaran' in val0.lower():
                break
            non_nulls = [v for v in row if pd.notna(v)]
            if len(non_nulls) >= 3:
                lu_rows.append(non_nulls[:5])
            c_idx += 1

        if lu_rows:
            df_correl_lu = pd.DataFrame(lu_rows)
            expected_cols = ['Lapangan Usaha', 'Tipe PDRB', 'Korelasi dgn Penumpang', 'Korelasi dgn Bagasi', 'Korelasi dgn Barang']
            if len(df_correl_lu.columns) == len(expected_cols):
                df_correl_lu.columns = expected_cols
            else:
                df_correl_lu.columns = lu_headers[:len(df_correl_lu.columns)]
            for c in df_correl_lu.columns:
                if 'Korelasi' in c:
                    df_correl_lu[c] = pd.to_numeric(df_correl_lu[c], errors='coerce')

    # 3. Cari Seksi 3: Matriks Korelasi Pengeluaran
    df_correl_peng = None
    peng_correl_start = None
    for idx in range(curr_idx, len(raw_df)):
        row_str = " ".join([str(v) for v in raw_df.iloc[idx] if pd.notna(v)])
        if 'pengeluaran' in row_str.lower() and ('korelasi' in row_str.lower() or 'tipe pdrb' in row_str.lower()):
            peng_correl_start = idx
            break

    if peng_correl_start is not None:
        peng_header_idx = peng_correl_start
        if 'matriks' in str(raw_df.iloc[peng_correl_start, 0]).lower() or '3.' in str(raw_df.iloc[peng_correl_start, 0]):
            peng_header_idx = peng_correl_start + 1

        peng_headers = [clean_label(c) for c in raw_df.iloc[peng_header_idx] if pd.notna(c)]
        peng_rows = []
        c_idx = peng_header_idx + 1
        while c_idx < len(raw_df):
            row = raw_df.iloc[c_idx]
            non_nulls = [v for v in row if pd.notna(v)]
            if len(non_nulls) >= 3:
                peng_rows.append(non_nulls[:5])
            c_idx += 1

        if peng_rows:
            df_correl_peng = pd.DataFrame(peng_rows)
            expected_cols = ['Komponen Pengeluaran', 'Tipe PDRB', 'Korelasi dgn Penumpang', 'Korelasi dgn Bagasi', 'Korelasi dgn Barang']
            if len(df_correl_peng.columns) == len(expected_cols):
                df_correl_peng.columns = expected_cols
            else:
                df_correl_peng.columns = peng_headers[:len(df_correl_peng.columns)]
            for c in df_correl_peng.columns:
                if 'Korelasi' in c:
                    df_correl_peng[c] = pd.to_numeric(df_correl_peng[c], errors='coerce')

    return df_triwulan, df_correl_lu, df_correl_peng


def main():
    print("=" * 70)
    print("🚀 MEMULAI KONVERSI FILE EXCEL KE PARQUET (KORELASI PDRB)")
    print("=" * 70)

    TARGET_PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Copy convert_to_parquet.py to TARGET_PROJECT_DIR as well
    self_script = Path(__file__)
    shutil.copy(self_script, TARGET_PROJECT_DIR / "convert_to_parquet.py")

    # 1. Cari seluruh file *_updated.xlsx
    excel_files = []
    for root, _, files in os.walk(EXCEL_SOURCE_DIR):
        for f in files:
            if 'updated' in f.lower() and f.endswith('.xlsx') and not f.startswith('~$'):
                excel_files.append(Path(root) / f)

    print(f"📦 Ditemukan {len(excel_files)} file Excel updated untuk dikonversi.\n")

    manifest = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "provinces": {},
        "files_count": len(excel_files)
    }

    converted_count = 0

    for file_path in excel_files:
        filename = file_path.name
        rel_path = file_path.relative_to(EXCEL_SOURCE_DIR)

        prov_key, prov_display = normalize_province_key(str(rel_path))
        year_match = re.search(r'20\d{2}', filename)
        year_str = year_match.group(0) if year_match else "2024"

        print(f"📄 Memproses: [{prov_display}] {filename}")

        if prov_key not in manifest["provinces"]:
            manifest["provinces"][prov_key] = {
                "name": prov_display,
                "years": {}
            }

        manifest["provinces"][prov_key]["years"][year_str] = {
            "source_file": filename,
            "sheets": []
        }

        try:
            xls = pd.ExcelFile(file_path)
            for sheet_name in xls.sheet_names:
                sheet_lower = sheet_name.strip().lower()

                if sheet_lower == 'pdrb':
                    raw_pdrb = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
                    df_tri, df_lu, df_peng = parse_pdrb_sheet(raw_pdrb)

                    if df_tri is not None:
                        out_tri = DATA_DIR / f"{prov_key}_{year_str}_pdrb_triwulan.parquet"
                        df_tri_clean = sanitize_df_for_parquet(df_tri)
                        df_tri_clean.to_parquet(out_tri, index=False, engine='pyarrow')
                        manifest["provinces"][prov_key]["years"][year_str]["sheets"].append("pdrb_triwulan")
                        converted_count += 1

                    if df_lu is not None:
                        out_lu = DATA_DIR / f"{prov_key}_{year_str}_pdrb_correl_lu.parquet"
                        df_lu_clean = sanitize_df_for_parquet(df_lu)
                        df_lu_clean.to_parquet(out_lu, index=False, engine='pyarrow')
                        manifest["provinces"][prov_key]["years"][year_str]["sheets"].append("pdrb_correl_lu")
                        converted_count += 1

                    if df_peng is not None:
                        out_peng = DATA_DIR / f"{prov_key}_{year_str}_pdrb_correl_peng.parquet"
                        df_peng_clean = sanitize_df_for_parquet(df_peng)
                        df_peng_clean.to_parquet(out_peng, index=False, engine='pyarrow')
                        manifest["provinces"][prov_key]["years"][year_str]["sheets"].append("pdrb_correl_peng")
                        converted_count += 1

                else:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    df.columns = [clean_label(c) if pd.notna(c) else f"col_{i}" for i, c in enumerate(df.columns)]
                    df = df.loc[:, ~df.columns.str.contains('^col_|^unnamed', case=False)]

                    clean_sheet_name = re.sub(r'[^a-z0-9_]+', '', sheet_lower)
                    out_file = DATA_DIR / f"{prov_key}_{year_str}_{clean_sheet_name}.parquet"
                    df_clean = sanitize_df_for_parquet(df)
                    df_clean.to_parquet(out_file, index=False, engine='pyarrow')
                    manifest["provinces"][prov_key]["years"][year_str]["sheets"].append(clean_sheet_name)
                    converted_count += 1

            print(f"   ✅ Sukses dikonversi ke Parquet.")

        except Exception as e:
            print(f"   ❌ Gagal memproses {filename}: {str(e)}")

    # Simpan manifest.json
    manifest_path = DATA_DIR / "manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"🎉 KONVERSI SELESAI: {converted_count} file .parquet berhasil dibuat di {DATA_DIR}")
    print(f"📑 Manifest metadata tersimpan di: {manifest_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
