"""
========================================================================================
FastAPI Serverless Application - Korelasi PDRB & Transportasi (Vercel Serverless Ready)
Dynamic Path Resolution, In-Memory Parquet Cache, dan Full Diagnostic Logging.
v4.0 — Growth Rate Analysis (YoY/QoQ) + Pooled Multi-Year Correlation
========================================================================================
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
from scipy.stats import pearsonr


# ======================================================================================
# DYNAMIC PATH RESOLUTION FOR VERCEL LINUX SERVERLESS & LOCAL ENVIRONMENT
# ======================================================================================

def resolve_data_dir() -> Path:
    """Mencari lokasi folder data/ secara dinamis di berbagai lingkungan Vercel & Lokal."""
    candidates = [
        Path(__file__).resolve().parent / "data",
        Path(__file__).resolve().parent.parent / "data",
        Path(os.getcwd()) / "data",
        Path(os.getcwd()) / "api" / "data",
        Path("/var/task/data"),
        Path("/var/task/api/data"),
    ]
    for p in candidates:
        if p.exists() and (p / "manifest.json").exists():
            return p
    for p in candidates:
        if p.exists() and len(list(p.glob("*.parquet"))) > 0:
            return p
    return Path(__file__).resolve().parent / "data"


def resolve_templates_dir() -> Path:
    """Mencari lokasi folder templates/ secara dinamis."""
    candidates = [
        Path(__file__).resolve().parent.parent / "templates",
        Path(__file__).resolve().parent / "templates",
        Path(os.getcwd()) / "templates",
        Path(os.getcwd()) / "api" / "templates",
        Path("/var/task/templates"),
        Path("/var/task/api/templates"),
    ]
    for p in candidates:
        if p.exists() and (p / "index.html").exists():
            return p
    return Path(__file__).resolve().parent.parent / "templates"


def resolve_public_dir() -> Path:
    """Mencari lokasi folder public/ secara dinamis."""
    candidates = [
        Path(__file__).resolve().parent.parent / "public",
        Path(__file__).resolve().parent / "public",
        Path(os.getcwd()) / "public",
        Path("/var/task/public"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return Path(__file__).resolve().parent.parent / "public"


DATA_DIR = resolve_data_dir()
TEMPLATES_DIR = resolve_templates_dir()
PUBLIC_DIR = resolve_public_dir()

# Inisialisasi FastAPI App
app = FastAPI(
    title="Korelasi PDRB & Transportasi API",
    description="Serverless API & Dashboard Analisis PDRB dan Transportasi BPS",
    version="4.0.0"
)

# Mount static files jika direktori ada
if PUBLIC_DIR.exists():
    try:
        app.mount("/public", StaticFiles(directory=str(PUBLIC_DIR)), name="public")
    except Exception as e:
        print(f"⚠️ Warning mounting /public: {e}")

# Setup Jinja2 Templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Cache data parquet di memory
_parquet_cache: Dict[str, pd.DataFrame] = {}

# Cache multi-year aggregated DataFrames
_multi_year_cache: Dict[str, pd.DataFrame] = {}

# Mapping triwulan string → integer yang aman (case-insensitive, toleran terhadap variasi)
_TRIWULAN_MAP: Dict[str, int] = {
    "triwulan i": 1, "triwulan 1": 1, "q1": 1, "tw i": 1, "tw 1": 1, "i": 1,
    "triwulan ii": 2, "triwulan 2": 2, "q2": 2, "tw ii": 2, "tw 2": 2, "ii": 2,
    "triwulan iii": 3, "triwulan 3": 3, "q3": 3, "tw iii": 3, "tw 3": 3, "iii": 3,
    "triwulan iv": 4, "triwulan 4": 4, "q4": 4, "tw iv": 4, "tw 4": 4, "iv": 4,
}


def _parse_triwulan_num(label: str) -> int:
    """Konversi label triwulan ke integer 1-4 secara aman."""
    key = str(label).strip().lower()
    if key in _TRIWULAN_MAP:
        return _TRIWULAN_MAP[key]
    # Fallback: cari angka romawi atau digit di dalam string
    roman_match = re.search(r'\b(iv|iii|ii|i)\b', key)
    if roman_match:
        roman_to_int = {"i": 1, "ii": 2, "iii": 3, "iv": 4}
        return roman_to_int.get(roman_match.group(1), 1)
    digit_match = re.search(r'(\d)', key)
    if digit_match:
        d = int(digit_match.group(1))
        return d if 1 <= d <= 4 else 1
    return 1


def load_manifest() -> Dict[str, Any]:
    """Membaca manifest metadata data parquet."""
    manifest_path = DATA_DIR / "manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Error reading manifest: {e}")

    # Fallback auto-discovery
    fallback_provinces: Dict[str, Any] = {}
    if DATA_DIR.exists():
        for p_file in DATA_DIR.glob("*.parquet"):
            year_match = re.search(r"20\d{2}", p_file.stem)
            if year_match:
                y = year_match.group(0)
                prov_key = p_file.stem[:p_file.stem.find(y)-1]
                prov_name = " ".join([w.capitalize() for w in prov_key.split("_")])
                if prov_key not in fallback_provinces:
                    fallback_provinces[prov_key] = {"name": prov_name, "years": {}}
                if y not in fallback_provinces[prov_key]["years"]:
                    fallback_provinces[prov_key]["years"][y] = {"sheets": []}
                sheet_name = p_file.stem[p_file.stem.find(y)+5:]
                fallback_provinces[prov_key]["years"][y]["sheets"].append(sheet_name)

    return {"provinces": fallback_provinces, "auto_discovered": True}


def get_parquet_df(file_stem: str) -> Optional[pd.DataFrame]:
    """Membaca file parquet dengan in-memory cache dan multi-candidate path search."""
    if file_stem in _parquet_cache:
        return _parquet_cache[file_stem]

    candidate_paths = [
        DATA_DIR / f"{file_stem}.parquet",
        resolve_data_dir() / f"{file_stem}.parquet",
        Path(__file__).resolve().parent / "data" / f"{file_stem}.parquet",
        Path(__file__).resolve().parent.parent / "data" / f"{file_stem}.parquet",
    ]

    for fp in candidate_paths:
        if fp.exists():
            try:
                df = pd.read_parquet(fp)
                _parquet_cache[file_stem] = df
                return df
            except Exception as e:
                print(f"❌ Error reading parquet {fp}: {e}")

    return None


# ======================================================================================
# GROWTH RATE ANALYSIS ENGINE — Multi-Year Aggregation & Pertumbuhan
# ======================================================================================

def build_multi_year_df(province: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Menggabungkan semua file pdrb_triwulan.parquet lintas tahun untuk satu provinsi
    menjadi satu DataFrame panjang yang tersortir kronologis.

    Returns:
        (df_combined, available_years): DataFrame gabungan + daftar tahun tersedia.
    """
    cache_key = f"_multiyear_{province}"
    if cache_key in _multi_year_cache:
        df_cached = _multi_year_cache[cache_key]
        years = [str(y) for y in sorted(df_cached["_tahun"].unique().tolist())]
        return df_cached.copy(), years

    manifest = load_manifest()
    prov_info = manifest.get("provinces", {}).get(province, {})
    available_years = sorted(list(prov_info.get("years", {}).keys())) if prov_info.get("years") else []

    # Fallback: scan data directory
    if not available_years:
        for fp in DATA_DIR.glob(f"{province}_*_pdrb_triwulan.parquet"):
            ym = re.search(r"(\d{4})", fp.stem)
            if ym:
                available_years.append(ym.group(1))
        available_years = sorted(set(available_years))

    frames = []
    for yr in available_years:
        df_yr = get_parquet_df(f"{province}_{yr}_pdrb_triwulan")
        if df_yr is not None and "Triwulan" in df_yr.columns:
            df_copy = df_yr.copy()
            df_copy["_tahun"] = int(yr)
            df_copy["_triwulan_num"] = df_copy["Triwulan"].apply(_parse_triwulan_num)
            # Kolom sorting kronologis: tahun * 10 + triwulan_num
            df_copy["_time_order"] = df_copy["_tahun"] * 10 + df_copy["_triwulan_num"]
            frames.append(df_copy)

    if not frames:
        return pd.DataFrame(), available_years

    df_combined = pd.concat(frames, ignore_index=True)
    df_combined = df_combined.sort_values("_time_order").reset_index(drop=True)

    _multi_year_cache[cache_key] = df_combined
    return df_combined.copy(), available_years


def compute_growth_df(df: pd.DataFrame, periods: int = 4) -> pd.DataFrame:
    """
    Menghitung laju pertumbuhan (pct_change) pada seluruh kolom numerik.

    Args:
        df: DataFrame multi-tahun tersortir kronologis (dari build_multi_year_df).
        periods: 4 untuk YoY (year-over-year), 1 untuk QoQ (quarter-over-quarter).

    Returns:
        DataFrame dengan kolom numerik berisi pertumbuhan (%), tanpa inf/nan.
    """
    if df.empty:
        return df

    df_g = df.copy()
    # Identifikasi kolom numerik (kecuali kolom internal _*)
    numeric_cols = [c for c in df_g.columns
                    if c not in ("Triwulan", "_tahun", "_triwulan_num", "_time_order")
                    and pd.api.types.is_numeric_dtype(df_g[c])]

    for col in numeric_cols:
        df_g[col] = df_g[col].pct_change(periods=periods) * 100.0

    # Guard: ganti inf/-inf dengan NaN sebelum dropna
    df_g.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop baris yang punya NaN di kolom numerik (baris awal akibat lag)
    df_g = df_g.dropna(subset=numeric_cols, how="all").reset_index(drop=True)

    return df_g


def _compute_correlation_matrix_from_growth(
    df_growth: pd.DataFrame,
    category: str = "lu"
) -> pd.DataFrame:
    """
    Menghitung matriks korelasi (+ p-value) antara setiap sektor PDRB
    dan 3 metrik transportasi dari data pertumbuhan.

    Args:
        df_growth: DataFrame pertumbuhan dari compute_growth_df().
        category: "lu" (Lapangan Usaha) atau "peng" (Pengeluaran).

    Returns:
        DataFrame dengan kolom: [label_col, "Tipe PDRB", "Korelasi dgn Penumpang",
        "p-value Penumpang", "Korelasi dgn Bagasi", "p-value Bagasi",
        "Korelasi dgn Barang", "p-value Barang"]
    """
    p_col = next((c for c in df_growth.columns if "penumpang" in c.lower()), None)
    bag_col = next((c for c in df_growth.columns if "bagasi" in c.lower()), None)
    bar_col = next((c for c in df_growth.columns if "barang" in c.lower()), None)

    prefix = "LU (HK)" if category == "lu" else "Peng (HK)"
    label_col = "Lapangan Usaha" if category == "lu" else "Komponen Pengeluaran"
    tipe_label = "Harga Konstan (HK)"

    # Temukan semua kolom sektor HK
    sector_cols = [c for c in df_growth.columns if c.lower().startswith(prefix.lower())]

    rows = []
    for sec_col in sector_cols:
        clean_name = re.sub(r"^(LU|Peng)\s*\([A-Z]+\)\s*-\s*", "", sec_col).strip()

        row: Dict[str, Any] = {label_col: clean_name, "Tipe PDRB": tipe_label}

        for metric_label, metric_col in [("Penumpang", p_col), ("Bagasi", bag_col), ("Barang", bar_col)]:
            cor_key = f"Korelasi dgn {metric_label}"
            pv_key = f"p-value {metric_label}"

            if metric_col is None or metric_col not in df_growth.columns:
                row[cor_key] = None
                row[pv_key] = None
                continue

            # Ambil pasangan data valid (non-NaN)
            mask = df_growth[[sec_col, metric_col]].dropna()
            x_vals = mask[sec_col].values.astype(float)
            y_vals = mask[metric_col].values.astype(float)

            if len(x_vals) >= 3:
                try:
                    r_val, p_val = pearsonr(x_vals, y_vals)
                    row[cor_key] = float(r_val) if not np.isnan(r_val) else None
                    row[pv_key] = float(p_val) if not np.isnan(p_val) else None
                except Exception:
                    row[cor_key] = None
                    row[pv_key] = None
            else:
                row[cor_key] = None
                row[pv_key] = None

        rows.append(row)

    return pd.DataFrame(rows)


# ======================================================================================
# API ENDPOINTS
# ======================================================================================

@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    """Serve halaman dashboard utama."""
    manifest = load_manifest()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "title": "Dashboard Analisis PDRB & Transportasi",
            "manifest": manifest,
            "data_dir": str(DATA_DIR)
        }
    )


@app.get("/api/health")
async def health_check():
    """Health check dan diagnostic report."""
    data_files = [f.name for f in DATA_DIR.glob("*")] if DATA_DIR.exists() else []
    return {
        "status": "ok",
        "service": "KorelasiPDRB-FastAPI",
        "version": "4.0.0",
        "data_dir": str(DATA_DIR),
        "data_dir_exists": DATA_DIR.exists(),
        "total_data_files": len(data_files),
        "sample_files": data_files[:5]
    }


@app.get("/api/manifest")
async def get_manifest():
    """Mengembalikan seluruh manifest metadata."""
    return load_manifest()


@app.get("/api/options")
async def get_options(province: str = "sulawesi_selatan", year: str = "2024"):
    """Mengembalikan opsi filter (provinsi, tahun, sektor, dan tipe PDRB)."""
    manifest = load_manifest()
    
    lu_stem = f"{province}_{year}_pdrb_correl_lu"
    df_lu = get_parquet_df(lu_stem)
    
    sectors = []
    if df_lu is not None and "Lapangan Usaha" in df_lu.columns:
        sectors = sorted(df_lu["Lapangan Usaha"].dropna().unique().tolist())
    
    return {
        "manifest": manifest,
        "selected_province": province,
        "selected_year": year,
        "sectors": sectors,
        "transport_metrics": [
            {"key": "penumpang", "label": "✈️ Penumpang (Orang)"},
            {"key": "bagasi", "label": "🧳 Bagasi (Kg)"},
            {"key": "barang", "label": "📦 Barang/Kargo (Kg)"}
        ]
    }


@app.get("/api/correlations")
async def get_correlations(
    province: str = "sulawesi_selatan",
    year: str = "2024",
    type: str = "lu",
    filter_mode: str = "all",
    sort_by: str = "default",
    pdrb_type: str = "all",
    search: str = "",
    analysis_mode: str = "growth_yoy"
):
    """
    Mengembalikan data matriks korelasi (Lapangan Usaha atau Pengeluaran)
    dengan dukungan filter 6 sektor utama (G, F, C, H, I, E), sorting, search,
    dan mode analisis: growth_yoy, growth_qoq, abs_hk, abs_hb.
    """
    is_growth = analysis_mode.startswith("growth_")
    label_col = "Lapangan Usaha" if type == "lu" else "Komponen Pengeluaran"
    data_warning = None

    if is_growth:
        # === MODE GROWTH: Hitung korelasi on-the-fly dari data pertumbuhan multi-tahun ===
        periods = 4 if analysis_mode == "growth_yoy" else 1
        df_multi, avail_years = build_multi_year_df(province)

        if df_multi.empty:
            return {
                "type": type,
                "analysis_mode": analysis_mode,
                "count": 0,
                "rows": [],
                "error": f"Data triwulan multi-tahun untuk {province} tidak ditemukan."
            }

        # Periksa kecukupan data untuk YoY
        total_quarters = len(df_multi)
        if analysis_mode == "growth_yoy" and total_quarters <= 4:
            data_warning = (
                f"Data hanya tersedia {total_quarters} triwulan. "
                f"YoY membutuhkan >4 triwulan. Pertimbangkan gunakan mode QoQ."
            )

        # Notifikasi khusus jika user memilih tahun 2021 pada YoY
        if analysis_mode == "growth_yoy" and avail_years and str(year) == str(avail_years[0]):
            data_warning = (
                f"Tahun {year} adalah tahun pertama dalam dataset. "
                f"Tidak ada data periode sebelumnya untuk menghitung YoY. "
                f"Korelasi akan dihitung dari seluruh tahun yang tersedia (pooled)."
            )

        df_growth = compute_growth_df(df_multi, periods=periods)

        if df_growth.empty:
            return {
                "type": type,
                "analysis_mode": analysis_mode,
                "count": 0,
                "rows": [],
                "data_warning": data_warning,
                "error": "Data pertumbuhan kosong setelah kalkulasi (kemungkinan rentang tahun terlalu pendek)."
            }

        # Hitung matriks korelasi dari data pertumbuhan (pooled seluruh tahun)
        df_view = _compute_correlation_matrix_from_growth(df_growth, category=type)

        analysis_label = "Pertumbuhan YoY (%)" if analysis_mode == "growth_yoy" else "Pertumbuhan QoQ (%)"
        n_observations = len(df_growth)

    else:
        # === MODE ABSOLUT: Gunakan file korelasi statis yang sudah ada ===
        stem = f"{province}_{year}_pdrb_correl_lu" if type == "lu" else f"{province}_{year}_pdrb_correl_peng"
        df = get_parquet_df(stem)

        if df is None:
            return {
                "type": type,
                "analysis_mode": analysis_mode,
                "count": 0,
                "rows": [],
                "error": f"Data korelasi {stem}.parquet tidak ditemukan di {DATA_DIR}."
            }

        df_view = df.copy()

        # Filter Tipe PDRB (HB / HK) untuk mode absolut
        effective_pdrb_type = "HK" if analysis_mode == "abs_hk" else ("HB" if analysis_mode == "abs_hb" else pdrb_type)
        if effective_pdrb_type == "HB" and "Tipe PDRB" in df_view.columns:
            df_view = df_view[df_view["Tipe PDRB"].str.contains("HB", case=False, na=False)]
        elif effective_pdrb_type == "HK" and "Tipe PDRB" in df_view.columns:
            df_view = df_view[df_view["Tipe PDRB"].str.contains("HK", case=False, na=False)]

        analysis_label = "Nilai Riil (Harga Konstan)" if analysis_mode == "abs_hk" else "Nilai Nominal (Harga Berlaku)"
        n_observations = 4  # Single year

    # === FILTERS (berlaku untuk semua mode) ===

    # Filter Pencarian Nama Sektor
    if search and label_col in df_view.columns:
        df_view = df_view[df_view[label_col].str.contains(search, case=False, na=False)]

    # Filter Mode
    if filter_mode == "6_core" and type == "lu":
        core_keywords = [
            "perdagangan",
            "konstruksi",
            "industri pengolahan",
            "transportasi dan pergudangan",
            "akomodasi",
            "pengadaan air"
        ]
        pattern = "|".join(core_keywords)
        df_view = df_view[df_view[label_col].str.contains(pattern, case=False, na=False)]

    elif filter_mode == "top5_high" and "Korelasi dgn Penumpang" in df_view.columns:
        df_view = df_view.sort_values(by="Korelasi dgn Penumpang", ascending=False).head(10)
    elif filter_mode == "top5_low" and "Korelasi dgn Penumpang" in df_view.columns:
        df_view = df_view.sort_values(by="Korelasi dgn Penumpang", ascending=True).head(10)
    elif filter_mode == "strong_70":
        cor_cols = [c for c in df_view.columns if "Korelasi" in c]
        if cor_cols:
            cond = df_view[cor_cols].abs().ge(0.70).any(axis=1)
            df_view = df_view[cond]
    elif filter_mode == "transport_only" and label_col in df_view.columns:
        df_view = df_view[df_view[label_col].str.contains("transportasi", case=False, na=False)]

    # Sorting
    if sort_by == "p_desc" and "Korelasi dgn Penumpang" in df_view.columns:
        df_view = df_view.sort_values(by="Korelasi dgn Penumpang", ascending=False)
    elif sort_by == "p_asc" and "Korelasi dgn Penumpang" in df_view.columns:
        df_view = df_view.sort_values(by="Korelasi dgn Penumpang", ascending=True)
    elif sort_by == "bag_desc" and "Korelasi dgn Bagasi" in df_view.columns:
        df_view = df_view.sort_values(by="Korelasi dgn Bagasi", ascending=False)
    elif sort_by == "bag_asc" and "Korelasi dgn Bagasi" in df_view.columns:
        df_view = df_view.sort_values(by="Korelasi dgn Bagasi", ascending=True)
    elif sort_by == "bar_desc" and "Korelasi dgn Barang" in df_view.columns:
        df_view = df_view.sort_values(by="Korelasi dgn Barang", ascending=False)
    elif sort_by == "bar_asc" and "Korelasi dgn Barang" in df_view.columns:
        df_view = df_view.sort_values(by="Korelasi dgn Barang", ascending=True)

    df_view = df_view.replace({np.nan: None})
    rows = df_view.to_dict(orient="records")

    result: Dict[str, Any] = {
        "type": type,
        "label_col": label_col,
        "count": len(rows),
        "rows": rows,
        "analysis_mode": analysis_mode,
        "analysis_label": analysis_label,
        "n_observations": n_observations,
    }
    if data_warning:
        result["data_warning"] = data_warning

    return result


@app.get("/api/data")
async def get_dashboard_data(
    province: str = "sulawesi_selatan",
    year: str = "2024",
    sector: str = "Transportasi dan Pergudangan",
    category: str = "lu",
    tipe_pdrb: str = "HK",
    transport_metric: str = "penumpang",
    analysis_mode: str = "growth_yoy"
):
    """
    Endpoint agregasi lengkap: KPI, Data Triwulanan (raw, index 100, QoQ untuk HK & HB),
    Regresi Linear OLS dengan persamaan $y=mx+c$, $R^2$, $R$, p-value,
    dan ringkasan sektoral untuk visualisasi frontend.

    Mendukung analysis_mode: growth_yoy, growth_qoq, abs_hk, abs_hb.
    Mode growth menggunakan pooled data multi-tahun untuk regresi (N≥12).
    """
    is_growth = analysis_mode.startswith("growth_")
    data_warning = None

    # Saat mode growth, paksa HK (Harga Konstan / Riil)
    if is_growth:
        chosen_type = "HK"
    elif analysis_mode == "abs_hb":
        chosen_type = "HB"
    else:
        chosen_type = "HB" if str(tipe_pdrb).upper() == "HB" else "HK"

    # ==================================================================================
    # LOAD DATA TRIWULAN TAHUN TERPILIH (untuk KPI & time-series)
    # ==================================================================================
    stem_tri = f"{province}_{year}_pdrb_triwulan"
    df_tri = get_parquet_df(stem_tri)

    if df_tri is None:
        raise HTTPException(
            status_code=404,
            detail=f"Data triwulan {stem_tri}.parquet tidak ditemukan di {DATA_DIR}."
        )

    # Kolom Transportasi
    p_col = next((c for c in df_tri.columns if "penumpang" in c.lower()), None)
    bag_col = next((c for c in df_tri.columns if "bagasi" in c.lower()), None)
    bar_col = next((c for c in df_tri.columns if "barang" in c.lower()), None)

    # Kolom Total PDRB
    lu_hk_cols = [c for c in df_tri.columns if c.startswith("LU (HK)")]
    lu_hb_cols = [c for c in df_tri.columns if c.startswith("LU (HB)")]
    total_lu_hk_col = next((c for c in lu_hk_cols if "produk domestik" in c.lower() or "pdrb" in c.lower()), None)

    # Cari kolom target (Lapangan Usaha atau Pengeluaran) untuk HK dan HB
    clean_sec = re.sub(r'^(LU|Peng)\s*\([A-Z]+\)\s*-\s*', '', sector, flags=re.IGNORECASE).strip().lower()
    prefix_hk = "Peng (HK)" if category == "peng" else "LU (HK)"
    prefix_hb = "Peng (HB)" if category == "peng" else "LU (HB)"

    matching_hk = [c for c in df_tri.columns if clean_sec in c.lower() and prefix_hk.lower() in c.lower()]
    matching_hb = [c for c in df_tri.columns if clean_sec in c.lower() and prefix_hb.lower() in c.lower()]

    col_hk = matching_hk[0] if matching_hk else None
    col_hb = matching_hb[0] if matching_hb else None

    # Tentukan kolom PDRB aktif
    actual_tipe = chosen_type
    if chosen_type == "HB" and col_hb:
        sector_col = col_hb
        actual_tipe = "HB"
    elif col_hk:
        sector_col = col_hk
        actual_tipe = "HK"
    elif col_hb:
        sector_col = col_hb
        actual_tipe = "HB"
    else:
        sector_col = total_lu_hk_col or df_tri.columns[1]
        actual_tipe = "HK"

    active_label = re.sub(r'^(LU|Peng)\s*\([A-Z]+\)\s*-\s*', '', sector_col).strip()

    # ==================================================================================
    # 1. KPI Cards Calculation (selalu dari data tahun terpilih)
    # ==================================================================================
    tot_p = float(df_tri[p_col].sum()) if p_col else 0
    tot_bag = float(df_tri[bag_col].sum()) if bag_col else 0
    tot_bar = float(df_tri[bar_col].sum()) if bar_col else 0
    tot_pdrb_hk = float(df_tri[total_lu_hk_col].sum()) if total_lu_hk_col else 0

    def get_q4_growth(col):
        if col and col in df_tri and len(df_tri) > 1:
            q_last = float(df_tri[col].iloc[-1])
            q_prev = float(df_tri[col].iloc[-2])
            return ((q_last - q_prev) / q_prev * 100) if q_prev != 0 else 0
        return 0.0

    kpi = {
        "penumpang_total": tot_p,
        "penumpang_q4_growth": get_q4_growth(p_col),
        "bagasi_total": tot_bag,
        "bagasi_q4_growth": get_q4_growth(bag_col),
        "barang_total": tot_bar,
        "barang_q4_growth": get_q4_growth(bar_col),
        "pdrb_hk_total": tot_pdrb_hk,
        "pdrb_hk_q4_growth": get_q4_growth(total_lu_hk_col),
        "max_correl_sector": "Transportasi dan Pergudangan",
        "max_correl_val": 0.811,
        "max_correl_type": "HK"
    }

    # Sektor Terkorelasi Max dari df_correl_lu
    stem_lu = f"{province}_{year}_pdrb_correl_lu"
    df_lu = get_parquet_df(stem_lu)
    if df_lu is not None and "Korelasi dgn Penumpang" in df_lu.columns:
        df_f = df_lu[~df_lu["Lapangan Usaha"].str.contains("Produk Domestik|PDRB", case=False, na=False)]
        if not df_f.empty:
            top_row = df_f.sort_values(by="Korelasi dgn Penumpang", ascending=False).iloc[0]
            kpi["max_correl_sector"] = str(top_row["Lapangan Usaha"])
            kpi["max_correl_val"] = float(top_row["Korelasi dgn Penumpang"])
            kpi["max_correl_type"] = str(top_row.get("Tipe PDRB", ""))

    # ==================================================================================
    # 2. Time Series Data (Triwulanan) — selalu dari tahun terpilih
    # ==================================================================================
    triwulan_labels = df_tri["Triwulan"].tolist()
    
    def calc_idx100(series):
        base = series.iloc[0]
        return [(float(v) / float(base) * 100.0) if base != 0 and pd.notna(v) else 100.0 for v in series]

    def calc_qoq(series):
        pct = series.pct_change() * 100.0
        return [0.0 if pd.isna(v) else float(v) for v in pct]

    pdrb_series = df_tri[sector_col].astype(float)
    pdrb_hk_series = df_tri[col_hk].astype(float) if col_hk else pdrb_series
    pdrb_hb_series = df_tri[col_hb].astype(float) if col_hb else pdrb_series

    p_series = df_tri[p_col].astype(float) if p_col else pd.Series([0]*len(df_tri))
    bag_series = df_tri[bag_col].astype(float) if bag_col else pd.Series([0]*len(df_tri))
    bar_series = df_tri[bar_col].astype(float) if bar_col else pd.Series([0]*len(df_tri))

    pdrb_idx = calc_idx100(pdrb_series)
    pdrb_qoq = calc_qoq(pdrb_series)
    pdrb_hk_idx = calc_idx100(pdrb_hk_series)
    pdrb_hk_qoq = calc_qoq(pdrb_hk_series)
    pdrb_hb_idx = calc_idx100(pdrb_hb_series)
    pdrb_hb_qoq = calc_qoq(pdrb_hb_series)

    p_idx = calc_idx100(p_series)
    p_qoq = calc_qoq(p_series)
    bag_idx = calc_idx100(bag_series)
    bag_qoq = calc_qoq(bag_series)
    bar_idx = calc_idx100(bar_series)
    bar_qoq = calc_qoq(bar_series)

    # Jika mode growth, hitung juga pertumbuhan multi-tahun untuk time-series display
    growth_ts_data: Dict[str, List[float]] = {}
    if is_growth:
        df_multi_all, avail_years = build_multi_year_df(province)
        if not df_multi_all.empty:
            periods_g = 4 if analysis_mode == "growth_yoy" else 1
            df_g_all = compute_growth_df(df_multi_all, periods=periods_g)

            # Filter ke tahun terpilih untuk time-series display
            df_g_year = df_g_all[df_g_all["_tahun"] == int(year)]

            if not df_g_year.empty:
                # Cari kolom yang cocok di data growth
                g_sector_col = next((c for c in df_g_year.columns if clean_sec in c.lower() and prefix_hk.lower() in c.lower()), None)
                g_p_col = next((c for c in df_g_year.columns if "penumpang" in c.lower()), None)
                g_bag_col = next((c for c in df_g_year.columns if "bagasi" in c.lower()), None)
                g_bar_col = next((c for c in df_g_year.columns if "barang" in c.lower()), None)

                growth_ts_data["pdrb_growth"] = df_g_year[g_sector_col].fillna(0).tolist() if g_sector_col and g_sector_col in df_g_year.columns else [0]*len(df_g_year)
                growth_ts_data["penumpang_growth"] = df_g_year[g_p_col].fillna(0).tolist() if g_p_col and g_p_col in df_g_year.columns else [0]*len(df_g_year)
                growth_ts_data["bagasi_growth"] = df_g_year[g_bag_col].fillna(0).tolist() if g_bag_col and g_bag_col in df_g_year.columns else [0]*len(df_g_year)
                growth_ts_data["barang_growth"] = df_g_year[g_bar_col].fillna(0).tolist() if g_bar_col and g_bar_col in df_g_year.columns else [0]*len(df_g_year)
            else:
                # Tahun pertama pada YoY: semua growth = NaN → kosong
                growth_mode_label = "YoY" if analysis_mode == "growth_yoy" else "QoQ"
                data_warning = (
                    f"Tahun {year} tidak memiliki data pertumbuhan {growth_mode_label} "
                    f"(tahun awal dataset). Grafik pertumbuhan kosong untuk tahun ini."
                )

    triwulan_data = []
    for i, tw in enumerate(triwulan_labels):
        row_data: Dict[str, Any] = {
            "triwulan": tw,
            "pdrb_raw": float(pdrb_series.iloc[i]),
            "pdrb_idx": float(pdrb_idx[i]),
            "pdrb_qoq": float(pdrb_qoq[i]),
            "pdrb_hk_raw": float(pdrb_hk_series.iloc[i]),
            "pdrb_hk_idx": float(pdrb_hk_idx[i]),
            "pdrb_hk_qoq": float(pdrb_hk_qoq[i]),
            "pdrb_hb_raw": float(pdrb_hb_series.iloc[i]),
            "pdrb_hb_idx": float(pdrb_hb_idx[i]),
            "pdrb_hb_qoq": float(pdrb_hb_qoq[i]),
            "penumpang_raw": float(p_series.iloc[i]),
            "penumpang_idx": float(p_idx[i]),
            "penumpang_qoq": float(p_qoq[i]),
            "bagasi_raw": float(bag_series.iloc[i]),
            "bagasi_idx": float(bag_idx[i]),
            "bagasi_qoq": float(bag_qoq[i]),
            "barang_raw": float(bar_series.iloc[i]),
            "barang_idx": float(bar_idx[i]),
            "barang_qoq": float(bar_qoq[i]),
        }
        # Tambah kolom growth jika tersedia
        if growth_ts_data:
            row_data["pdrb_growth"] = float(growth_ts_data.get("pdrb_growth", [0]*4)[i]) if i < len(growth_ts_data.get("pdrb_growth", [])) else 0.0
            row_data["penumpang_growth"] = float(growth_ts_data.get("penumpang_growth", [0]*4)[i]) if i < len(growth_ts_data.get("penumpang_growth", [])) else 0.0
            row_data["bagasi_growth"] = float(growth_ts_data.get("bagasi_growth", [0]*4)[i]) if i < len(growth_ts_data.get("bagasi_growth", [])) else 0.0
            row_data["barang_growth"] = float(growth_ts_data.get("barang_growth", [0]*4)[i]) if i < len(growth_ts_data.get("barang_growth", [])) else 0.0

        triwulan_data.append(row_data)

    # ==================================================================================
    # 3. OLS Linear Regression
    # ==================================================================================
    if transport_metric == "bagasi" and bag_col:
        y_raw_series = bag_series
        label_y = "Bagasi (Kg)"
    elif transport_metric == "barang" and bar_col:
        y_raw_series = bar_series
        label_y = "Barang/Kargo (Kg)"
    else:
        y_raw_series = p_series
        label_y = "Penumpang (Orang)"

    def compute_regression(x_s: pd.Series, y_s: pd.Series, label_x_str: str, label_y_str: str,
                           point_labels: Optional[List[str]] = None) -> Dict[str, Any]:
        x_vals = x_s.values.astype(float)
        y_vals = y_s.values.astype(float)
        valid_m = ~(np.isnan(x_vals) | np.isnan(y_vals) | np.isinf(x_vals) | np.isinf(y_vals))
        x_clean = x_vals[valid_m]
        y_clean = y_vals[valid_m]

        if point_labels is None:
            point_labels = triwulan_labels
        lbl_clean = [point_labels[k] for k in range(len(point_labels)) if k < len(valid_m) and valid_m[k]]

        res: Dict[str, Any] = {
            "slope": 0.0,
            "intercept": 0.0,
            "r_val": 0.0,
            "r_squared": 0.0,
            "p_value": 1.0,
            "equation": "y = 0x + 0",
            "label_x": label_x_str,
            "label_y": label_y_str,
            "points": [],
            "trend_line": [],
            "n_points": 0
        }

        if len(x_clean) >= 2:
            slope, intercept = np.polyfit(x_clean, y_clean, 1)
            r_matrix = np.corrcoef(x_clean, y_clean)
            r_val = float(r_matrix[0, 1]) if not np.isnan(r_matrix[0, 1]) else 0.0
            r_squared = float(r_val ** 2)

            # p-value dari pearsonr
            try:
                _, p_val = pearsonr(x_clean, y_clean)
                p_val = float(p_val) if not np.isnan(p_val) else 1.0
            except Exception:
                p_val = 1.0

            sign_c = "+" if intercept >= 0 else "-"
            slope_str = f"{slope:.3e}" if abs(slope) < 0.0001 else f"{slope:.4f}"
            eq_str = f"y = {slope_str}x {sign_c} {abs(intercept):,.2f}"

            points = [{"triwulan": lbl_clean[k], "x": float(x_clean[k]), "y": float(y_clean[k])} for k in range(len(x_clean))]

            x_min, x_max = float(x_clean.min()), float(x_clean.max())
            trend_line = [
                {"x": x_min, "y": float(slope * x_min + intercept)},
                {"x": x_max, "y": float(slope * x_max + intercept)}
            ]

            res.update({
                "slope": float(slope),
                "intercept": float(intercept),
                "r_val": r_val,
                "r_squared": r_squared,
                "p_value": p_val,
                "equation": eq_str,
                "points": points,
                "trend_line": trend_line,
                "n_points": len(x_clean)
            })
        return res

    # --- Regresi untuk mode Growth (pooled multi-tahun, N≥12) ---
    if is_growth:
        df_multi_all, avail_years = build_multi_year_df(province)
        periods_g = 4 if analysis_mode == "growth_yoy" else 1
        df_g_pooled = compute_growth_df(df_multi_all, periods=periods_g)

        growth_mode_label = "YoY" if analysis_mode == "growth_yoy" else "QoQ"
        label_y_growth = f"Pertumbuhan {label_y.split('(')[0].strip()} ({growth_mode_label} %)"

        # Notifikasi tahun 2021
        if analysis_mode == "growth_yoy" and avail_years and year == avail_years[0]:
            data_warning = (
                f"Tahun {year} adalah tahun pertama dalam dataset. "
                f"Regresi menggunakan pooled data seluruh tahun ({', '.join(avail_years)}) agar N cukup."
            )

        if not df_g_pooled.empty:
            g_sector_col = next((c for c in df_g_pooled.columns if clean_sec in c.lower() and prefix_hk.lower() in c.lower()), None)
            g_p_col = next((c for c in df_g_pooled.columns if "penumpang" in c.lower()), None)
            g_bag_col = next((c for c in df_g_pooled.columns if "bagasi" in c.lower()), None)
            g_bar_col = next((c for c in df_g_pooled.columns if "barang" in c.lower()), None)

            # Pilih Y untuk regresi
            if transport_metric == "bagasi" and g_bag_col and g_bag_col in df_g_pooled.columns:
                g_y_col = g_bag_col
            elif transport_metric == "barang" and g_bar_col and g_bar_col in df_g_pooled.columns:
                g_y_col = g_bar_col
            elif g_p_col and g_p_col in df_g_pooled.columns:
                g_y_col = g_p_col
            else:
                g_y_col = None

            # Buat label triwulan untuk pooled data
            pooled_labels = [f"{int(r['_tahun'])} Q{int(r['_triwulan_num'])}" for _, r in df_g_pooled.iterrows()]

            if g_sector_col and g_sector_col in df_g_pooled.columns and g_y_col:
                x_pooled = df_g_pooled[g_sector_col].astype(float)
                y_pooled = df_g_pooled[g_y_col].astype(float)

                label_x_growth = f"Pertumbuhan PDRB: {active_label} ({growth_mode_label} %)"
                reg_result = compute_regression(x_pooled, y_pooled, label_x_growth, label_y_growth, pooled_labels)
            else:
                reg_result = compute_regression(pd.Series([0]), pd.Series([0]), f"Pertumbuhan PDRB ({growth_mode_label} %)", label_y_growth)
        else:
            reg_result = compute_regression(pd.Series([0]), pd.Series([0]), f"Pertumbuhan PDRB ({growth_mode_label} %)", label_y_growth)

        # Untuk mode growth, HK dan HB regression diarahkan ke result yang sama (karena growth selalu HK)
        reg_hk = reg_result
        reg_hb = reg_result

    else:
        # --- Regresi absolut (perilaku lama, backward-compatible) ---
        reg_result = compute_regression(pdrb_series, y_raw_series, f"{active_label} ({actual_tipe})", label_y)
        reg_hk = compute_regression(pdrb_hk_series, y_raw_series, f"{active_label} (HK)", label_y)
        reg_hb = compute_regression(pdrb_hb_series, y_raw_series, f"{active_label} (HB)", label_y)

    # ==================================================================================
    # 3b. Regresi Multi-Tahun (Tahunan: 2021 s.d. 2024) — hanya untuk mode absolut
    # ==================================================================================
    prov_manifest = load_manifest()
    prov_info = prov_manifest.get("provinces", {}).get(province, {})
    available_years = sorted(list(prov_info.get("years", {}).keys())) if (prov_info and prov_info.get("years")) else ["2021", "2022", "2023", "2024"]

    if not is_growth:
        annual_pts_hk = []
        annual_pts_hb = []
        for yr in available_years:
            df_yr = get_parquet_df(f"{province}_{yr}_pdrb_triwulan")
            if df_yr is not None:
                m_hk = [c for c in df_yr.columns if clean_sec in c.lower() and prefix_hk.lower() in c.lower()]
                m_hb = [c for c in df_yr.columns if clean_sec in c.lower() and prefix_hb.lower() in c.lower()]
                yr_p = next((c for c in df_yr.columns if "penumpang" in c.lower()), None)
                yr_bag = next((c for c in df_yr.columns if "bagasi" in c.lower()), None)
                yr_bar = next((c for c in df_yr.columns if "barang" in c.lower()), None)

                if transport_metric == "bagasi" and yr_bag:
                    y_col = yr_bag
                elif transport_metric == "barang" and yr_bar:
                    y_col = yr_bar
                else:
                    y_col = yr_p

                if y_col and y_col in df_yr:
                    tot_y = float(df_yr[y_col].sum())
                    if m_hk and m_hk[0] in df_yr:
                        tot_x_hk = float(df_yr[m_hk[0]].sum())
                        annual_pts_hk.append({"label": yr, "x": tot_x_hk, "y": tot_y})
                    if m_hb and m_hb[0] in df_yr:
                        tot_x_hb = float(df_yr[m_hb[0]].sum())
                        annual_pts_hb.append({"label": yr, "x": tot_x_hb, "y": tot_y})

        def compute_regression_from_pts(pts: List[Dict[str, Any]], label_x_str: str, label_y_str: str) -> Dict[str, Any]:
            res: Dict[str, Any] = {
                "slope": 0.0,
                "intercept": 0.0,
                "r_val": 0.0,
                "r_squared": 0.0,
                "p_value": 1.0,
                "equation": "y = 0x + 0",
                "label_x": label_x_str,
                "label_y": label_y_str,
                "points": [],
                "trend_line": [],
                "n_points": 0
            }
            if len(pts) >= 2:
                x_arr = np.array([p["x"] for p in pts], dtype=float)
                y_arr = np.array([p["y"] for p in pts], dtype=float)
                valid_m = ~(np.isnan(x_arr) | np.isnan(y_arr))
                xc = x_arr[valid_m]
                yc = y_arr[valid_m]
                lbls = [pts[k]["label"] for k in range(len(pts)) if valid_m[k]]

                if len(xc) >= 2:
                    slope, intercept = np.polyfit(xc, yc, 1)
                    r_mat = np.corrcoef(xc, yc)
                    r_val = float(r_mat[0, 1]) if not np.isnan(r_mat[0, 1]) else 0.0
                    r_sq = float(r_val ** 2)

                    try:
                        _, p_val = pearsonr(xc, yc)
                        p_val = float(p_val) if not np.isnan(p_val) else 1.0
                    except Exception:
                        p_val = 1.0

                    sign_c = "+" if intercept >= 0 else "-"
                    slope_str = f"{slope:.3e}" if abs(slope) < 0.0001 else f"{slope:.4f}"
                    eq_str = f"y = {slope_str}x {sign_c} {abs(intercept):,.2f}"

                    points = [{"triwulan": lbls[k], "label": lbls[k], "x": float(xc[k]), "y": float(yc[k])} for k in range(len(xc))]
                    x_min, x_max = float(xc.min()), float(xc.max())
                    trend_line = [
                        {"x": x_min, "y": float(slope * x_min + intercept)},
                        {"x": x_max, "y": float(slope * x_max + intercept)}
                    ]
                    res.update({
                        "slope": float(slope),
                        "intercept": float(intercept),
                        "r_val": r_val,
                        "r_squared": r_sq,
                        "p_value": p_val,
                        "equation": eq_str,
                        "points": points,
                        "trend_line": trend_line,
                        "n_points": len(xc)
                    })
            return res

        annual_pts_active = annual_pts_hb if actual_tipe == "HB" else annual_pts_hk
        reg_annual = compute_regression_from_pts(annual_pts_active, f"{active_label} ({actual_tipe}) - Tahunan", label_y)
        reg_annual_hk = compute_regression_from_pts(annual_pts_hk, f"{active_label} (HK) - Tahunan", label_y)
        reg_annual_hb = compute_regression_from_pts(annual_pts_hb, f"{active_label} (HB) - Tahunan", label_y)
    else:
        # Mode growth: regresi tahunan tidak relevan — gunakan pooled growth sebagai fallback
        reg_annual = reg_result
        reg_annual_hk = reg_result
        reg_annual_hb = reg_result

    # ==================================================================================
    # 4. Sektor Lapangan Usaha Top 10 Summary
    # ==================================================================================
    target_lu_cols = lu_hb_cols if actual_tipe == "HB" else lu_hk_cols
    sectors_summary = []
    for col in target_lu_cols:
        clean_name = re.sub(r"^LU\s*\([A-Z]+\)\s*-\s*", "", col)
        if "produk domestik" not in clean_name.lower() and "pdrb" not in clean_name.lower():
            tot_val = float(df_tri[col].sum())
            sectors_summary.append({"sector": clean_name, "total_output": tot_val})

    sectors_summary = sorted(sectors_summary, key=lambda x: x["total_output"], reverse=True)
    total_lu_sum = sum(s["total_output"] for s in sectors_summary) or 1.0
    for s in sectors_summary:
        s["percentage"] = round((s["total_output"] / total_lu_sum) * 100.0, 2)

    # ==================================================================================
    # Build Response
    # ==================================================================================
    response_data: Dict[str, Any] = {
        "province": province,
        "year": year,
        "active_sector": active_label,
        "category": category,
        "tipe_pdrb": actual_tipe,
        "has_hk": col_hk is not None,
        "has_hb": col_hb is not None,
        "analysis_mode": analysis_mode,
        "kpi": kpi,
        "triwulan_data": triwulan_data,
        "regression": reg_result,
        "regression_hk": reg_hk,
        "regression_hb": reg_hb,
        "regression_annual": reg_annual,
        "regression_annual_hk": reg_annual_hk,
        "regression_annual_hb": reg_annual_hb,
        "sectors_summary": sectors_summary[:10],
        "all_sectors_summary": sectors_summary,
        "available_years": available_years,
    }
    if data_warning:
        response_data["data_warning"] = data_warning
    if is_growth and growth_ts_data:
        response_data["growth_ts_available"] = True

    return response_data


@app.get("/api/raw_sheet")
async def get_raw_sheet(
    province: str = "sulawesi_selatan",
    year: str = "2024",
    sheet: str = "pdrb_triwulan"
):
    """Membaca raw sheet parquet tabular untuk ekspor dan inspeksi."""
    stem = f"{province}_{year}_{sheet}"
    df = get_parquet_df(stem)
    if df is None:
        raise HTTPException(status_code=404, detail=f"Sheet {stem}.parquet tidak ditemukan di {DATA_DIR}.")
    
    df_clean = df.replace({np.nan: None})
    return {
        "stem": stem,
        "columns": df_clean.columns.tolist(),
        "rows": df_clean.to_dict(orient="records"),
        "count": len(df_clean)
    }
