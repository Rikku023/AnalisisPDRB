"""
========================================================================================
FastAPI Serverless Application - Korelasi PDRB & Transportasi (Vercel Serverless Ready)
Dynamic Path Resolution, In-Memory Parquet Cache, dan Full Diagnostic Logging.
========================================================================================
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np


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

print(f"🚀 [INIT] DATA_DIR resolved to: {DATA_DIR} (Files: {len(list(DATA_DIR.glob('*'))) if DATA_DIR.exists() else 0})")
print(f"🚀 [INIT] TEMPLATES_DIR resolved to: {TEMPLATES_DIR}")
print(f"🚀 [INIT] PUBLIC_DIR resolved to: {PUBLIC_DIR}")

# Inisialisasi FastAPI App
app = FastAPI(
    title="Korelasi PDRB & Transportasi API",
    description="Serverless API & Dashboard Analisis PDRB dan Transportasi BPS",
    version="3.1.0"
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


def load_manifest() -> Dict[str, Any]:
    """Membaca manifest metadata data parquet."""
    manifest_path = DATA_DIR / "manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Error reading manifest: {e}")

    # Fallback auto-discovery dari file parquet yang ada jika manifest.json tidak terbaca
    fallback_provinces: Dict[str, Any] = {}
    if DATA_DIR.exists():
        for p_file in DATA_DIR.glob("*.parquet"):
            parts = p_file.stem.split("_")
            if len(parts) >= 3:
                # Format: {prov_key}_{year}_{sheet}
                # e.g., sulawesi_selatan_2024_pdrb_triwulan
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

    # Cek kandidat path
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

    print(f"⚠️ Parquet file not found: {file_stem}.parquet in candidates: {candidate_paths}")
    return None


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
        "version": "3.1.0",
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
    
    # Ambil sektor yang tersedia dari df_correl_lu atau df_triwulan
    lu_stem = f"{province}_{year}_pdrb_correl_lu"
    df_lu = get_parquet_df(lu_stem)
    
    sectors = []
    if df_lu is not None and "Lapangan Usaha" in df_lu.columns:
        sectors = sorted(df_lu["Lapangan Usaha"].dropna().unique().tolist())
    
    # Fallback sectors standar jika parquet belum ter-load
    if not sectors:
        sectors = [
            "Pertanian, Kehutanan, dan Perikanan",
            "Pertambangan dan Penggalian",
            "Industri Pengolahan",
            "Pengadaan Listrik dan Gas",
            "Pengadaan Air, Pengelolaan Sampah, Limbah, dan Daur Ulang",
            "Konstruksi",
            "Perdagangan Besar dan Eceran; Reparasi Mobil dan Sepeda Motor",
            "Transportasi dan Pergudangan",
            "Penyediaan Akomodasi dan Makan Minum",
            "Informasi dan Komunikasi",
            "Jasa Keuangan dan Asuransi",
            "Real Estat",
            "Jasa Perusahaan",
            "Administrasi Pemerintahan, Pertahanan dan Jaminan Sosial Wajib",
            "Jasa Pendidikan",
            "Jasa Kesehatan dan Kegiatan Sosial",
            "Jasa lainnya"
        ]
    
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
    search: str = ""
):
    """
    Mengembalikan data matriks korelasi (Lapangan Usaha atau Pengeluaran)
    dengan dukungan filter 6 sektor utama (G, F, C, H, I, E), sorting, dan search.
    """
    stem = f"{province}_{year}_pdrb_correl_lu" if type == "lu" else f"{province}_{year}_pdrb_correl_peng"
    df = get_parquet_df(stem)

    if df is None:
        return {
            "type": type,
            "count": 0,
            "rows": [],
            "error": f"Data korelasi {stem}.parquet tidak ditemukan di {DATA_DIR}."
        }

    df_view = df.copy()
    label_col = "Lapangan Usaha" if type == "lu" else "Komponen Pengeluaran"

    # Filter Tipe PDRB (HB / HK)
    if pdrb_type == "HB" and "Tipe PDRB" in df_view.columns:
        df_view = df_view[df_view["Tipe PDRB"].str.contains("HB", case=False, na=False)]
    elif pdrb_type == "HK" and "Tipe PDRB" in df_view.columns:
        df_view = df_view[df_view["Tipe PDRB"].str.contains("HK", case=False, na=False)]

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

    # Convert NaN to None for clean JSON serialization
    df_view = df_view.replace({np.nan: None})
    rows = df_view.to_dict(orient="records")
    return {
        "type": type,
        "count": len(rows),
        "rows": rows
    }


@app.get("/api/data")
async def get_dashboard_data(
    province: str = "sulawesi_selatan",
    year: str = "2024",
    sector: str = "Transportasi dan Pergudangan",
    tipe_pdrb: str = "HK",
    transport_metric: str = "penumpang"
):
    """
    Endpoint agregasi lengkap: KPI, Data Triwulanan (raw, index 100, QoQ),
    Regresi Linear OLS (X=PDRB, Y=Transport) dengan persamaan $y=mx+c$, $R^2$, $R$,
    dan ringkasan sektoral untuk visualisasi frontend.
    """
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
    total_lu_hb_col = next((c for c in lu_hb_cols if "produk domestik" in c.lower() or "pdrb" in c.lower()), None)

    # Cari kolom sektor yang dipilih
    pdrb_prefix = "LU (HB)" if tipe_pdrb == "HB" else "LU (HK)"
    matching_sector_cols = [c for c in df_tri.columns if sector.lower() in c.lower() and pdrb_prefix in c]
    if not matching_sector_cols:
        matching_sector_cols = [c for c in df_tri.columns if sector.lower() in c.lower()]
    sector_col = matching_sector_cols[0] if matching_sector_cols else (total_lu_hk_col or df_tri.columns[1])

    # 1. KPI Cards Calculation
    tot_p = float(df_tri[p_col].sum()) if p_col else 0
    tot_bag = float(df_tri[bag_col].sum()) if bag_col else 0
    tot_bar = float(df_tri[bar_col].sum()) if bar_col else 0
    tot_pdrb_hk = float(df_tri[total_lu_hk_col].sum()) if total_lu_hk_col else 0

    # Pertumbuhan Q4 q-to-q
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

    # 2. Time Series Data (Triwulanan)
    triwulan_labels = df_tri["Triwulan"].tolist()
    
    def calc_idx100(series):
        base = series.iloc[0]
        return [(float(v) / float(base) * 100.0) if base != 0 and pd.notna(v) else 100.0 for v in series]

    def calc_qoq(series):
        pct = series.pct_change() * 100.0
        return [0.0 if pd.isna(v) else float(v) for v in pct]

    pdrb_series = df_tri[sector_col].astype(float)
    p_series = df_tri[p_col].astype(float) if p_col else pd.Series([0]*len(df_tri))
    bag_series = df_tri[bag_col].astype(float) if bag_col else pd.Series([0]*len(df_tri))
    bar_series = df_tri[bar_col].astype(float) if bar_col else pd.Series([0]*len(df_tri))

    triwulan_data = []
    for i, tw in enumerate(triwulan_labels):
        triwulan_data.append({
            "triwulan": tw,
            "pdrb_raw": float(pdrb_series.iloc[i]),
            "pdrb_idx": float(calc_idx100(pdrb_series)[i]),
            "pdrb_qoq": float(calc_qoq(pdrb_series)[i]),
            "penumpang_raw": float(p_series.iloc[i]),
            "penumpang_idx": float(calc_idx100(p_series)[i]),
            "penumpang_qoq": float(calc_qoq(p_series)[i]),
            "bagasi_raw": float(bag_series.iloc[i]),
            "bagasi_idx": float(calc_idx100(bag_series)[i]),
            "bagasi_qoq": float(calc_qoq(bag_series)[i]),
            "barang_raw": float(bar_series.iloc[i]),
            "barang_idx": float(calc_idx100(bar_series)[i]),
            "barang_qoq": float(calc_qoq(bar_series)[i]),
        })

    # 3. OLS Linear Regression (Sumbu X = PDRB, Sumbu Y = Transport Metric)
    if transport_metric == "bagasi" and bag_col:
        y_raw_series = bag_series
        label_y = "Bagasi (Kg)"
    elif transport_metric == "barang" and bar_col:
        y_raw_series = bar_series
        label_y = "Barang/Kargo (Kg)"
    else:
        y_raw_series = p_series
        label_y = "Penumpang (Orang)"

    x_vals = pdrb_series.values
    y_vals = y_raw_series.values
    valid_m = ~(np.isnan(x_vals) | np.isnan(y_vals))
    x_clean = x_vals[valid_m]
    y_clean = y_vals[valid_m]
    tw_clean = [triwulan_labels[k] for k in range(len(triwulan_labels)) if valid_m[k]]

    reg_result = {
        "slope": 0.0,
        "intercept": 0.0,
        "r_val": 0.0,
        "r_squared": 0.0,
        "equation": "y = 0x + 0",
        "label_x": f"PDRB: {sector} ({tipe_pdrb})",
        "label_y": label_y,
        "points": [],
        "trend_line": []
    }

    if len(x_clean) >= 2:
        slope, intercept = np.polyfit(x_clean, y_clean, 1)
        r_matrix = np.corrcoef(x_clean, y_clean)
        r_val = float(r_matrix[0, 1])
        r_squared = float(r_val ** 2)

        sign_c = "+" if intercept >= 0 else "-"
        slope_str = f"{slope:.3e}" if abs(slope) < 0.0001 else f"{slope:.4f}"
        eq_str = f"y = {slope_str}x {sign_c} {abs(intercept):,.2f}"

        points = [{"triwulan": tw_clean[k], "x": float(x_clean[k]), "y": float(y_clean[k])} for k in range(len(x_clean))]

        x_min, x_max = float(x_clean.min()), float(x_clean.max())
        trend_line = [
            {"x": x_min, "y": float(slope * x_min + intercept)},
            {"x": x_max, "y": float(slope * x_max + intercept)}
        ]

        reg_result.update({
            "slope": float(slope),
            "intercept": float(intercept),
            "r_val": r_val,
            "r_squared": r_squared,
            "equation": eq_str,
            "points": points,
            "trend_line": trend_line
        })

    # 4. Sektor Lapangan Usaha Top 10 Summary
    target_lu_cols = lu_hk_cols if tipe_pdrb == "HK" else lu_hb_cols
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

    return {
        "province": province,
        "year": year,
        "active_sector": sector,
        "tipe_pdrb": tipe_pdrb,
        "kpi": kpi,
        "triwulan_data": triwulan_data,
        "regression": reg_result,
        "sectors_summary": sectors_summary[:10],
        "all_sectors_summary": sectors_summary
    }


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
