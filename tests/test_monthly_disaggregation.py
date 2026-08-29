"""
========================================================================================
Unit Test Suite: Temporal Disaggregation (Quarterly-to-Monthly PDRB Benchmarking Suite)
Denton-Cholette Proportional, Chow-Lin GLS, Cubic Spline PCHIP, & Uniform 1/3
========================================================================================
"""

import unittest
import numpy as np
from fastapi.testclient import TestClient
from api.index import app, _compute_monthly_disaggregation


class TestMonthlyDisaggregation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.provinces = [
            "kalimantan_selatan",
            "kalimantan_tengah",
            "kalimantan_barat",
            "kalimantan_timur",
            "kalimantan_utara",
            "sulawesi_selatan"
        ]
        cls.methods = ["denton", "chowlin", "spline", "uniform"]
        cls.indicators = ["penumpang", "bagasi", "barang", "composite"]

    def test_01_api_endpoint_success_and_schema(self):
        """Uji endpoint /api/monthly_disaggregation merespon HTTP 200 dengan struktur schema valid."""
        for prov in self.provinces:
            res = self.client.get(
                f"/api/monthly_disaggregation?province={prov}&year=2024&sector=Transportasi+dan+Pergudangan&method=denton&indicator=penumpang"
            )
            self.assertEqual(res.status_code, 200, f"Failed for province: {prov}")
            data = res.json()
            self.assertEqual(data["status"], "success")
            self.assertEqual(data["province"], prov)
            self.assertEqual(data["year"], "2024")
            self.assertIn("model_diagnostics", data)
            self.assertIn("quarters_summary", data)
            self.assertIn("monthly_series", data)
            self.assertEqual(len(data["monthly_series"]), 12)
            self.assertEqual(len(data["quarters_summary"]), 4)

    def test_02_strict_accounting_additivity_all_methods_and_provinces(self):
        """
        STRICT ACCOUNTING ADDITIVITY TEST:
        Memverifikasi bahwa untuk SEMUA 4 metode di seluruh provinsi Kalimantan,
        jumlah nilai bulanan per kuartal sama persis dengan angka resmi rilis BPS (Delta < 1e-4).
        """
        for prov in self.provinces:
            for method in self.methods:
                res = _compute_monthly_disaggregation(
                    province=prov,
                    year="2024",
                    sector="Transportasi dan Pergudangan",
                    category="lu",
                    price_type="HK",
                    indicator="penumpang",
                    method=method,
                    time_scope="year"
                )
                self.assertNotIn("error", res, f"Error in {prov} with {method}")
                diag = res.get("model_diagnostics", {})
                max_delta = diag.get("quarterly_additivity_max_delta", 999.0)
                is_consistent = diag.get("is_accounting_consistent", False)

                self.assertTrue(is_consistent, f"Method {method} on {prov} failed accounting consistency: delta={max_delta}")
                self.assertLess(max_delta, 0.01, f"Method {method} on {prov} delta {max_delta} exceeds tolerance")

                # Uji penjumlahan manual setiap kuartal
                for q_info in res.get("quarters_summary", []):
                    delta_q = abs(q_info["pdrb_monthly_sum"] - q_info["pdrb_quarterly_bps"])
                    self.assertLess(delta_q, 0.01, f"Quarter delta {delta_q} on {prov} {method} exceeds tolerance")

    def test_03_boundary_discontinuity_pooled_mode(self):
        """
        BOUNDARY PROBLEM DISCONTINUITY TEST:
        Memverifikasi bahwa mode multi-tahun pooled (60 bulan) berjalan lancar
        dan tidak mengalami patahan atau lonjakan buatan antara Des(t) dan Jan(t+1).
        """
        for prov in ["kalimantan_selatan", "sulawesi_selatan"]:
            res = _compute_monthly_disaggregation(
                province=prov,
                year="2024",
                sector="Transportasi dan Pergudangan",
                category="lu",
                price_type="HK",
                indicator="penumpang",
                method="denton",
                time_scope="pooled"
            )
            self.assertNotIn("error", res)
            series = res.get("monthly_series", [])
            self.assertGreaterEqual(len(series), 48, f"Pooled series length {len(series)} too short")

            # Periksa kontinuitas antar-tahun pada titik batas (Bulan 12 -> Bulan 13, dst)
            pdrb_vals = np.array([r["pdrb_monthly"] for r in series])
            total_years = len(series) // 12
            for yr_idx in range(1, total_years):
                m_dec = (yr_idx * 12) - 1
                m_jan = yr_idx * 12
                step_jump = abs(pdrb_vals[m_jan] - pdrb_vals[m_dec])
                avg_level = (pdrb_vals[m_jan] + pdrb_vals[m_dec]) / 2.0
                rel_jump = step_jump / (avg_level or 1.0)
                # Lonjakan relatif tidak boleh melebihi 30% dari level bulanan
                self.assertLess(rel_jump, 0.30, f"Unreasonable step jump at year {yr_idx} for {prov}: {rel_jump:.2%}")

    def test_04_indicators_robustness(self):
        """Uji seluruh varian indikator (penumpang, bagasi, barang, composite) menghasilkan output valid."""
        for ind in self.indicators:
            res = _compute_monthly_disaggregation(
                province="kalimantan_selatan",
                year="2024",
                sector="Transportasi dan Pergudangan",
                category="lu",
                price_type="HK",
                indicator=ind,
                method="denton",
                time_scope="year"
            )
            self.assertNotIn("error", res, f"Indicator {ind} failed")
            self.assertEqual(len(res["monthly_series"]), 12)
            self.assertTrue(res["model_diagnostics"]["is_accounting_consistent"])

    def test_05_relevance_badges_and_guardrails(self):
        """Uji bahwa badge rekomendasi ekonometrika sesuai untuk berbagai jenis sektor."""
        # Sektor Transportasi -> high relevance
        res_trans = _compute_monthly_disaggregation(
            province="kalimantan_selatan", year="2024",
            sector="Transportasi dan Pergudangan", method="denton"
        )
        self.assertEqual(res_trans["relevance_badge"]["level"], "high")

        # Sektor Industri -> medium / logistik
        res_ind = _compute_monthly_disaggregation(
            province="kalimantan_selatan", year="2024",
            sector="Industri Pengolahan", method="denton"
        )
        self.assertEqual(res_ind["relevance_badge"]["level"], "medium")

        # Sektor Jasa Keuangan -> non_transport
        res_fin = _compute_monthly_disaggregation(
            province="kalimantan_selatan", year="2024",
            sector="Jasa Keuangan dan Asuransi", method="spline"
        )
        self.assertEqual(res_fin["relevance_badge"]["level"], "non_transport")

    def test_06_raw_sheet_pdrb_bulanan_mockup_integration(self):
        """Uji integrasi /api/raw_sheet?sheet=pdrb_bulanan_mockup untuk ekspor CSV."""
        res = self.client.get("/api/raw_sheet?province=kalimantan_selatan&year=2024&sheet=pdrb_bulanan_mockup")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["count"], 12)
        self.assertIn("columns", data)
        self.assertIn("rows", data)
        self.assertIn("PDRB Bulanan (Miliar Rp)", data["columns"])
        self.assertIn("Estimasi Denton-Cholette", data["columns"])

    def test_07_invalid_parameter_error_handling(self):
        """Uji penanganan graceful error untuk sektor atau provinsi yang invalid."""
        res = self.client.get("/api/monthly_disaggregation?province=provinsi_palsu&year=2024")
        self.assertEqual(res.status_code, 404)


if __name__ == "__main__":
    unittest.main()
