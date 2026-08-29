"""
========================================================================================
Unit Tests for Kalimantan Provinces Integration (2020–2024)
Menguji integritas API untuk:
  - kalimantan_selatan
  - kalimantan_timur
  - kalimantan_barat
  - kalimantan_tengah
========================================================================================
"""

import unittest
from fastapi.testclient import TestClient
from api.index import app


class TestKalimantanIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.kal_provinces = [
            "kalimantan_selatan",
            "kalimantan_timur",
            "kalimantan_barat",
            "kalimantan_tengah",
            "kalimantan_utara"
        ]

    def test_manifest_includes_kalimantan_provinces(self):
        """Memverifikasi bahwa ke-4 provinsi Kalimantan dan 5 tahun (2020-2024) terdaftar."""
        res = self.client.get("/api/manifest")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("provinces", data)
        for p in self.kal_provinces:
            self.assertIn(p, data["provinces"])
            prov_data = data["provinces"][p]
            self.assertIn("years", prov_data)
            self.assertEqual(len(prov_data["years"]), 5)
            self.assertIn("2020", prov_data["years"])
            self.assertIn("2024", prov_data["years"])

    def test_kalimantan_options(self):
        """Memverifikasi opsi sektor untuk seluruh provinsi Kalimantan."""
        for p in self.kal_provinces:
            res = self.client.get(f"/api/options?province={p}&year=2024")
            self.assertEqual(res.status_code, 200, f"Options failed for {p}")
            data = res.json()
            self.assertIn("sectors", data)
            self.assertGreaterEqual(len(data["sectors"]), 17)

    def test_kalimantan_correlations_growth_and_absolute(self):
        """Memverifikasi kalkulasi korelasi untuk seluruh mode analisis."""
        for p in self.kal_provinces:
            for mode in ["growth_yoy", "growth_qoq", "abs_hk", "abs_hb"]:
                for cat in ["lu", "peng"]:
                    res = self.client.get(
                        f"/api/correlations?province={p}&year=2024&type={cat}&analysis_mode={mode}"
                    )
                    self.assertEqual(res.status_code, 200, f"Failed for {p} {mode} {cat}")
                    data = res.json()
                    self.assertIn("rows", data)
                    self.assertGreater(data["count"], 0)

    def test_kalimantan_dashboard_data_and_regression(self):
        """Memverifikasi time-series, KPI, dan model regresi OLS."""
        for p in self.kal_provinces:
            for metric in ["penumpang", "bagasi", "barang"]:
                res = self.client.get(
                    f"/api/data?province={p}&year=2024&transport_metric={metric}&analysis_mode=growth_yoy&ols_scope=pooled"
                )
                self.assertEqual(res.status_code, 200, f"Data failed for {p} {metric}")
                data = res.json()
                self.assertIn("kpi", data)
                self.assertIn("triwulan_data", data)
                self.assertIn("regression", data)
                self.assertIn("slope", data["regression"])
                self.assertIn("r_squared", data["regression"])

    def test_kalimantan_multicollinearity(self):
        """Memverifikasi kalkulasi VIF, korelasi silang, dan multiko transportasi."""
        for p in self.kal_provinces:
            res = self.client.post("/api/multicollinearity", json={
                "province": p,
                "analysis_mode": "growth_yoy",
                "category": "lu",
                "price_type": "HK"
            })
            self.assertEqual(res.status_code, 200, f"Multico failed for {p}")
            data = res.json()
            self.assertEqual(data["status"], "success")
            self.assertIn("condition_number", data)
            self.assertIn("vif_results", data)
            self.assertIn("cross_transport_matrix", data)
            self.assertIn("transport_multicollinearity", data)


if __name__ == "__main__":
    unittest.main()
