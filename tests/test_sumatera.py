"""
========================================================================================
Unit Tests for 10 Sumatera Provinces Integration (2020–2024)
Menguji integritas API untuk:
  - aceh, sumatera_utara, sumatera_barat, riau, kep_riau
  - jambi, sumatera_selatan, bengkulu, lampung, kep_bangka_belitung
========================================================================================
"""

import unittest
from fastapi.testclient import TestClient
from api.index import app


class TestSumateraIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.sumatera_provinces = [
            "aceh",
            "sumatera_utara",
            "sumatera_barat",
            "riau",
            "kep_riau",
            "jambi",
            "sumatera_selatan",
            "bengkulu",
            "lampung",
            "kep_bangka_belitung"
        ]

    def test_manifest_includes_all_10_sumatera_provinces(self):
        """Memverifikasi bahwa ke-10 provinsi Sumatera terdaftar di manifest beserta 5 tahunnya."""
        res = self.client.get("/api/manifest")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("provinces", data)
        self.assertGreaterEqual(len(data["provinces"]), 17)

        for p in self.sumatera_provinces:
            self.assertIn(p, data["provinces"], f"{p} missing from manifest")
            prov_data = data["provinces"][p]
            self.assertIn("years", prov_data)
            self.assertEqual(len(prov_data["years"]), 5, f"{p} does not have 5 years")
            self.assertIn("2020", prov_data["years"])
            self.assertIn("2024", prov_data["years"])

    def test_sumatera_options(self):
        """Memverifikasi opsi sektor PDRB untuk seluruh 10 provinsi Sumatera."""
        for p in self.sumatera_provinces:
            res = self.client.get(f"/api/options?province={p}&year=2024")
            self.assertEqual(res.status_code, 200, f"Options failed for {p}")
            data = res.json()
            self.assertIn("sectors", data)
            self.assertGreaterEqual(len(data["sectors"]), 17)

    def test_sumatera_correlations_growth_and_absolute(self):
        """Memverifikasi kalkulasi korelasi untuk seluruh mode analisis pada 10 provinsi Sumatera."""
        for p in self.sumatera_provinces:
            for mode in ["growth_yoy", "growth_qoq", "abs_hk", "abs_hb"]:
                for cat in ["lu", "peng"]:
                    res = self.client.get(
                        f"/api/correlations?province={p}&year=2024&type={cat}&analysis_mode={mode}"
                    )
                    self.assertEqual(res.status_code, 200, f"Correlations failed for {p} {mode} {cat}")
                    data = res.json()
                    self.assertIn("rows", data)
                    self.assertGreater(data["count"], 0)

    def test_sumatera_dashboard_data_and_regression(self):
        """Memverifikasi time-series, KPI, dan model regresi OLS pada 10 provinsi Sumatera."""
        for p in self.sumatera_provinces:
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

    def test_sumatera_multicollinearity(self):
        """Memverifikasi kalkulasi VIF, korelasi silang, dan multiko transportasi pada 10 provinsi Sumatera."""
        for p in self.sumatera_provinces:
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
