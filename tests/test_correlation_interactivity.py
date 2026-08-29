"""
========================================================================================
Unit Test Suite: Dynamic Single-Year vs Multi-Year Pooled Correlation Interactivity
========================================================================================
"""

import unittest
from fastapi.testclient import TestClient
from api.index import app, _compute_correlations


class TestCorrelationInteractivity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.provinces = [
            "sulawesi_selatan",
            "kalimantan_selatan",
            "kalimantan_barat",
            "kalimantan_timur"
        ]

    def test_01_single_year_vs_pooled_response_structure(self):
        """Uji respons endpoint /api/correlations menyertakan time_scope dan sample_label."""
        res_single = self.client.get(
            "/api/correlations?province=sulawesi_selatan&year=2024&analysis_mode=growth_yoy&time_scope=single_year"
        )
        self.assertEqual(res_single.status_code, 200)
        data_s = res_single.json()
        self.assertEqual(data_s["time_scope"], "single_year")
        self.assertEqual(data_s["n_observations"], 4)
        self.assertIn("Tahun 2024", data_s["sample_label"])

        res_pooled = self.client.get(
            "/api/correlations?province=sulawesi_selatan&year=2024&analysis_mode=growth_yoy&time_scope=pooled"
        )
        self.assertEqual(res_pooled.status_code, 200)
        data_p = res_pooled.json()
        self.assertEqual(data_p["time_scope"], "pooled")
        self.assertGreater(data_p["n_observations"], 4)
        self.assertIn("Multi-Tahun Pooled", data_p["sample_label"])

    def test_02_year_change_interactivity_different_correlations(self):
        """
        Uji interaktivitas perubahan dropdown tahun:
        Korelasi tahun 2023 vs 2024 pada single_year mode harus mencerminkan data dinamis kuartal tahun bersangkutan.
        """
        res_2023 = self.client.get(
            "/api/correlations?province=sulawesi_selatan&year=2023&analysis_mode=growth_yoy&time_scope=single_year"
        ).json()
        res_2024 = self.client.get(
            "/api/correlations?province=sulawesi_selatan&year=2024&analysis_mode=growth_yoy&time_scope=single_year"
        ).json()

        def get_trans_corr(data):
            for r in data.get("rows", []):
                if "Transportasi" in r.get("Lapangan Usaha", ""):
                    return r.get("Korelasi dgn Penumpang")
            return None

        corr_2023 = get_trans_corr(res_2023)
        corr_2024 = get_trans_corr(res_2024)

        self.assertIsNotNone(corr_2023)
        self.assertIsNotNone(corr_2024)
        # Nilai korelasi tahun 2023 dan 2024 tidak boleh identik/terkunci
        self.assertNotAlmostEqual(corr_2023, corr_2024, places=3, msg="Tahun 2023 dan 2024 menghasilkan korelasi identik!")

    def test_03_kalimantan_provinces_support_single_year_and_pooled(self):
        """Uji bahwa seluruh provinsi Kalimantan mendukung single_year dan pooled correlation."""
        for prov in ["kalimantan_selatan", "kalimantan_barat", "kalimantan_timur"]:
            # Single year
            res_s = self.client.get(
                f"/api/correlations?province={prov}&year=2024&analysis_mode=growth_yoy&time_scope=single_year"
            )
            self.assertEqual(res_s.status_code, 200, f"Failed for {prov} single_year")
            data_s = res_s.json()
            self.assertGreater(data_s["count"], 0)

            # Pooled
            res_p = self.client.get(
                f"/api/correlations?province={prov}&year=2024&analysis_mode=growth_yoy&time_scope=pooled"
            )
            self.assertEqual(res_p.status_code, 200, f"Failed for {prov} pooled")
            data_p = res_p.json()
            self.assertGreater(data_p["count"], 0)

    def test_04_level_mode_single_year_and_pooled(self):
        """Uji mode level/absolut (abs_hk & abs_hb) pada single_year dan pooled."""
        res_abs_s = self.client.get(
            "/api/correlations?province=sulawesi_selatan&year=2024&analysis_mode=abs_hk&time_scope=single_year"
        ).json()
        self.assertEqual(res_abs_s["n_observations"], 4)

        res_abs_p = self.client.get(
            "/api/correlations?province=sulawesi_selatan&year=2024&analysis_mode=abs_hk&time_scope=pooled"
        ).json()
        self.assertGreaterEqual(res_abs_p["n_observations"], 12)


if __name__ == "__main__":
    unittest.main()
