import os
import sys
import unittest
from fastapi.testclient import TestClient

# Pastikan path modul terdaftar
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.index import app, compute_multicollinearity

client = TestClient(app)

class TestMulticollinearity(unittest.TestCase):

    def test_multicollinearity_api_post_core_6_sectors(self):
        """Uji endpoint POST /api/multicollinearity dengan 6 sektor inti (termasuk sektor ber-koma & titik-koma)."""
        core_6_sectors = [
            "Industri Pengolahan",
            "Konstruksi",
            "Perdagangan Besar dan Eceran; Reparasi Mobil dan Sepeda Motor",
            "Transportasi dan Pergudangan",
            "Penyediaan Akomodasi dan Makan Minum",
            "Pengadaan Air, Pengelolaan Sampah, Limbah, dan Daur Ulang"
        ]
        payload = {
            "province": "sulawesi_selatan",
            "price_type": "HK",
            "analysis_mode": "growth_yoy",
            "category": "lu",
            "sectors": core_6_sectors
        }
        response = client.post("/api/multicollinearity", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data.get("status"), "success")
        self.assertIn("condition_number", data)
        self.assertIn("has_multicollinearity", data)
        self.assertIn("vif_results", data)
        self.assertIn("inter_sector_matrix", data)
        self.assertIn("recommendation", data)

        # Pastikan seluruh 6 sektor lengkap di vif_results dan inter_sector_matrix
        vif_list = data["vif_results"]
        self.assertEqual(len(vif_list), 6, f"Expected 6 sectors in VIF results, got {len(vif_list)}")
        
        matrix_obj = data["inter_sector_matrix"]
        self.assertEqual(len(matrix_obj["labels"]), 6)
        self.assertEqual(len(matrix_obj["matrix"]), 6)

        sec_names = [v["sector"] for v in vif_list]
        self.assertTrue(any("Pengadaan Air" in s for s in sec_names), "Pengadaan Air harus ada di vif_results")
        self.assertTrue(any("Perdagangan" in s for s in sec_names), "Perdagangan harus ada di vif_results")

        for item in vif_list:
            self.assertIn("sector", item)
            self.assertIn("vif", item)
            self.assertIn("tolerance", item)
            self.assertIn("status", item)
            self.assertIn("badge_color", item)
            self.assertGreaterEqual(item["vif"], 1.0)
            self.assertGreater(item["tolerance"], 0.0)

    def test_multicollinearity_api_post_default(self):
        """Uji endpoint POST /api/multicollinearity dengan payload default / kosong."""
        payload = {
            "province": "sulawesi_selatan",
            "price_type": "HK",
            "analysis_mode": "growth_yoy",
            "category": "lu"
        }
        response = client.post("/api/multicollinearity", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "success")
        self.assertEqual(len(data["vif_results"]), 6)

    def test_multicollinearity_api_post_single_sector_edge_case(self):
        """Uji endpoint POST dengan hanya 1 sektor -> harus ditolak (HTTP 400)."""
        payload = {
            "province": "sulawesi_selatan",
            "sectors": ["Konstruksi"]
        }
        response = client.post("/api/multicollinearity", json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data.get("status"), "error")
        self.assertIn("Minimal 2 sektor", data.get("message", ""))

    def test_multicollinearity_api_post_pengeluaran(self):
        """Uji endpoint POST untuk kategori Pengeluaran."""
        payload = {
            "province": "sulawesi_selatan",
            "category": "peng",
            "price_type": "HK",
            "analysis_mode": "growth_yoy",
            "sectors": [
                "Pengeluaran Konsumsi Rumah Tangga",
                "Pembentukan Modal Tetap Bruto",
                "Ekspor Barang dan Jasa",
                "Impor Barang dan Jasa"
            ]
        }
        response = client.post("/api/multicollinearity", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "success")
        self.assertEqual(len(data["vif_results"]), 4)

    def test_multicollinearity_api_get_backward_compat(self):
        """Uji backward compatibility endpoint GET /api/multicollinearity."""
        response = client.get("/api/multicollinearity?province=sulawesi_selatan&price_type=HK&analysis_mode=growth_yoy")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "success")
        self.assertEqual(len(data["vif_results"]), 6)

    def test_multicollinearity_modes_and_prices(self):
        """Uji mode QoQ, Absolut, dan Harga Berlaku (HB) via POST."""
        # Mode QoQ
        res_qoq = client.post("/api/multicollinearity", json={
            "province": "sulawesi_selatan",
            "price_type": "HK",
            "analysis_mode": "growth_qoq"
        })
        self.assertEqual(res_qoq.status_code, 200)
        self.assertEqual(res_qoq.json()["status"], "success")

        # Mode HB
        res_hb = client.post("/api/multicollinearity", json={
            "province": "sulawesi_selatan",
            "price_type": "HB",
            "analysis_mode": "growth_yoy_hb"
        })
        self.assertEqual(res_hb.status_code, 200)
        self.assertEqual(res_hb.json()["status"], "success")

        # Mode Absolut
        res_abs = client.post("/api/multicollinearity", json={
            "province": "sulawesi_selatan",
            "price_type": "HK",
            "analysis_mode": "abs_hk"
        })
        self.assertEqual(res_abs.status_code, 200)
        self.assertEqual(res_abs.json()["status"], "success")

    def test_multicollinearity_transport_and_cross_matrix(self):
        """Uji output struktur korelasi silang (PDRB × Transportasi) dan VIF indikator Transportasi."""
        payload = {
            "province": "sulawesi_selatan",
            "price_type": "HK",
            "analysis_mode": "growth_yoy",
            "category": "lu",
            "sectors": [
                "Industri Pengolahan",
                "Transportasi dan Pergudangan",
                "Konstruksi"
            ]
        }
        response = client.post("/api/multicollinearity", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "success")

        # 1. Validasi cross_transport_matrix
        self.assertIn("cross_transport_matrix", data)
        cross_obj = data["cross_transport_matrix"]
        self.assertEqual(len(cross_obj["sector_labels"]), 3)
        self.assertEqual(cross_obj["transport_labels"], ["Penumpang", "Bagasi", "Barang"])
        self.assertEqual(len(cross_obj["matrix"]), 3)
        self.assertEqual(len(cross_obj["p_values"]), 3)
        for row in cross_obj["matrix"]:
            self.assertEqual(len(row), 3)
        self.assertEqual(len(cross_obj["strongest_targets"]), 3)

        # 2. Validasi transport_multicollinearity
        self.assertIn("transport_multicollinearity", data)
        t_multi = data["transport_multicollinearity"]
        self.assertIn("condition_number", t_multi)
        self.assertIn("has_multicollinearity", t_multi)
        self.assertIn("vif_results", t_multi)
        self.assertEqual(len(t_multi["vif_results"]), 3)
        indicators = [item["indicator"] for item in t_multi["vif_results"]]
        self.assertIn("Penumpang", indicators)
        self.assertIn("Bagasi", indicators)
        self.assertIn("Barang", indicators)
        self.assertIn("matrix", t_multi)
        self.assertEqual(len(t_multi["matrix"]["matrix"]), 3)
        self.assertIn("recommendation", t_multi)

if __name__ == "__main__":
    unittest.main()
