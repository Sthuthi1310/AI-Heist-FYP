"""Integration tests for Flask API endpoints."""
import unittest
import json
import config
from web.app import app
from utils.preprocessing import get_sample_employee

class TestAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        config.SECURITY_CONFIG["rate_limiting_enabled"] = False
        config.SECURITY_CONFIG["suspicious_detection_enabled"] = False
        config.SECURITY_CONFIG["perturbation_enabled"] = False
        config.SECURITY_CONFIG["auth_required"] = False

    def test_sample_employee(self):
        res = self.client.get("/api/sample-employee?risk=high")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("Age", data)
        self.assertIn("Department", data)

    def test_predict_endpoint_valid_payload(self):
        sample = get_sample_employee("high")
        res = self.client.post("/api/predict", json=sample)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "success")
        self.assertIn("prediction", data)
        self.assertIn("probability", data)
        self.assertIn("attrition", data)
        self.assertIn("security", data)

    def test_security_config_update(self):
        res = self.client.post("/api/security/config", json={
            "rate_limit_per_minute": 75,
            "suspicion_threshold": 0.45
        })
        self.assertEqual(res.status_code, 200)
        self.assertEqual(config.SECURITY_CONFIG["rate_limit_per_minute"], 75)
        self.assertEqual(config.SECURITY_CONFIG["suspicion_threshold"], 0.45)

    def test_metrics_endpoint(self):
        res = self.client.get("/api/metrics")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("status", data)

    def test_predict_endpoint_rejects_missing_features(self):
        res = self.client.post("/api/predict", json={"Age": 34})
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertEqual(data.get("status"), "error")
        self.assertIn("Missing required features", data.get("message", ""))

    def test_predict_endpoint_rejects_invalid_category(self):
        sample = get_sample_employee("high")
        sample["Gender"] = "Unknown"
        res = self.client.post("/api/predict", json=sample)
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertEqual(data.get("status"), "error")

    def test_high_risk_profile_predicts_attrition(self):
        res = self.client.post("/api/predict", json=get_sample_employee("high"))
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json().get("attrition"), "Yes")

    def test_low_risk_profile_predicts_retention(self):
        res = self.client.post("/api/predict", json=get_sample_employee("low"))
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json().get("attrition"), "No")

if __name__ == "__main__":
    unittest.main()
