"""Basic API health test."""
import unittest
from web.app import app

class TestHealthEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("status"), "ok")
        self.assertIn("service", data)

if __name__ == "__main__":
    unittest.main()
