"""Unit tests for API defense mechanisms (Rate Limiting, Anomaly Detection, Perturbation)."""
import unittest
import time
import config
from security.rate_limiter import SlidingWindowRateLimiter
from security.suspicious_detection import SuspiciousQueryDetector
from security.perturbation import apply_output_perturbation
from utils.preprocessing import get_sample_employee

class TestSecurityDefenses(unittest.TestCase):
    def setUp(self):
        config.SECURITY_CONFIG["rate_limiting_enabled"] = True
        config.SECURITY_CONFIG["rate_limit_per_minute"] = 5
        config.SECURITY_CONFIG["burst_limit_per_second"] = 3
        config.SECURITY_CONFIG["suspicious_detection_enabled"] = True
        config.SECURITY_CONFIG["suspicion_threshold"] = 0.40
        config.SECURITY_CONFIG["perturbation_enabled"] = True
        config.SECURITY_CONFIG["perturbation_mode"] = "adaptive"

    def test_rate_limiter_burst_and_minute_limits(self):
        rl = SlidingWindowRateLimiter()
        client = "test_client_rl"
        
        # 1-3 requests allowed
        for _ in range(3):
            allowed, info = rl.check_rate_limit(client)
            self.assertTrue(allowed)
            
        # 4th burst request within same second must be rejected
        allowed, info = rl.check_rate_limit(client)
        self.assertFalse(allowed)
        self.assertEqual(info.get("reason"), "burst_limit_exceeded")

    def test_suspicious_query_detector_burst_cadence(self):
        detector = SuspiciousQueryDetector()
        client = "test_client_bot"
        sample = get_sample_employee("high")
        
        # Send very rapid queries (< 10ms interval) to simulate automated extraction script
        scores = []
        for _ in range(12):
            res = detector.analyze_query(client, sample, predicted_probability=0.49)
            scores.append(res["suspicion_score"])
            time.sleep(0.005)
            
        # Later queries in the burst should trigger elevated suspicion scores
        self.assertGreater(scores[-1], 0.30)
        self.assertGreater(detector.get_stats()["active_clients"], 0)

    def test_adaptive_perturbation(self):
        # 1. Genuine request (is_suspicious=False) must NOT be perturbed
        label, prob, applied, reason = apply_output_perturbation(
            original_label=1,
            original_prob=0.85,
            is_suspicious=False
        )
        self.assertEqual(label, 1)
        self.assertEqual(prob, 0.85)
        self.assertFalse(applied)
        self.assertEqual(reason, "genuine_query_unaltered")

        # 2. Suspicious request (is_suspicious=True) must have defense applied
        config.SECURITY_CONFIG["flip_probability"] = 1.0  # Force deterministic flip for test
        label, prob, applied, reason = apply_output_perturbation(
            original_label=1,
            original_prob=0.85,
            is_suspicious=True
        )
        self.assertEqual(label, 0)  # Flipped!
        self.assertTrue(applied)
        self.assertEqual(reason, "adaptive_perturbation_on_suspicious_query")

if __name__ == "__main__":
    unittest.main()
