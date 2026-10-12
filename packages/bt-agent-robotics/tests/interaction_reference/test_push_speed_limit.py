import importlib.util
import math
import unittest
from pathlib import Path

path = Path(__import__("bt_agent_robotics").__file__).parent / "_push/speed_limit.py"
spec = importlib.util.spec_from_file_location("push_speed_candidate", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
TOKEN = "1234567890abcdef"


class SpeedLimitTests(unittest.TestCase):
    def setUp(self):
        self.limit = module.PushSpeedLimit()

    def request(self, now=10, **kwargs):
        data = dict(token=TOKEN, active=True, max_speed=0.15)
        data.update(kwargs)
        return self.limit.observe(data, now, braked=True)

    def test_inactive_preserves_commands(self):
        self.assertEqual(self.limit.limit((1, 2, 3), 10, braked=False), (1, 2, 3))

    def test_acquisition_requires_brake(self):
        self.assertFalse(self.limit.observe(dict(token=TOKEN, active=True, max_speed=0.15), 10, braked=False))

    def test_enforces_vector_speed_without_changing_direction(self):
        self.request()
        v = self.limit.limit((3, 4, 2), 10.1, braked=False)
        self.assertAlmostEqual(math.hypot(*v[:2]), 0.15)
        self.assertAlmostEqual(v[0] / v[1], 3 / 4)
        self.assertEqual(v[2], 0.3)

    def test_existing_brake_always_wins(self):
        self.request()
        self.assertEqual(self.limit.limit((1, 0, 0), 10.1, braked=True), (0, 0, 0))

    def test_expiry_sticky(self):
        self.request()
        self.assertEqual(self.limit.limit((1, 0, 0), 11, braked=False), (0, 0, 0))
        self.assertFalse(self.request(11))

    def test_release_waits_for_brake(self):
        self.request()
        self.limit.observe(dict(token=TOKEN, active=False, max_speed=0), 10.1, braked=False)
        self.assertEqual(self.limit.limit((1, 0, 0), 10.2, braked=False), (0, 0, 0))
        self.assertIsNotNone(self.limit.token)
        self.limit.advance(10.3, braked=True)
        self.assertIsNone(self.limit.token)

    def test_wrong_token_cannot_release(self):
        self.request()
        self.assertFalse(
            self.limit.observe(dict(token="other-token-12345", active=False, max_speed=0), 10.1, braked=True)
        )
        self.assertEqual(self.limit.token, TOKEN)

    def test_limit_cannot_be_increased_by_renewal(self):
        self.request()
        self.assertFalse(self.request(10.1, max_speed=0.2))
        self.assertEqual(self.limit.max_speed, 0.15)

    def test_stalled_heartbeat_cannot_renew_before_expiry_check(self):
        self.request()
        self.assertFalse(self.request(11))
        self.assertEqual(self.limit.snapshot()["status"], "speed_lease_expired")

    def test_reverse_clock_fails(self):
        self.request()
        self.assertEqual(self.limit.limit((1, 0, 0), 9, braked=False), (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
