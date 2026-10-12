"""CPU-only tests for the disabled controller telemetry adapter."""

import importlib.util
import unittest
from pathlib import Path

path = Path(__import__("bt_agent_robotics").__file__).parent / "_push/feedback.py"
spec = importlib.util.spec_from_file_location("push_feedback_candidate", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def event(sample, **changes):
    value = dict(
        schema="native-push-observation/v1",
        run_id="r",
        task_id="t",
        task_index=1,
        mechanism_id="push:dining_chair",
        status="push_observation",
        evidence=dict(canonical_key="dining_chair", native_instance_id=99, sample=sample),
    )
    value.update(changes)
    return value


class PushFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.buffer = module.PushFeedbackBuffer(capacity=2)
        self.buffer.bind(("r", "t", 1), "dining_chair", 99)

    def test_keeps_every_sample(self):
        self.buffer.observe(event(51), 10)
        self.buffer.observe(event(52), 10.1)
        self.assertEqual([e["evidence"]["sample"] for e, t in self.buffer.drain(10.2)], [51, 52])

    def test_gap_sticky_until_new_binding(self):
        self.buffer.observe(event(1), 10)
        self.assertFalse(self.buffer.observe(event(3), 10.1))
        with self.assertRaisesRegex(RuntimeError, "sequence_gap"):
            self.buffer.drain(10.2)
        self.buffer.bind(("r", "t", 1), "dining_chair", 99)
        self.assertTrue(self.buffer.observe(event(9), 11))

    def test_overflow_not_silent_drop(self):
        for i in range(3):
            self.buffer.observe(event(i + 1), 10 + i * 0.1)
        with self.assertRaisesRegex(RuntimeError, "overflow"):
            self.buffer.drain(10.3)

    def test_stale_fails_and_remains_failed(self):
        self.buffer.observe(event(1), 10)
        with self.assertRaisesRegex(RuntimeError, "stale"):
            self.buffer.drain(12)
        self.assertFalse(self.buffer.observe(event(2), 12))

    def test_old_episode_ignored(self):
        self.assertFalse(self.buffer.observe(event(1, run_id="old"), 10))
        self.assertTrue(self.buffer.observe(event(5), 10))

    def test_wrong_instance_stops(self):
        e = event(1)
        e["evidence"]["native_instance_id"] = 100
        self.assertFalse(self.buffer.observe(e, 10))
        with self.assertRaisesRegex(RuntimeError, "identity"):
            self.buffer.drain(10)

    def test_receive_clock_not_sim_time(self):
        e = event(1)
        e["sim_time"] = 900000
        self.buffer.observe(e, 10)
        self.assertEqual(len(self.buffer.drain(10.5)), 1)

    def test_no_alias_to_mutable_ros_payload(self):
        e = event(1)
        self.buffer.observe(e, 10)
        e["evidence"]["sample"] = 42
        self.assertEqual(self.buffer.drain(10)[0][0]["evidence"]["sample"], 1)

    def test_reset_discards_bound_state(self):
        self.buffer.observe(event(1), 10)
        self.buffer.reset()
        with self.assertRaisesRegex(RuntimeError, "not_bound"):
            self.buffer.drain(10)

    def test_duplicate_sample_fails(self):
        self.buffer.observe(event(1), 10)
        self.buffer.observe(event(1), 10.1)
        self.assertEqual(self.buffer.failure, "feedback_sequence_gap")


if __name__ == "__main__":
    unittest.main()
