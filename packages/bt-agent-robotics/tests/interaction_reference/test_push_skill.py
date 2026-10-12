"""Safety/dispatch integration with the real skill class and a fake ROS planner."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from bt_agent_robotics._push.skill import PushSkill, _PushMotion


class Planner:
    def __init__(self):
        self.calls = []
        self.ack = False
        self.push_telemetry = object()

    def cancel(self):
        self.calls.append(("cancel",))

    def set_brake(self, value):
        self.calls.append(("brake", value))

    def request_push_speed(self, token, speed):
        self.calls.append(("limit", token, speed))

    def push_speed_acknowledged(self, token, speed):
        return self.ack

    def release_push_speed(self, token):
        self.calls.append(("release", token))

    def send_waypoint(self, *pose):
        self.calls.append(("waypoint", *pose))


class PushSkillTests(unittest.TestCase):
    def test_completed_approach_is_not_restarted_during_precise_arrival(self):
        import sys

        import py_trees
        from bt_agent_robotics._push import skill as push

        class CountingSuccess(py_trees.behaviours.Success):
            def __init__(self, name):
                super().__init__(name)
                self.calls = 0

            def update(self):
                self.calls += 1
                return super().update()

        class Controller:
            def __init__(self, asset, episode, native_id, motion):
                self.asset, self.motion = asset, motion
                self.approach_radius = 0.2
                self.approach_heading_tolerance = 0.02

            def approach_position_ready(self, pose):
                return False

            def tick(self, *args):
                if self.motion.mode == "new":
                    self.motion.approach(self.asset.approach)
                return "running"

        planner = Planner()
        planner.get_robot_pose = lambda: (0, 0, 0)
        planner.get_pose_age = lambda: 0
        planner.push_telemetry = SimpleNamespace(
            begin=lambda handle: dict(asset=dict(approach=(1, 2, 0)), episode_token="episode", native_instance_id=99),
            drain=lambda now: [object()],
            end=lambda: None,
        )
        handle = SimpleNamespace(is_cancel_requested=False, result_data={})
        node = push._PushExecution(dict(target_handle="chair", target_label="chair"), None, handle, planner, None)
        plan, follow = CountingSuccess("plan"), CountingSuccess("follow")
        controller_module = SimpleNamespace(
            Affordance=lambda **values: SimpleNamespace(**values), PushController=Controller
        )
        with (
            patch.dict(sys.modules, {"bt_agent_robotics._push.controller": controller_module}),
            patch.object(push, "_PlanPath", return_value=plan),
            patch.object(push, "_FollowPath", return_value=follow),
        ):
            node.initialise()
            for _ in range(3):
                self.assertEqual(node.update(), py_trees.common.Status.RUNNING)
        self.assertEqual((plan.calls, follow.calls), (1, 1))
        self.assertEqual(
            [call for call in planner.calls if call[0] == "waypoint"],
            [("waypoint", 1 + 0.25 / (5**0.5), 2 + 0.5 / (5**0.5), 0)] * 3,
        )




    def test_missing_query_handle_is_rejected(self):
        skill = PushSkill(Planner(), None)
        ctx = SimpleNamespace(object_store={})
        self.assertIsNotNone(skill.prepare(dict(target_object="guessed"), ctx)[1])

    def test_prepared_params_do_not_expose_native_telemetry(self):
        obj = SimpleNamespace(handle="dining chair 1", label="dining chair")
        skill = PushSkill(Planner(), None)
        prepared, error = skill.prepare(dict(target_object=obj.handle), SimpleNamespace(object_store={obj.handle: obj}))
        self.assertIsNone(error)
        self.assertEqual(set(prepared), {"target_handle", "target_label"})

    def test_contact_cannot_move_until_mux_ack(self):
        planner = Planner()
        motion = _PushMotion(planner)
        motion.waypoint((1, 2, 0), max_speed=0.15)
        motion.service(motion.acquiring)
        self.assertFalse(any(c[0] == "waypoint" for c in planner.calls))
        self.assertNotIn(("brake", False), planner.calls)
        planner.ack = True
        motion.service(motion.acquiring)
        self.assertIn(("waypoint", 1.25, 2, 0), planner.calls)
        self.assertIn(("brake", False), planner.calls)

    def test_missing_speed_ack_is_bounded(self):
        planner = Planner()
        motion = _PushMotion(planner)
        motion.waypoint((1, 2, 0), max_speed=0.15)
        with self.assertRaises(RuntimeError):
            motion.service(motion.acquiring + 3)
        self.assertIn(("brake", True), planner.calls)

    def test_terminal_stop_parks_brakes_and_releases_own_lease(self):
        planner = Planner()
        motion = _PushMotion(planner)
        motion.stop()
        self.assertEqual(planner.calls, [("cancel",), ("brake", True), ("release", motion.token)])

    def test_no_ignored_higher_speed_request(self):
        motion = _PushMotion(Planner())
        with self.assertRaises(RuntimeError):
            motion.waypoint((1, 2, 0), max_speed=0.4)


if __name__ == "__main__":
    unittest.main()
