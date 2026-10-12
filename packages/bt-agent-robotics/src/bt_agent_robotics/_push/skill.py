"""Optional measured-rail PushSkill using existing navigation and mux brakes.

Native poses stay inside the controller. Only a typed completion/failure result
is published to the agent. A completed waypoint alone never proves pushing.
"""

import logging
import math
import os
import time
import uuid

import py_trees
from bt_agent.skills import ActionSkill
from bt_agent_robotics._push.navigation import (
    _AlignHeading,
    _FollowPath,
    _TerminalDiscipline,
)
from bt_agent_robotics.skills.press import _PlanPath


class PushSkill(ActionSkill):
    name = "push"
    description = "Move a supported object along its physical push affordance."
    usage = (
        "Query the target first. The skill includes navigation to its push approach; completion requires measured"
        " object motion."
    )
    params = {"target_object": "Exact target handle returned by query."}

    def __init__(self, local_planner, path_planner):
        self.local_planner = local_planner
        self.path_planner = path_planner

    def precondition(self, state):
        return (state.object_store is not None, "ObjectStore required.")

    def is_available(self, state, context):
        return True, "push available after query"

    def safeguard_invariant(self, state, handle):
        age = self.local_planner.get_pose_age()
        if age is None or not 0 <= age <= 1:
            return "stale_pose"
        return True

    def prepare(self, params, context):
        target = params.get("target_object", "")
        obj = context.object_store.get(target) if context.object_store is not None else None
        if obj is None:
            return {}, "Query the target first."
        if getattr(self.local_planner, "push_telemetry", None) is None:
            return {}, "Physical push profile is unavailable."
        return dict(target_handle=obj.handle, target_label=obj.label), None

    def build_subtree(self, params, context, handle):
        return _TerminalDiscipline(
            child=_PushExecution(params, context, handle, self.local_planner, self.path_planner),
            local_planner=self.local_planner,
        )


class _PushMotion:
    def __init__(self, planner, *, contact_control=False):
        self.planner = planner
        self.contact_control = contact_control
        self.token = uuid.uuid4().hex
        self.mode = "new"
        self.goal = None
        self.acquiring = None
        self.limit = 0.15

    def approach(self, pose):
        self.mode = "approach"
        self.goal = pose

    def waypoint(self, pose, *, max_speed):
        if not 0 < max_speed <= 0.15:
            raise RuntimeError("Unqualified push speed")
        self.goal = pose
        self.limit = max_speed
        if self.mode != "contact":
            self.mode = "contact"
            self.acquiring = time.monotonic()
            self.planner.cancel()
            self.planner.set_brake(True)

    def service(self, now):
        if self.mode != "contact":
            return
        self.planner.request_push_speed(self.token, self.limit)
        if not self.planner.push_speed_acknowledged(self.token, self.limit):
            self.planner.set_brake(True)
            if self.acquiring is None or now - self.acquiring > 2:
                raise RuntimeError("Push speed lease was not acknowledged")
            return
        self.acquiring = None
        if self.contact_control:
            age = self.planner.get_pose_age()
            if age is None or not 0 <= age <= 1:
                raise RuntimeError("Stale pose in contact control")
            x, y, yaw = self.planner.get_robot_pose()
            gx, gy, gyaw = self.goal
            if not all(math.isfinite(v) for v in (x, y, yaw, gx, gy, gyaw)):
                raise RuntimeError("Invalid contact pose")
            dx, dy = gx - x, gy - y
            length = math.hypot(dx, dy)
            scale = min(1.0, self.limit / length) if length else 0.0
            vx, vy = dx * scale, dy * scale
            body_x = math.cos(yaw) * vx + math.sin(yaw) * vy
            body_y = -math.sin(yaw) * vx + math.cos(yaw) * vy
            error = math.atan2(math.sin(gyaw - yaw), math.cos(gyaw - yaw))
            yaw_rate = max(-0.3, min(0.3, error))
            self.planner.set_brake(False)
            # Review-gated contact segment: use the existing mux primitive
            # channel and native collision-constrained motion. Local-planner
            # obstacle avoidance is NOT used during this contact segment.
            self.planner.send_velocity(body_x, body_y, yaw_rate)
            return
        self.planner.set_brake(False)
        # The unchanged Habitat path follower stops0.15m before its goal.
        # Lead the nominal contact point by0.25m so the robot can enter the
        # controller's0.08m waypoint window. Native displacement still decides
        # completion; neither the authored object target nor planner changes.
        x, y, yaw = self.goal
        self.planner.send_waypoint(x + 0.25 * math.cos(yaw), y + 0.25 * math.sin(yaw), yaw)

    def stop(self):
        self.planner.cancel()
        self.planner.set_brake(True)
        self.planner.release_push_speed(self.token)
        self.mode = "stopped"


class _PushExecution(py_trees.behaviour.Behaviour):
    def __init__(self, params, context, handle, local, path):
        super().__init__("measured_push")
        self.params = params
        self.context = context
        self.handle = handle
        self.local = local
        self.path = path
        self.motion = None
        self.controller = None
        self.nav = None
        self.align = None
        self.approach_params = None
        self.last_observation = None
        self.started = None
        self.completed_at = None

    def initialise(self):
        self.started = time.monotonic()
        self.handle.wall_time_budget_seconds = 100
        mode = os.environ.get("LAM_HABITAT_CONTACT_CONTROL", "0")
        if mode not in {"0", "1"}:
            raise RuntimeError("Unknown contact-control profile")
        if mode == "1" and os.environ.get("LAM_HABITAT_MEASURED_PUSH") != "1":
            raise RuntimeError("Contact control requires the measured-push profile")
        self.motion = _PushMotion(self.local, contact_control=mode == "1")

    def finish(self, status):
        if self.motion:
            self.motion.stop()
        self.local.push_telemetry.end()
        self.handle.result_data = dict(target_handle=self.params["target_handle"], push_status=status)
        self.handle.observation = f"Push '{self.params['target_label']}': {status}."
        return py_trees.common.Status.SUCCESS if status == "completed_verified" else py_trees.common.Status.FAILURE

    def update(self):
        from bt_agent_robotics._push.controller import Affordance, PushController

        now = time.monotonic()
        if self.handle.is_cancel_requested:
            return self.finish("cancelled")
        if now - self.started > 98:
            return self.finish("timeout")
        try:
            if self.controller is None:
                binding = self.local.push_telemetry.begin(self.params["target_handle"])
                asset = Affordance(**binding["asset"])
                resume_mode = os.environ.get("LAM_HABITAT_PUSH_EPISODE_RESUME", "0")
                if resume_mode not in {"0", "1"}:
                    raise RuntimeError("Unknown push-resume profile")
                options = {"resume_enabled": True} if resume_mode == "1" else {}
                self.controller = PushController(
                    asset, binding["episode_token"], binding["native_instance_id"], self.motion, **options
                )
                params = dict(
                    self.params, goal_x=asset.approach[0], goal_y=asset.approach[1], goal_theta=asset.approach[2]
                )
                params["heading_tolerance_rad"] = self.controller.approach_heading_tolerance
                self.approach_params = params
                self.nav = py_trees.composites.Sequence(
                    "push_approach",
                    memory=True,
                    children=[
                        _PlanPath(params, self.handle, self.path, self.local, self.context),
                        _FollowPath(params, self.handle, self.path, self.local),
                    ],
                )
            observations = self.local.push_telemetry.drain(now)
            if observations:
                self.last_observation = observations[-1]
            if self.last_observation is None:
                if now - self.started > 1:
                    return self.finish("feedback_missing")
                return py_trees.common.Status.RUNNING
            pose = self.local.get_robot_pose()
            age = self.local.get_pose_age()
            if age is None:
                return self.finish("stale_pose")
            result = "running"
            for observation in observations or [self.last_observation]:
                result = self.controller.tick(now, pose, age, observation)
                if result != "running":
                    break
            if result == "completed_verified":
                self.motion.stop()
                if self.local.push_telemetry.completed_snapshot_received(self.controller.asset.key):
                    return self.finish(result)
                self.completed_at = self.completed_at or now
                if now - self.completed_at > 2:
                    return self.finish("completion_feedback_missing")
                return py_trees.common.Status.RUNNING
            if result != "running":
                return self.finish(result)
            if self.motion.mode == "approach":
                # A source-verified retry starts behind the current object pose.
                # The existing path-planning/following nodes share this params dict.
                target = self.controller.asset.approach
                self.approach_params.update(goal_x=target[0], goal_y=target[1], goal_theta=target[2])
                if self.nav.status != py_trees.common.Status.SUCCESS:
                    self.nav.tick_once()
                self.handle.wall_time_budget_seconds = 100
                if self.nav.status == py_trees.common.Status.FAILURE:
                    return self.finish("approach_failed")
                if self.nav.status == py_trees.common.Status.SUCCESS:
                    target = self.controller.asset.approach
                    if not self.controller.approach_position_ready(pose):
                        # Reuse the existing0.25m goal lead for precise entry;
                        # normal navigation still chooses and checks the path.
                        dx, dy = target[0] - pose[0], target[1] - pose[1]
                        distance = math.hypot(dx, dy)
                        if distance <= 1e-9:
                            return self.finish("invalid_approach_precision")
                        self.local.send_waypoint(
                            target[0] + 0.25 * dx / distance, target[1] + 0.25 * dy / distance, target[2]
                        )
                    else:
                        if self.align is None:
                            self.local.cancel()
                            self.align = _AlignHeading(self.approach_params, self.handle, self.local)
                        if self.align.status != py_trees.common.Status.SUCCESS:
                            self.align.tick_once()
                        if self.align.status == py_trees.common.Status.FAILURE:
                            return self.finish("approach_heading_failed")
                        alignment = self.handle.result_data.get("heading_alignment", {})
                        if (
                            self.align.status == py_trees.common.Status.SUCCESS
                            and alignment.get("within_tolerance") is False
                        ):
                            return self.finish("approach_heading_not_reached")
            else:
                if self.nav.status == py_trees.common.Status.RUNNING:
                    self.nav.stop(py_trees.common.Status.INVALID)
                if self.align is not None and self.align.status == py_trees.common.Status.RUNNING:
                    self.align.stop(py_trees.common.Status.INVALID)
            self.motion.service(now)
            return py_trees.common.Status.RUNNING
        except Exception:
            logging.getLogger(__name__).exception("Measured push adapter failed")
            # Detailed native telemetry remains in the independent private journal.
            return self.finish("push_feedback_or_adapter_error")

    def terminate(self, new_status):
        if self.align is not None and self.align.status == py_trees.common.Status.RUNNING:
            self.align.stop(py_trees.common.Status.INVALID)
        if self.nav is not None and self.nav.status == py_trees.common.Status.RUNNING:
            self.nav.stop(py_trees.common.Status.INVALID)
        if self.motion is not None:
            self.motion.stop()
        if getattr(self.local, "push_telemetry", None) is not None:
            self.local.push_telemetry.end()
