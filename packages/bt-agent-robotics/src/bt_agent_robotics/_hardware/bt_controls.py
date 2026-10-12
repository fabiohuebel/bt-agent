"""Opt-in, execution-bound BT interventions. Not a certified safety controller.

The BT guard stops only its current execution. The agent continues normally;
future navigation keeps the existing GO gate. There is no extra release step,
no task cancellation, and no non-BT fallback counted as BT stopping evidence.
"""

from __future__ import annotations

import contextlib
import copy
import math
import threading
import time
import uuid

import py_trees


class MotionHeld(RuntimeError):
    pass


class Controller:
    def __init__(self, backend, *, emit=lambda event: None, clock=time.monotonic):
        self.backend, self.emit_sink = backend, emit
        self.clock = clock
        self.lock = threading.RLock()
        self.active = None
        self.pending = None
        self.last_intervention = None
        self.seen = {}
        self.stationary_since = None
        self.stationary_recorded = False

    def emit(self, kind, **data):
        event = {"kind": kind, "wall_ns": time.time_ns(), "monotonic_s": self.clock(), **data}
        # Diagnostic failure must not disable the command hold.
        with contextlib.suppress(Exception):
            self.emit_sink(event)

    def begin(self, skill, context, handle):
        with self.lock:
            if self.active is not None:
                raise RuntimeError("An instrumented motion skill is already active")
            key = uuid.uuid4().hex
            self.active = {
                "key": key,
                "skill": skill,
                "call_id": context.skill_call_id,
                "task": context.main_task,
                "handle": handle,
                "revoked": False,
            }
            self.emit("motion_execution_begin", key=key, skill=skill, call_id=context.skill_call_id)
            return key

    def finish(self, key, status):
        with self.lock:
            if self.active is not None and self.active["key"] == key:
                if self.pending is not None:
                    self.emit("intervention_expired_unconsumed", request_id=self.pending["request_id"])
                    self.pending = None
                self.emit("motion_execution_end", key=key, status=str(status))
                self.active = None

    def snapshot(self):
        with self.lock:
            active = None
            if self.active:
                active = {k: v for k, v in self.active.items() if k != "handle"}
                tree = self.active["handle"].subtree
                tip = tree.tip() if tree is not None else None
                active["leaf"] = tip.name if tip is not None else None
            return {
                "active": active,
                "last_intervention": self.last_intervention,
                "pending": self.pending is not None,
                "experimental": True,
                "agent_continues": True,
                "next_motion_requires_normal_go": True,
            }

    def request(self, key, kind, request_id):
        if kind not in {"interrupt", "timeout", "failure"}:
            raise ValueError("Unknown intervention kind")
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
            raise ValueError("Invalid request ID")
        with self.lock:
            if request_id in self.seen:
                if self.seen[request_id] != (key, kind):
                    raise ValueError("Request ID reused with different contents")
                return False
            if not self.active or self.active["key"] != key or self.active["handle"].is_terminal:
                raise ValueError("No matching active motion execution")
            if self.pending is not None or self.active["revoked"]:
                raise ValueError("An intervention is already active")
            self.seen[request_id] = (key, kind)
            if len(self.seen) > 128:
                self.seen.pop(next(iter(self.seen)))
            self.pending = {
                "key": key,
                "intervention": kind,
                "request_id": request_id,
                "requested_s": self.clock(),
                "call_id": self.active["call_id"],
            }
            self.emit("intervention_requested", **self.pending)
            return True

    def _stop(self, request, initiator):
        if self.active["revoked"]:
            return
        self.active["revoked"] = True
        self.last_intervention = {**request, "initiator": initiator}
        self.stationary_since, self.stationary_recorded = None, False
        self.emit("execution_revoked", initiator=initiator, **request)
        # Latch first, brake second, then zero/park. Never release the brake.
        operations = [
            ("brake", lambda: self.backend.set_brake(True)),
            ("zero", lambda: self.backend.send_velocity(0.0, 0.0, 0.0)),
        ]
        for name, operation in operations:
            try:
                operation()
                self.emit("hold_command_sent", command=name, request_id=request["request_id"])
            except Exception as error:
                self.emit("hold_command_error", command=name, error=str(error), request_id=request["request_id"])
        try:
            age = self.backend.get_pose_age()
            pose = self.backend.get_robot_pose()
            if age is not None and 0 <= age <= 0.5 and all(math.isfinite(v) for v in pose):
                self.backend.cancel()
                self.emit("hold_command_sent", command="park", request_id=request["request_id"])
        except Exception as error:
            self.emit("hold_command_error", command="park", error=str(error), request_id=request["request_id"])

    def consume(self, key):
        with self.lock:
            if not self.pending or not self.active or self.active["key"] != key:
                return None
            request = self.pending
            self._stop(request, "bt_guard")
            self.pending = None
            self.emit("intervention_consumed_by_bt", stop_initiator="bt_guard", **request)
            return request

    def observe_motion(self, linear, angular):
        with self.lock:
            if self.last_intervention is None or self.stationary_recorded:
                return
            if not all(math.isfinite(v) and abs(v) < 0.03 for v in (linear, angular)):
                self.stationary_since = None
                return
            now = self.clock()
            if self.stationary_since is None:
                self.stationary_since = now
            elif now - self.stationary_since >= 0.2:
                self.stationary_recorded = True
                self.emit(
                    "stationary_confirmed",
                    request_id=self.last_intervention["request_id"],
                    first_stationary_s=self.stationary_since,
                    confirmation_s=now,
                )

    def call(self, key, method, *args):
        with self.lock:
            if self.active is None or self.active["key"] != key:
                raise MotionHeld("Stale execution command suppressed")
            safe = (
                method == "cancel"
                or (method == "set_brake" and args == (True,))
                or (method == "send_velocity" and all(v == 0 for v in args))
                or (method == "set_press_precision" and args == (False,))
            )
            if method == "send_waypoint" and self.active["revoked"] and len(args) == 3:
                # Permit terminal discipline's park-at-self only, not another
                # target. Stale scopes are already rejected above.
                age = self.backend.get_pose_age()
                pose = self.backend.get_robot_pose()
                safe = (
                    age is not None
                    and 0 <= age <= 0.5
                    and all(math.isfinite(v) for v in (*args, *pose))
                    and math.dist(args[:2], pose[:2]) < 0.005
                    and abs(math.atan2(math.sin(args[2] - pose[2]), math.cos(args[2] - pose[2]))) < 0.01
                )
            if self.active["revoked"] and not safe:
                raise MotionHeld("Motion held: command suppressed")
            return getattr(self.backend, method)(*args)


class ScopedPlanner:
    def __init__(self, controller, key):
        self.controller, self.key = controller, key

    def __getattr__(self, name):
        return getattr(self.controller.backend, name)

    def send_waypoint(self, *args):
        return self.controller.call(self.key, "send_waypoint", *args)

    def send_velocity(self, *args):
        return self.controller.call(self.key, "send_velocity", *args)

    def set_brake(self, engaged):
        return self.controller.call(self.key, "set_brake", engaged)

    def set_press_precision(self, enabled):
        if not hasattr(self.controller.backend, "set_press_precision"):
            return True
        return self.controller.call(self.key, "set_press_precision", enabled)

    def cancel(self):
        return self.controller.call(self.key, "cancel")


class InterventionGuard(py_trees.behaviour.Behaviour):
    def __init__(self, controller, key, handle):
        super().__init__("OperatorInterventionGuard")
        self.controller, self.key, self.handle = controller, key, handle
        self.consumed = False

    def update(self):
        if self.consumed:
            return py_trees.common.Status.FAILURE
        request = self.controller.consume(self.key)
        if request is None:
            return py_trees.common.Status.SUCCESS
        self.consumed = True
        kind = request["intervention"]
        if kind == "interrupt":
            reason = "operator_interrupt"
            observation = (
                "[leaf=OperatorInterventionGuard] operator_interrupt: current skill stopped. "
                "The task remains active; choose the next action."
            )
        elif kind == "timeout":
            reason = "wall_time_exceeded"
            observation = "[leaf=WallTimeBudgetGuard] The skill timed out before completion."
        else:
            reason = "execution_failed"
            observation = "[leaf=SkillExecution] The skill failed before completion."
        # Result data can reach public skill events/history: no experiment-only
        # identifiers or injection labels belong there. The private controller
        # journal already links the intervention to this skill call.
        result = dict(self.handle.result_data)
        for name in ("intervention_request_id", "injected"):
            result.pop(name, None)
        result["failure_reason"] = reason
        if kind != "interrupt":
            # The standard worker preserves an existing [leaf=...] prefix.
            # Present the simulated fault location, not the test interception
            # mechanism; the real tree node is preserved in the journal below.
            result["failed_leaf"] = "WallTimeBudgetGuard" if kind == "timeout" else "SkillExecution"
        self.handle.result_data = result
        self.handle.observation = observation
        self.controller.emit(
            "intervention_feedback",
            request_id=request["request_id"],
            call_id=request["call_id"],
            intervention=kind,
            injected=kind != "interrupt",
            actual_leaf=self.name,
            public_failure_reason=reason,
        )
        # Reactive Sequence invalidates the running body. Do not cancel/reset
        # the agent or set the task's cancellation flag.
        return py_trees.common.Status.FAILURE


class ExecutionScope(py_trees.decorators.Decorator):
    def __init__(self, child, controller, key):
        super().__init__(name="InterventionExecutionScope", child=child)
        self.controller, self.key = controller, key

    def update(self):
        return self.decorated.status

    def terminate(self, new_status):
        try:
            if new_status == py_trees.common.Status.INVALID:
                self.decorated.stop(py_trees.common.Status.INVALID)
        finally:
            self.controller.finish(self.key, new_status)

    def tick(self):
        try:
            yield from super().tick()
        except Exception:
            try:
                self.decorated.stop(py_trees.common.Status.INVALID)
            finally:
                self.controller.finish(self.key, "exception")
            raise


def instrument_skill(skill, controller):
    """Wrap only build_subtree; metadata, prepare, safeguards and actor stay intact."""
    if skill.name not in {"navigate", "press"}:
        raise ValueError("Only navigate and press are qualified intervention targets")

    def build(params, context, handle):
        key = controller.begin(skill.name, context, handle)
        try:
            scoped = copy.copy(skill)
            scoped._local_planner = ScopedPlanner(controller, key)
            body = type(skill).build_subtree(scoped, params, context, handle)
            root = py_trees.composites.Sequence(
                "InterruptibleMotion",
                memory=False,
                children=[
                    InterventionGuard(controller, key, handle),
                    body,
                ],
            )
            return ExecutionScope(root, controller, key)
        except Exception:
            controller.finish(key, "build_error")
            raise

    skill.build_subtree = build
    return skill
