"""Convention for the observability instrumentation that can be stripped.

Every line tagged ``# OBS-DROP:`` and every block wrapped in
``# ---- OBS-DROP-BEGIN ----`` / ``# ---- OBS-DROP-END ----`` is purely
for the live visualizer. Removing them does not affect agent behavior;
the existing ``BB.PENDING_EVENTS`` mechanism remains untouched.

Strip recipe::

    # 1. Remove all OBS-DROP-BEGIN…END blocks in instrumented files.
    sed -i '/# ---- OBS-DROP-BEGIN ----/,/# ---- OBS-DROP-END ----/d' \\
        lam_v2_pycore/src/lam_v2/agent/lam_agent/{planner,actor,evaluator}.py \\
        lam_v2_pycore/src/lam_v2/bt/nodes/{plan_task,select_skill,evaluate_completion}.py

    # 2. Find any remaining single-line markers for manual review.
    grep -rn 'OBS-DROP' lam_v2_pycore/src/lam_v2/

    # 3. Delete the observability package + harness viz wiring.
    rm -rf lam_v2_pycore/src/lam_v2/observability/
    rm -rf lam_v2_testing/src/lam_v2_testing/viz/
    # Then remove the EVENT_SINK key from agent/blackboard_keys.py
    # and the event_sink param from LAMAgent.__init__.
"""

OBS_DROP_TAG = "OBS-DROP"
OBS_BEGIN = "# ---- OBS-DROP-BEGIN ----"
OBS_END = "# ---- OBS-DROP-END ----"
