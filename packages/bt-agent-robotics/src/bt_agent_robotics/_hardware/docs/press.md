# PRESS

Step on or press down on a floor-level object. The robot navigates to
the object via the nav graph, then drives precisely onto it.

## When to Use

- After QUERY returned a handle for a floor panel, button, or platform.
- To physically interact with floor objects (e.g., step on colored panels
  in a specific order to unlock a safe).
- The robot will drive ONTO the target, not just near it.
- Requires NAVIGATE + SURROUND_SURVEY to have been called first.

## Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `target_object` | yes | Handle from QUERY (e.g. `blue_panel_1`). |

## Behavior Tree

The framework wraps the body inside a safeguarding Sequence
(see [Safeguarding] below). The body itself:

```
PressSkill body (Sequence, memory=True)
├── validate_target    — resolve handle, extract object position
├── plan_path          — Dijkstra → waypoint list to nearest node
├── follow_path        — waypoint sequencing to get near target
└── drive_onto_target  — precise approach directly onto the object
```

## Subtree Leaves (failure tags)

When a skill fails, `handle.observation` is prefixed `[leaf=<name>]`
identifying which child returned FAILURE.

| Leaf | What FAILURE means | Recovery hint |
|------|---------------------|---------------|
| `validate_target` | Handle not in ObjectStore or position missing/invalid. | Re-QUERY the target. |
| `plan_path` | Dijkstra found no path from current pose to target. | Try from a different location. |
| `follow_path` | Local planner blocked at a waypoint; replan exhausted. | Wait, or try alternative approach. |
| `drive_onto_target` | Could not reach object within 0.2m after 30s. | Object may be obstructed or unreachable from this angle. |

Wrap-level failure tags (framework, all skills):

| Leaf | What it means |
|------|----------------|
| `CancellationGuard`   | Skill cancelled by the BT. |
| `WallTimeBudgetGuard` | Skill exceeded its wall-time budget. |
| `InvariantGuard`      | `safeguard_invariant` returned False. See [Safeguarding]. |

## Safeguarding

Framework Wrap:

```
Sequence(memory=False, [
    CancellationGuard,
    WallTimeBudgetGuard,
    InvariantGuard,
    <body subtree above>,
])
```

`safeguard_invariant` returns False (aborting within one tick) when:

- `state.robot_position is None` — localization lost.
- `handle.result_data["planned_path"]` exists and is empty - path invalidated.

## Gating

- `precondition`: ObjectStore must have at least one object.
- `is_available`: Requires a prior `navigate` call AND a `surround_survey`
  call after it in the current milestone history.

## Outcomes

| Status | press_status | Observation |
|--------|-------------|-------------|
| Pressed | `pressed` | "Stepped on 'blue_panel_1'." |
| Pressed (blocked) | `pressed` | "Stepped on 'blue_panel_1'. (blocked)" |
| No path | `failed_no_path` | "No path to target." |
| Blocked en route | `failed_blocked` | "Blocked en route, replan exhausted." |
| Timeout (no move) | `failed_no_movement` | "Robot never started moving." |
| Timeout (moved) | `pressed` | "Stepped on 'blue_panel_1'. (timeout)" |

## Result Data

```json
{
    "press_status": "pressed",
    "target_handle": "blue_panel_1",
    "target_label": "blue floor panel",
    "goal": {"x": 1.5, "y": 2.5},
    "final_distance": 0.12
}
```

## Typical Sequence

```
QUERY("blue floor panel") → handle "blue_panel_1"
  → NAVIGATE("blue_panel_1")           ← get near panel
    → SURROUND_SURVEY("is the panel visible?")
      → PRESS("blue_panel_1")          ← step onto it
```
