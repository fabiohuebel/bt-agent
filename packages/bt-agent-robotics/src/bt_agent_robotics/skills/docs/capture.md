# CAPTURE

Photograph a target object from a specified camera. Verifies visibility,
refines the bounding box, optionally runs segmentation, and produces a
`CaptureResult` in the ObjectStore for downstream submission.

## When to Use

- After NAVIGATE confirmed the robot is near the target.
- After SURROUND_SURVEY identified which camera has the best view.
- As the final action of a `capture` milestone (the system submits the
  artifact automatically when the milestone advances).

## Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `target_object` | yes | Handle from QUERY (e.g. `red_umbrella_1`). |
| `camera_view` | yes | Camera name: `frontleft`, `frontright`, `left`, `right`, or `back`. |

## Behavior Tree

The framework auto-wraps the body inside a safeguarding Sequence
(see [Safeguarding] below). The body itself:

```
CaptureSkill body (Sequence, memory=True)
├── validate_target       — single tick: input validation
├── analyze_view          — VLM: visible? tight bbox?
│                            → FAILURE if target not visible
├── refine_pointing       — VLM: 1-2 iterations to tighten bbox (best-effort)
├── segment_mask          — pointing pipeline if injected, else skip
└── finalize_capture      — build CaptureResult, set observation
```

## Subtree Leaves (failure tags)

When a skill fails, `handle.observation` is prefixed `[leaf=<name>]`
identifying which child returned FAILURE.

| Leaf | What FAILURE means | Recovery hint |
|------|---------------------|---------------|
| `validate_target`   | `image_bytes` missing — `prepare()` should have caught this; reaching here means a malformed handle slipped in. | Re-run SURROUND_SURVEY; verify the camera is producing frames. |
| `analyze_view`      | VLM call failed (network/model error) OR the VLM reports the target is not visible in the named camera. **Most common capture failure.** | Try a different camera (re-SURROUND_SURVEY) or NAVIGATE to a different `viewpoint_index`. |
| `refine_pointing`   | Always succeeds (best-effort). VLM iteration errors are caught locally. | — |
| `segment_mask`      | Always succeeds (best-effort). Pipeline errors degrade to bbox-only. | — |
| `finalize_capture`  | Always succeeds. Builds the `CaptureResult`. | — |

Wrap-level failure tags (framework, all skills):

| Leaf | What it means |
|------|----------------|
| `CancellationGuard`   | Skill cancelled by the BT (preemption) or by the agent/human. |
| `WallTimeBudgetGuard` | Skill exceeded its wall-time budget (default 120 s). |
| `InvariantGuard`      | `safeguard_invariant` returned False. See [Safeguarding]. |

## Safeguarding

Framework wrap:

```
Sequence(memory=False, [
    CancellationGuard,
    WallTimeBudgetGuard,
    InvariantGuard,
    <body subtree above>,
])
```

`safeguard_invariant` is true before `analyze_view` runs.
After `analyze_view` succeeds (handle marks the chosen camera), the
predicate returns False if:

- `state.camera_images is None` — camera feed dropped out, OR
- the chosen camera disappeared from `state.camera_images`.

This catches mid-capture camera-feed dropout that the body wouldn't
otherwise notice. Recovery: re-SURROUND_SURVEY then CAPTURE.

## Outcomes

| Status | Observation | Result data |
|--------|-------------|-------------|
| Success | `Captured 'red umbrella' from frontleft camera. Handle: 'cap_red_umbrella_3'.` | `CaptureResult` added to ObjectStore |
| Not visible | `Target 'red umbrella' not visible in frontleft camera.` | observation only |
| VLM failure | `VLM analysis failed: ...` | observation only |

## Result Data

```python
{
    "observation": "Captured 'red umbrella' from frontleft camera. Handle: 'cap_red_umbrella_1'.",
    "new_objects": [CaptureResult(
        handle="cap_red_umbrella_1",
        description="red umbrella",
        camera="frontleft",
        target_coords=(640, 480),
        bbox_1000=(312, 240, 720, 580),
        rotated_image_width=1024,
        rotated_image_height=768,
        mask_backend="",
        mask_similarity=1.0,
        image=<JPEG bytes>,
    )]
}
```

The `CaptureResult` is added to the ObjectStore.

## Gating

- `requires_query = True` — QUERY must succeed first.
- Available only on milestones with `capture: True`.

## Typical Sequences

```
QUERY("red umbrella") → handle "red_umbrella_1"
  → NAVIGATE("red_umbrella_1") → arrived
    → SURROUND_SURVEY("red umbrella visible?") → "best in frontleft"
      → CAPTURE(target="red_mug_1", camera="frontleft") ← this skill
```

## Failure Recovery

- **Target not visible**: Try a different camera (re-run SURROUND_SURVEY) or
  re-NAVIGATE to a different viewpoint via `viewpoint_index`.
- **VLM rejected refinement**: The bbox stays at the last accepted value.
  Capture proceeds with whatever bbox was identified.
- **Segmentation failed**: Capture still proceeds with bbox-only data.
  The pipeline is best-effort.
