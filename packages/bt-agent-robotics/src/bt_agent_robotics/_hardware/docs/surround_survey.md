# SURROUND_SURVEY

Check whether ONE specific target is visible across the robot's five camera
views. Each camera is judged independently by a per-frame VLM call. The skill
cannot reason across views or about multiple objects at once.

## When to Use

- After NAVIGATE to **confirm arrival** or verify the target is visible.
- To **survey the environment** before deciding what to do next.
- When you need a broad overview but don't need full image detail.

Use IMAGE_INJECT instead when you need fine detail, spatial reasoning
(`"where is X relative to Y?"`), or open-ended description (`"what's in the
room?"`). For **multiple targets**, issue ONE SURROUND_SURVEY per target. Do not combine them into one call.

## Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `target_description` | yes | ONE atomic visibility question. <br>**GOOD:** `"is the red umbrella visible?"`, `"is the safe door open?"`. <br>**BAD (compound):** `"is X AND Y visible?"` → make two calls. <br>**BAD (spatial):** `"where is X relative to Y?"` → use IMAGE_INJECT. <br>**BAD (vague):** `"safe door"` → say what condition you want to verify. |

## Behavior Tree

```
SurroundSurveySkill body (Sequence, memory=True)
├── analyze_frontleft     ← VLM call: is target in this view?
├── analyze_frontright    ← VLM call
├── analyze_left          ← VLM call
├── analyze_right         ← VLM call
├── analyze_back          ← VLM call
└── compile_results       ← picks best view, builds summary
```

## Subtree Leaves (failure tags)

When a skill fails, `handle.observation` is prefixed `[leaf=<name>]`.

| Leaf | What FAILURE means |
|------|---------------------|
| `analyze_frontleft` / `analyze_frontright` / `analyze_left` / `analyze_right` / `analyze_back` | VLM call failed (network/model error) or returned an unparseable response. One bad camera fails the whole survey. |
| `compile_results` | Aggregation failed — currently only on a malformed shared-state dict. |
| `CancellationGuard` | Skill cancelled by the BT. |

Each `AnalyzeView` node:
1. Checks cancellation (aborts if requested)
2. Sends camera image + question to VLM
3. Gets structured response: `{reasoning, evaluation, visible: bool, bounding_box}`
4. Appends result

`CompileResults` node:
- **0 detections**: calls VLM to summarize what WAS seen across all cameras
- **1 detection**: returns that camera's analysis directly
- **2+ detections**: calls VLM to compare views and pick the best one

## Outcomes

| Outcome | Observation | Result Data |
|---------|-------------|-------------|
| **Found (1 camera)** | `"Target found in left camera. The red button is visible on the wall panel."` | `{verdict: true, camera: "left"}` |
| **Found (2+ cameras)** | `"Target found in frontleft camera. ..."` (best view picked) | `{verdict: true, camera: "frontleft"}` |
| **Not found** | `"The red button was not visible. Front cameras show a hallway, ..."` | `{verdict: false, camera: "none"}` |

## Result Data

```
{
    "verdict": true,           # was the target detected?
    "camera": "left",          # best camera (or "none")
    "analyses": [              # per-camera results
        {"camera": "frontleft", "reasoning": "...", "evaluation": "...", "visible": false},
        {"camera": "frontright", ...},
        {"camera": "left", "reasoning": "...", "evaluation": "...", "visible": true,
         "bounding_box": [120, 350, 480, 720]},
        ...
    ]
}
```

## Gating

None. Available any time camera images exist.

## Token Cost

~500 tokens total (5 short text summaries). Compare to IMAGE_INJECT: ~1500 tokens per image, ~7500 for all 5 cameras.

## Cancellation

Checked between each camera. If cancelled after 3 cameras, the remaining 2 are skipped. The thread exits cleanly.

## Typical Sequence

```
NAVIGATE("obj_hallway_1") → arrived
  → SURROUND_SURVEY("is the red umbrella visible?")
    → verdict: true, camera: "frontleft"
      → CAPTURE("frontleft", "the red umbrella") or task_complete
    → verdict: false
      → QUERY("red umbrella") again, or NAVIGATE elsewhere
```
