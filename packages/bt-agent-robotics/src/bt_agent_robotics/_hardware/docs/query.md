# QUERY

Search the scene graph for an object or location. Returns a semantic handle that downstream skills use to access coordinates.

## When to Use

- **First step** when you need to find something. Always QUERY before NAVIGATE.
- When you need a handle for a target object (e.g., `obj_red_mug_1`).
- When the task mentions an object or location you haven't queried yet.

## Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `target_description` | yes | What to search for. Be specific: "red umbrella with black handle", not just "umbrella". |

## Behavior Tree

The framework wraps with a `CancellationGuard` only.

```
QuerySkill body (single-node BT)
└── query
      update():
        ├── Validates target_description
        ├── Calls SceneGraphAgent.query(target_description, ...)
        ├── Builds result: found objects / exploration / not found
        ├── Writes observation + result_data to handle
        └── Returns SUCCESS or FAILURE
```

## Subtree Leaves (failure tags)

| Leaf | What FAILURE means |
|------|---------------------|
| `query` | Empty `target_description`, or scene-graph-agent call raised. |
| `CancellationGuard` | Skill cancelled by the BT. |

## Outcomes

| Outcome | Observation | Objects Produced |
|---------|-------------|-----------------|
| **Found** | `"Found 'red umbrella'. Use 'red_umbrella_1' for navigation."` | `SceneObject(handle="red_umbrella_1", position=..., viewpoints=...)` |
| **Exploration** | `"'red umbrella' not found. Explore near 'kitchen' using 'explore_1'."` | `SceneObject(handle="explore_1", is_exploration=True)` |
| **Not found** | `"'red umbrella' not found in scene graph."` | None |

## Result Data

```
{
    "observation": "Found 'red umbrella'. Use 'red_umbrella_1' for navigation.",
    "new_objects": [SceneObject(...)],       # added to ObjectStore
    "query_mappings": {"red umbrella": "red_umbrella_1"}  # registered for fuzzy lookup
}
```

## Gating

None. Always available.

## Typical Sequence

```
QUERY("red umbrella with black handle") → found handle "red_umbrella_1"
  → NAVIGATE("red_umbrella_1")
    → SURROUND_SURVEY("is the red umbrella visible?")
      → CAPTURE / task_complete
```
