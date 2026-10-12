# IMAGE_INJECT

Inject a single camera image into the agent's visual context. The image becomes visible to the VLM on the next decision step.

## When to Use

- When you need to see what a specific camera sees to make a visual judgment.
- When the surround survey found something and you need a closer look.
- When you need to visually confirm fine details (text, small objects, spatial relationships).

Do not use this when a text summary suffices — use SURROUND_SURVEY instead.

## Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `camera` | yes | Which camera: `frontleft`, `frontright`, `left`, `right`, `back` |

## Behavior Tree

The framework auto-wraps with a `CancellationGuard`. The body is a single leaf:

```
ImageInjectSkill body
└── image_inject
      update():
        ├── Validates camera name
        ├── Grabs JPEG bytes from context.camera_images[camera]
        ├── Writes observation + injected_images to handle
        └── Returns SUCCESS or FAILURE
```

## Subtree Leaves (failure tags)

| Leaf | What FAILURE means |
|------|---------------------|
| `image_inject` | Invalid camera name, or no image available from the named camera. |
| `CancellationGuard` | Skill cancelled by the BT |

## Outcomes

| Outcome | Observation |
|---------|-------------|
| **Success** | `"Injected frontleft camera image into context."` |
| **Invalid camera** | `"Invalid camera 'top'. Valid cameras: back, frontleft, ..."` |
| **No image** | `"No image available from back camera."` |

## Result Data

```
{
    "observation": "Injected frontleft camera image into context.",
    "injected_images": {"frontleft": <JPEG bytes>}
}
```

The actor includes `injected_images` as vision input in the next turn. The image is available for exactly one decision and then discarded.

## Gating

Available when `LAM_IMAGE_INJECT=on` (the default).

## Token Cost

~1500 tokens per injected image. Use sparingly — prefer SURROUND_SURVEY for broad overview (~500 tokens for all 5 cameras).

## Typical Sequence

```
SURROUND_SURVEY("is the button visible?") → "Target found in left camera."
  → IMAGE_INJECT("left")    # see the button in detail
    → CAPTURE("left", "the red button on the wall")
```
