"""System-prompt variants for the three LAM v2 roles (planner, actor, evaluator).

This module is the single source of truth for the prompt strings. Each
variant lives under ``system_prompts/<variant>/`` and exposes a single
``PROMPT_BUNDLE: PromptBundle`` constant (via its ``__init__.py``), which is
discoverable via ``get_prompts(<variant>)``.

The release ships one variant, ``tiamat_agent_submit_v5``, which provides
every field. The development variants it was assembled from are not part
of the release.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class PromptBundle:
    """The prompt strings that fully specify a role-prompt variant.

    Each string is a Python ``str.format`` template. The interpolation
    placeholders are fixed across variants so the call sites in
    ``planner.py`` / ``actor.py`` / ``evaluator.py`` stay variant-agnostic.

    Placeholders (must be supported by every variant):
      - ``planner_gather``:    ``{task}``, ``{scenario}``
      - ``planner_synthesis``: ``{task}``, ``{scenario}``, ``{gathered_info}``,
                               ``{milestone_fields}``
      - ``planner_replan``:    ``{completed_block}``, ``{failed_description}``,
                               ``{failed_milestone_id}``, ``{reason}``,
                               ``{base_user_prompt}``
      - ``actor``:             ``{milestone_idx}``, ``{total_milestones}``,
                               ``{milestone_description}``,
                               ``{milestone_instructions}``,
                               ``{milestone_verify}``, ``{object_summaries}``,
                               ``{history_summary}``. ``select_skill()`` also
                               always passes ``{budget_disclosure}`` (the
                               symmetric action-budget disclosure, empty
                               unless configured) -- unlike the list above,
                               this one is OPTIONAL: most variants don't
                               declare the placeholder, and ``str.format()``
                               ignores unused keyword arguments, so they are
                               unaffected.
      - ``evaluator``:         no placeholders (system prompt only)
      - ``actor_compaction``:  ``{actions_text}``
      - ``evaluator_user``:    ``{scenario}``, ``{skill_descriptions}``,
                               ``{description}``, ``{interaction_type}``,
                               ``{verify_condition}``,
                               ``{claim_section}``, ``{actor_trace}``,
                               ``{observation}``, ``{store_summaries}``
      - ``evaluator_claim``:   ``{claim_reasoning}``
    """

    name: str
    planner_gather: str
    planner_synthesis: str
    planner_replan: str
    actor: str
    evaluator: str
    # Sub-prompts — empty = "not provided by this variant".
    actor_compaction: str = ""
    evaluator_user: str = ""
    evaluator_claim: str = ""


_PACKAGE_ROOT = Path(__file__).parent


def list_variants() -> List[str]:
    """Return all available variant names (lexicographic order)."""
    return sorted(p.name for p in _PACKAGE_ROOT.iterdir() if p.is_dir() and not p.name.startswith("_"))


def _load_bundle(variant: str) -> PromptBundle:
    """Load a variant's PROMPT_BUNDLE without sub-prompt filling."""
    mod = import_module(f"bt_agent._engine.agent.system_prompts.{variant}")
    bundle = getattr(mod, "PROMPT_BUNDLE", None)
    if not isinstance(bundle, PromptBundle):
        raise TypeError(
            f"Variant {variant!r} did not export a PromptBundle "
            f"(got {type(bundle).__name__ if bundle is not None else 'None'})"
        )
    return bundle


def get_prompts(variant: str = "tiamat_agent_submit_v5") -> PromptBundle:
    """Load the ``PROMPT_BUNDLE`` from the named variant package.

    Raises:
        ValueError: if the variant doesn't exist.
    """
    available = list_variants()
    if variant not in available:
        raise ValueError(f"Unknown prompt variant {variant!r}. Available: {available}")

    return _load_bundle(variant)


__all__ = ["PromptBundle", "get_prompts", "list_variants"]
