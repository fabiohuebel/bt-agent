"""Build the LAM agent behavior tree.

The tree structure matches the diagram in ARCHITECTURE.md:

    Root: Sequence "agent_tick"
    ├── CheckCancellation
    └── Selector "main"
        ├── Sequence "has_plan"
        │   ├── CheckHasPlan
        │   └── Selector "milestone_or_done"
        │       ├── Sequence "work_milestone"
        │       │   ├── CheckMilestonesRemain
        │       │   └── Selector "skill_or_eval"
        │       │       ├── Sequence "evaluate"   (Step 3 — three-way verdict)
        │       │       │   ├── CheckCompletionClaimed
        │       │       │   ├── EvaluateCompletion           (writes BB.LAST_VERDICT)
        │       │       │   └── Selector "verdict_route"
        │       │       │       ├── Sequence "pass_route"   → CheckVerdictPass + AdvanceMilestone
        │       │       │       ├── Sequence "amend_route"  → CheckVerdictAmend + ApplyAmend
        │       │       │       └── Sequence "replan_route" → CheckVerdictReplan + ApplyReplan
        │       │       └── Sequence "execute"
        │       │           ├── Selector "select_or_resume"
        │       │           │   ├── CheckSkillRunning
        │       │           │   └── SelectSkill
        │       │           └── ExecuteSkill
        │       └── Sequence "all_done"
        │           ├── CheckAllMilestonesDone
        │           └── HandleAllDone  (real bridge submit after completion proof)
        └── PlanTask
"""

from __future__ import annotations

import py_trees
from bt_agent._engine.bt.nodes.advance_milestone import AdvanceMilestone
from bt_agent._engine.bt.nodes.conditions import (
    CheckAllMilestonesDone,
    CheckCancellation,
    CheckCompletionClaimed,
    CheckHasPlan,
    CheckMilestonesRemain,
    CheckReactFallbackActive,
    CheckSkillRunning,
)
from bt_agent._engine.bt.nodes.evaluate_completion import EvaluateCompletion
from bt_agent._engine.bt.nodes.execute_skill import ExecuteSkill
from bt_agent._engine.bt.nodes.handle_all_done import HandleAllDone
from bt_agent._engine.bt.nodes.plan_task import PlanTask
from bt_agent._engine.bt.nodes.route_verdict import (
    ApplyAmend,
    ApplyReplan,
    CheckVerdictAmend,
    CheckVerdictPass,
    CheckVerdictReplan,
    GiveUpOnReplan,
)
from bt_agent._engine.bt.nodes.select_skill import SelectSkill


def build_agent_tree(
    *,
    structural_fixes: bool = False,
    ablation: frozenset[str] = frozenset(),
) -> py_trees.behaviour.Behaviour:
    """Construct the full agent behavior tree. Returns the root node.

    ``structural_fixes`` (docs/spec/lam-improvement.md) is OFF by default, which
    returns the exact frozen tree — byte-identical for every existing variant.
    When ON it adds two branches:
      * CP2 — a react-fallback pass route inside ``skill_or_eval`` so a claim in
        react-fallback mode advances directly (no evaluator, no re-plan loop);
      * CP3 — a real ``all_done`` submit node (``HandleAllDone``) in place of the
        Phase-2 ``HandleDeferredOrSubmit`` stub. Completion is set only after the
        substrate confirms the canonical terminal submit action.

    ``ablation`` (docs/spec/tiamat-ablation-flags.md) is empty by default, which
    also returns the byte-identical frozen shape. Two members are live here:
      * ``"evaluator"``: no evaluator model call ever. The ``evaluate`` branch
        is replaced by an unconditional auto-accept (``CheckCompletionClaimed``
        -> ``AdvanceMilestone``, the same shape as the CP2 ``react_fallback_pass``
        route above). ``EvaluateCompletion`` is never built, so its amend-cap
        promotion (evaluate_completion.py:284) is unreachable by construction:
        there is no AMEND to escalate when the evaluator never runs. The same
        flag also threads into ``HandleAllDone(evaluator_ablated=...)`` below,
        so plan exhaustion is terminal on the first pass there too, instead of
        burning the one corrective replan every episode (see that node).
      * ``"replan"``: the evaluator still runs (unless also ablated); only the
        ``replan_route`` action changes: ``GiveUpOnReplan`` replaces
        ``ApplyReplan`` so a REPLAN verdict (raw from the evaluator, or
        cap-promoted inside ``EvaluateCompletion.update()``) ends the episode as
        an honest give-up instead of wiping ``BB.MILESTONES`` and re-firing
        ``PlanTask``.
      * ``"planner"`` (CP3 review round, docs/spec/tiamat-ablation-flags.md):
        also routes REPLAN to ``GiveUpOnReplan``, exactly as the "replan"
        ablation does, even when "replan" itself is not ablated. Replanning a
        static single-milestone plan is meaningless -- ``PlanTask`` (CP3) never
        calls the model and always reinstalls the identical task-text
        milestone, so letting ``ApplyReplan`` wipe and re-fire would only cycle
        (verdict -> wipe -> replan -> claim -> verdict -> ...) to the tick cap
        instead of terminating.
    """
    evaluator_ablated = "evaluator" in ablation
    replan_ablated = "replan" in ablation
    planner_ablated = "planner" in ablation

    if evaluator_ablated:
        # "evaluator" ablation: skip the verdict machinery entirely, no
        # EvaluateCompletion, no pass/amend/replan routing. Any claim advances.
        evaluate = py_trees.composites.Sequence(
            name="auto_accept",
            memory=True,
            children=[
                CheckCompletionClaimed(),
                AdvanceMilestone(),
            ],
        )
    else:
        # --- Verdict routing (Step 3): three terminal actions, one per verdict ---
        # EvaluateCompletion writes BB.LAST_VERDICT (with cap-promotion already
        # applied), then this Selector picks the matching branch.
        pass_route = py_trees.composites.Sequence(
            name="pass_route",
            memory=True,
            children=[
                CheckVerdictPass(),
                AdvanceMilestone(),
            ],
        )
        amend_route = py_trees.composites.Sequence(
            name="amend_route",
            memory=True,
            children=[
                CheckVerdictAmend(),
                ApplyAmend(),
            ],
        )
        replan_route = py_trees.composites.Sequence(
            name="replan_route",
            memory=True,
            children=[
                CheckVerdictReplan(),
                GiveUpOnReplan() if (replan_ablated or planner_ablated) else ApplyReplan(),
            ],
        )
        verdict_route = py_trees.composites.Selector(
            name="verdict_route",
            memory=False,
            children=[
                pass_route,
                amend_route,
                replan_route,
            ],
        )

        # --- Evaluate branch: completion check → run evaluator → route on verdict ---
        evaluate = py_trees.composites.Sequence(
            name="evaluate",
            memory=True,
            children=[
                CheckCompletionClaimed(),
                EvaluateCompletion(),
                verdict_route,
            ],
        )

    # --- Select or resume: skip SelectSkill if a skill is already running ---
    select_or_resume = py_trees.composites.Selector(
        name="select_or_resume",
        memory=False,
        children=[
            CheckSkillRunning(),
            SelectSkill(),
        ],
    )

    # --- Execute branch: select a skill then run it ---
    execute = py_trees.composites.Sequence(
        name="execute",
        memory=True,
        children=[
            select_or_resume,
            ExecuteSkill(),
        ],
    )

    # --- Skill or eval: try evaluation first, fall back to execution ---
    # CP2 react-fallback (gated): when the flat react-fallback plan is active, a
    # claimed completion advances directly — no evaluator LLM call, no verdict
    # that could route REPLAN and re-enter the empty-plan loop. When the fallback
    # is active but nothing is claimed yet, this route fails at
    # CheckCompletionClaimed and control falls through to execute (the actor acts),
    # giving the flat observe→reason→act loop. Absent entirely on frozen variants.
    skill_or_eval_children = []
    if structural_fixes:
        react_fallback_pass = py_trees.composites.Sequence(
            name="react_fallback_pass",
            memory=True,
            children=[
                CheckReactFallbackActive(),
                CheckCompletionClaimed(),
                AdvanceMilestone(),
            ],
        )
        skill_or_eval_children.append(react_fallback_pass)
    skill_or_eval_children.extend([evaluate, execute])
    skill_or_eval = py_trees.composites.Selector(
        name="skill_or_eval",
        memory=False,
        children=skill_or_eval_children,
    )

    # --- Work milestone: check remaining then do skill/eval ---
    work_milestone = py_trees.composites.Sequence(
        name="work_milestone",
        memory=True,
        children=[
            CheckMilestonesRemain(),
            skill_or_eval,
        ],
    )

    # --- All done: all milestones completed ---
    # CP3 (gated): HandleAllDone replaces the Phase-2 HandleDeferredOrSubmit stub —
    # evaluator-approved exhaustion issues a real bridge submit; the missing-
    # proof fallback gets one corrective replan, and its spent/give-up state also
    # issues a real submit. Frozen tree keeps the bare CheckAllMilestonesDone
    # (byte-identical).
    all_done_children = [CheckAllMilestonesDone()]
    if structural_fixes:
        # "evaluator" ablation reaches HandleAllDone too, not just the
        # per-milestone auto_accept route above: board_runner._build_agent
        # hard-requires structural_fixes=True for every real run, and without
        # this HandleAllDone's _evaluator_passed() would be permanently False
        # (no evaluator model ever writes BB.LAST_VERDICT), burning the one
        # corrective replan on the first plan exhaustion of every episode.
        all_done_children.append(HandleAllDone(evaluator_ablated=evaluator_ablated))
    all_done = py_trees.composites.Sequence(
        name="all_done",
        memory=True,
        children=all_done_children,
    )

    # --- Milestone or done: try working a milestone, else check if all done ---
    milestone_or_done = py_trees.composites.Selector(
        name="milestone_or_done",
        memory=False,
        children=[
            work_milestone,
            all_done,
        ],
    )

    # --- Has plan: check plan exists, then work milestones ---
    has_plan = py_trees.composites.Sequence(
        name="has_plan",
        memory=True,
        children=[
            CheckHasPlan(),
            milestone_or_done,
        ],
    )

    # --- Main selector: use existing plan or create one ---
    main = py_trees.composites.Selector(
        name="main",
        memory=False,
        children=[
            has_plan,
            PlanTask(),
        ],
    )

    # --- Root: check cancellation then proceed ---
    root = py_trees.composites.Sequence(
        name="agent_tick",
        memory=True,
        children=[
            CheckCancellation(),
            main,
        ],
    )

    return root  # noqa: R504
