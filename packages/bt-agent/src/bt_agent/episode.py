"""Backend-neutral report/submission host. No goal oracle or success grade."""

from typing import Callable

from ._engine.episode_bridge import RosClaimBridge


class EpisodeHost(RosClaimBridge):
    """Reuse the evaluated acknowledgement protocol without a ROS dependency.

    Callbacks must persist reports/artifacts before returning. The host does not
    declare that a task succeeded and does not expose an actor-selectable submit.
    """

    def __init__(self, publish_report: Callable[[str], None], flush_artifacts: Callable[[], None]):
        super().__init__(publish_report, flush_artifacts)
