"""Pure arbitration + fail-closed brake logic for cmd_vel_mux.

Kept free of rclpy so the safety-critical decisions are unit-testable on
hosts without ROS. The node (cmd_vel_mux.py) owns subscriptions, timers,
and publishing; every decision about WHAT to publish lives here.

Why the mux is a brake authority (2026-08-14 incident): /stop is consumed
by pathFollower only, so a stale localPlanner goal can regenerate fresh
navigate commands the moment the mux wires, and primitive commands never
pass a brake at all. The robot moved on mux-up with the operator brake
engaged. The mux is the last software gate before /spot/cmd_vel, so it is
born braked, gates every input, and releases only on an explicit /stop==0
received after its own startup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

STOP_RELEASE_VALUE = 0


@dataclass
class MuxInput:
    name: str
    topic: str
    priority: int
    timeout_s: float
    last_msg: Optional[Any] = None
    last_ts: Optional[float] = None


def brake_released_after_stop(stop_value: Optional[int]) -> bool:
    """Fold one /stop observation into the released/braked state.

    None (never observed since startup) and every nonzero value mean
    braked; only an explicit 0 releases. There is deliberately no memory
    here — the latest observation alone decides, so a re-engage always
    re-gates.
    """
    return stop_value == STOP_RELEASE_VALUE


def select_command(inputs: List[MuxInput], now: float, brake_released: bool) -> Tuple[Optional[MuxInput], bool]:
    """Return (winning input or None, publish_zero).

    Fail-closed: while the brake is not released, the answer is zero no
    matter what the inputs carry. Otherwise: freshest-window inputs
    compete, highest priority wins, no fresh input falls back to zero.
    """
    if not brake_released:
        return None, True
    active = [i for i in inputs if i.last_msg is not None and i.last_ts is not None and (now - i.last_ts) < i.timeout_s]
    if not active:
        return None, True
    winner = max(active, key=lambda i: i.priority)
    return winner, False
