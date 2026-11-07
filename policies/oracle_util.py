"""Helpers shared by the analysis-only oracle policies.

Nothing here reads the future by itself: the clocks and hour arithmetic are
plain functions of the payload timestamp that ``sbx.evalrun --oracle`` hands to
``observe_oracle`` (the simulator's own local wall clock at the step the policy
is about to act on).
"""

from __future__ import annotations

import datetime


def clock(stamp) -> tuple[int | None, float]:
    """``(weekday, hour)`` from the oracle payload's ``timestamp`` string.

    The simulator clock is naive local wall clock (see the forum digest B6), so
    the leading ``YYYY-MM-DD HH:MM:SS`` fields are used as-is. Returns
    ``(None, nan)``-free values only; a malformed stamp yields ``(None, 0.0)``
    so a policy degrades to "clock unknown" instead of raising mid-episode.
    """
    if not stamp or len(str(stamp)) < 16:
        return None, 0.0
    text = str(stamp)
    try:
        year, month, day = int(text[0:4]), int(text[5:7]), int(text[8:10])
        hour = int(text[11:13]) + int(text[14:16]) / 60.0
        weekday = datetime.date(year, month, day).weekday()
    except (ValueError, IndexError):
        return None, 0.0
    return weekday, hour


def hours_until(target_hour: float, hour: float) -> float:
    """Hours from ``hour`` forward to the next occurrence of ``target_hour``."""
    return (target_hour - hour) % 24.0
