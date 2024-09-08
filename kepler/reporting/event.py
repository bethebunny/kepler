from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


CallStack: TypeAlias = list[str]


@dataclass
class Event:
    call_stack: CallStack
    times: list[float]
