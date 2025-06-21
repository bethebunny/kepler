from __future__ import annotations

from dataclasses import dataclass

# Handle TypeAlias compatibility between Python versions
try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias


CallStack: TypeAlias = list[str]


@dataclass
class Event:
    call_stack: CallStack
    times: list[float]
