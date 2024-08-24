from __future__ import annotations

import collections
from dataclasses import dataclass
import inspect
from typing import Mapping


@dataclass(frozen=True)
class CallerID:
    label: str
    filename: str
    lineno: int

    @classmethod
    def from_frame(cls, label: str, frame: inspect.FrameInfo):
        assert frame.positions
        return cls(label, frame.filename, frame.positions.lineno or 0)

    @classmethod
    def from_fn(cls, fn: function):
        return cls(
            fn.__qualname__, fn.__code__.co_filename, fn.__code__.co_firstlineno
        )

    @classmethod
    def from_caller(cls, label: str, context: int = 2):
        frame = inspect.stack(context=context)[-context]
        return cls.from_frame(label, frame)


TimingKey = tuple[CallerID, ...]


@dataclass
class TimingEntry:
    caller_id: CallerID
    start_time: float


class Snapshot:
    name: str
    start: float
    times: Mapping[TimingKey, list[float]]

    def __init__(self, name: str, start: float):
        self.name = name
        self.start = start
        self.times = collections.defaultdict(list)
