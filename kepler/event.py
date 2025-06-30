from __future__ import annotations

from dataclasses import dataclass
import dataclasses
import inspect
from types import FrameType
from typing import Callable, Iterable, Protocol, TypeVar

try:
    from typing import ParamSpec
except ImportError:  # python 3.9
    from typing_extensions import ParamSpec

P = ParamSpec("P")
R = TypeVar("R")


@dataclass(frozen=True)
class CallerID:
    label: str
    filename: str
    lineno: int

    @classmethod
    def from_frame(cls, label: str, frame: FrameType):
        return cls(label, inspect.getfile(frame), frame.f_lineno)

    @classmethod
    def from_fn(cls, fn: Callable[P, R]):
        code = fn.__code__
        return cls(fn.__qualname__, code.co_filename, code.co_firstlineno)

    @classmethod
    def from_caller(cls, label: str, depth: int = 1):
        frame = inspect.currentframe()
        for _ in range(depth + 1):
            frame = frame and frame.f_back
        if frame:
            return cls.from_frame(label, frame)
        return cls(label, "<unknown>", 0)

    def json(self):
        return dataclasses.asdict(self)


CallStack = tuple[CallerID, ...]


@dataclass
class Log:
    events: list[ScopedEvents]

    @classmethod
    def from_events(cls, events: Iterable[ScopedEvents]):
        return cls(events=list(events))

    def json(self):
        return [e.json() for e in self.events]

    @classmethod
    def from_json(cls, data: list[dict]):
        return cls(events=[ScopedEvents.from_json(e) for e in data])


@dataclass
class ScopedEvents:
    call_stack: tuple[CallerID, ...]
    events: list[TimingEvent]

    def nest_under(self, caller_id: CallerID) -> ScopedEvents:
        return ScopedEvents(
            call_stack=(caller_id, *self.call_stack), events=self.events
        )

    def pop_from_front(self) -> ScopedEvents:
        return ScopedEvents(call_stack=self.call_stack[1:], events=self.events)

    def json(self):
        return {
            "call_stack": [e.json() for e in self.call_stack],
            "events": [e.json() for e in self.events],
        }

    @classmethod
    def from_json(cls, data: dict):
        return cls(
            call_stack=tuple(CallerID(**e) for e in data["call_stack"]),
            events=[TimingEvent(**e) for e in data["events"]],
        )


class Event(Protocol):
    @property
    def value(self) -> float: ...


@dataclass
class TimingEvent:
    timestamp: int  # in nanoseconds
    duration: int  # in nanoseconds

    @property
    def value(self) -> float:
        return self.duration

    def json(self):
        return dataclasses.asdict(self)
