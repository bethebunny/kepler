from __future__ import annotations

import dataclasses
import inspect
from dataclasses import dataclass
from time import time_ns
from types import FrameType, FunctionType
from typing import Mapping, ParamSpec, Protocol, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


@dataclass
class ExportContext:
    perf_counter_ns_offset: int

    def __init__(self):
        # Measurements should be able to register their export requirements.
        # For now this is a layering violation so import a cyclic dependency.
        from . import timer  # cyclic dependency

        self.perf_counter_ns_offset = time_ns() - timer.current_time()


@dataclass(frozen=True)
class CallerID:
    label: str
    filename: str
    lineno: int

    @classmethod
    def from_frame(cls, label: str, frame: FrameType):
        return cls(label, inspect.getfile(frame), frame.f_lineno)

    @classmethod
    def from_fn(cls, fn: FunctionType):
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
class ScopedEvents:
    call_stack: tuple[CallerID, ...]
    events: Mapping[type[Event], list[Event]]

    def nest_under(self, caller_id: CallerID) -> ScopedEvents:
        return ScopedEvents(
            call_stack=(caller_id, *self.call_stack), events=self.events
        )

    def pop_from_front(self) -> ScopedEvents:
        return ScopedEvents(call_stack=self.call_stack[1:], events=self.events)

    def json(self):
        return {
            "call_stack": [e.json() for e in self.call_stack],
            "events": {
                EventType.__name__: [e.json() for e in events]
                for EventType, events in self.events.items()
            },
        }

    @classmethod
    def from_json(cls, data: dict):
        return cls(
            call_stack=tuple(CallerID(**e) for e in data["call_stack"]),
            events={
                (EventType := Event.TYPES[typename]): [EventType(**e) for e in events]
                for typename, events in data["events"].items()
            },
        )


class Event(Protocol):
    TYPES: dict[str, type[Event]] = {}

    @property
    def value(self) -> float: ...

    def json(self) -> object:
        dict[str, object]

    def __init_subclass__(cls):
        super().__init_subclass__()
        cls.TYPES[cls.__qualname__] = cls

    def export(self, ctx: ExportContext):
        return self
