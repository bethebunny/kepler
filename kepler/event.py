from __future__ import annotations

import dataclasses
import inspect
from dataclasses import dataclass
from types import FrameType, FunctionType
from typing import ClassVar, MutableMapping, ParamSpec, TypeAlias, TypeVar

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
class Event:
    TYPES: ClassVar[dict[str, type[Event]]] = {}

    @property
    def value(self) -> float:
        raise NotImplementedError

    def json(self) -> object:
        return dataclasses.asdict(self)

    def __init_subclass__(cls):
        super().__init_subclass__()
        cls.TYPES[cls.__qualname__] = cls


E = TypeVar("E", bound=Event)
TypedEvents: TypeAlias = MutableMapping[type[E], list[E]]
