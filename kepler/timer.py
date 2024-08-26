from __future__ import annotations

import collections
import contextlib
import contextvars
from dataclasses import dataclass
import inspect
import timeit
from types import FrameType
import typing
from typing import Callable, Generator, Iterable, Mapping, Optional


GeneratorContextManager = contextlib._GeneratorContextManager  # type: ignore


current_time = timeit.default_timer


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


class Timer:
    def __init__(self):
        self.context = TimerContext()
        self.events: list[float] = []

    def log(self, start_time: float):
        self.events.append((time := current_time()) - start_time)
        return time


class TimerContext:
    def __init__(self):
        self.timers: Mapping[CallerID, Timer] = collections.defaultdict(Timer)
        self.stopwatches: Mapping[
            CallerID, TimerContext
        ] = collections.defaultdict(TimerContext)
        self._tokens: list[contextvars.Token[TimerContext]] = []

    def __getitem__(self, caller_id: CallerID) -> Timer:
        return self.timers[caller_id]

    def __enter__(self):
        self._tokens.append(_CURRENT_CONTEXT.set(self))
        return self

    def __exit__(self, *_):
        _CURRENT_CONTEXT.reset(self._tokens.pop())

    def stopwatch(self, name: str):
        ctx = self.stopwatches[CallerID.from_caller(name)]
        start = current_time()

        def split(label: str):
            nonlocal start
            start = ctx[CallerID.from_caller(label)].log(start)

        return split


_CURRENT_CONTEXT = contextvars.ContextVar[TimerContext]("_CURRENT_CONTEXT")
_CURRENT_CONTEXT.set(TimerContext())


def current_context() -> TimerContext:
    return _CURRENT_CONTEXT.get()


P = typing.ParamSpec("P")
R = typing.TypeVar("R")
T = typing.TypeVar("T")


@typing.overload
def time(label: str) -> GeneratorContextManager[None]:
    ...


@typing.overload
def time(label: str, it: Iterable[T]) -> Generator[T]:  # type: ignore
    ...


@typing.overload
def time(label: Callable[P, R]) -> Callable[P, R]:
    ...


def time(label: str | Callable[P, R], it: Optional[Iterable[T]] = None):
    if isinstance(label, str):
        caller = CallerID.from_caller(label)
        return _time(caller) if it is None else _time_iter(caller, it)
    else:
        return _time(CallerID.from_fn(label))(label)


@contextlib.contextmanager
def _time(caller_id: CallerID):
    timer = current_context()[caller_id]
    with timer.context:
        start = current_time()
        try:
            yield timer
        finally:
            timer.log(start)


def _time_iter(caller_id: CallerID, it: Iterable[T]) -> Generator[T]:
    it = iter(it)
    current_iter = current_time()
    timer = current_context()[caller_id]
    with timer.context:
        for value in it:
            yield value
            current_iter = timer.log(current_iter)


def stopwatch(name: str):
    return current_context().stopwatch(name)


def report(name: str = ""):
    from . import reporting

    reporter = reporting.RichReporter(name)
    reporter.report(current_context())
