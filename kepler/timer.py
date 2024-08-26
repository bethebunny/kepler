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
        self.timers: Mapping[CallerID, Timer] = collections.defaultdict(Timer)
        self.events: list[float] = []

    def publish(self, event: float):
        self.events.append(event)

    def split(self, caller_id: CallerID, time: float):
        self.timers[caller_id].publish(time)

    def stopwatch(self, name: str):
        name = f":stopwatch: {name}"

        timer = self.timers[CallerID.from_caller(name)]
        timer.events = self.events

        current_split = current_time()

        def split(label: str):
            nonlocal current_split
            time = current_time()
            timer.split(CallerID.from_caller(label), time - current_split)
            current_split = time

        return split

    @contextlib.contextmanager
    def context(self):
        token = _CURRENT_TIMER.set(self)
        try:
            yield self
        finally:
            _CURRENT_TIMER.reset(token)


_CURRENT_TIMER = contextvars.ContextVar[Timer]("_CURRENT_TIMER")
_CURRENT_TIMER.set(Timer())


def current_timer() -> Timer:
    return _CURRENT_TIMER.get()


P = typing.ParamSpec("P")
R = typing.TypeVar("R")
T = typing.TypeVar("T")


@typing.overload
def time(label: str) -> contextlib._GeneratorContextManager[None]:  # type: ignore
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
    with current_timer().timers[caller_id].context() as timer:
        start = current_time()
        try:
            yield timer
        finally:
            timer.publish(current_time() - start)


def _time_iter(caller_id: CallerID, it: Iterable[T]) -> Generator[T]:
    it = iter(it)
    current_iter = current_time()
    with _time(caller_id) as timer:
        for value in it:
            yield value
            time = current_time()
            timer.publish(time - current_iter)
            current_iter = time


def stopwatch(name: str):
    return current_timer().stopwatch(name)


def report(name: str = ""):
    timer = current_timer()

    # Special case: If the root timer has only one trivial entry,
    # we report that entry instead.
    if len(timer.timers) == 1:
        caller_id, subtimer = next(iter(timer.timers.items()))
        if len(subtimer.events) == 1:
            name = name or caller_id.label
            timer = subtimer

    from . import reporting

    reporter = reporting.RichReporter(name)
    reporter.report(timer)
