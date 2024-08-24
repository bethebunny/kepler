from __future__ import annotations

import collections
import contextlib
import contextvars
from dataclasses import dataclass
import inspect
import timeit
import typing
from typing import Callable, Mapping


current_time = timeit.default_timer


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


class Timer:
    def __init__(self):
        self.timers: Mapping[CallerID, Timer] = collections.defaultdict(Timer)
        self.events: list[float] = []
        self.current_split: float = 0

    @contextlib.contextmanager
    def time(self, label: str | CallerID):
        start = current_time()
        caller_id = label if isinstance(
            label, CallerID
        ) else CallerID.from_caller(label)
        timer = self.timers[caller_id]
        # Using a contextvar allows mutually- and self-recursive timers
        token = _STATE.set(TimerState(timer, start, start))
        try:
            yield
        finally:
            timer._publish(current_time() - start)
            _STATE.reset(token)

    def _publish(self, event: float):
        self.events.append(event)

    def split(self, label: str | CallerID):
        """Publish a single split event."""
        time = current_time()
        caller_id = label if isinstance(
            label, CallerID
        ) else CallerID.from_caller(label)
        state = _STATE.get()
        assert state.timer is self
        self.timers[caller_id]._publish(time - state.current_split)
        state.current_split = time


@dataclass
class TimerState:
    timer: Timer
    start: float
    current_split: float


_STATE = contextvars.ContextVar[TimerState]("_STATE")
_STATE.set(TimerState(Timer(), current_time(), current_time()))


P = typing.ParamSpec("P")
R = typing.TypeVar("R")


@typing.overload
def time(label: str) -> contextlib._GeneratorContextManager[None]:  # type: ignore
    ...


@typing.overload
def time(label: Callable[P, R]) -> Callable[P, R]:
    ...


def time(label: str | Callable[P, R]):
    if isinstance(label, str):
        return _time(CallerID.from_caller(label))
    else:
        return _time(CallerID.from_fn(label))(label)


@contextlib.contextmanager
def _time(caller_id: CallerID):
    with _STATE.get().timer.time(caller_id):
        yield


def split(label: str):
    caller_id = CallerID.from_caller(label)
    return _STATE.get().timer.split(caller_id)


def report(name: str = ""):
    from . import reporting

    state = _STATE.get()
    timer = state.timer

    # Special case: If the root timer has only one trivial entry,
    # we report that entry instead.
    if len(timer.timers) == 1:
        caller_id, subtimer = next(iter(timer.timers.items()))
        if len(subtimer.events) == 1:
            name = name or caller_id.label
            timer = subtimer

    reporting.report(timer, name)
