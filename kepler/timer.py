from __future__ import annotations

from dataclasses import dataclass
import collections
import contextlib
import contextvars
from time import perf_counter_ns as current_time, time_ns
import typing
from typing import Callable, Generator, Iterable, Mapping, Optional

try:
    from typing import ParamSpec
except ImportError:  # python 3.9
    from typing_extensions import ParamSpec

from .event import CallerID, Log, ScopedEvents, TimingEvent


GeneratorContextManager = contextlib._GeneratorContextManager  # type: ignore

# Always use `current_time` to get a timestamp.
# - This is the single source of truth for timestamps
# - This time is not guaranteed to map to the system time!


@dataclass
class ExportContext:
    perf_counter_ns_offset: int

    def __init__(self):
        self.perf_counter_ns_offset = time_ns() - current_time()

    def convert(self, event: TimingEvent) -> TimingEvent:
        return TimingEvent(
            timestamp=event.timestamp + self.perf_counter_ns_offset,
            duration=event.duration,
        )


class Timer:
    def __init__(self):
        self.context = TimerContext()
        self.events: list[TimingEvent] = []

    def log(self, start_time: float):
        time = current_time()
        self.events.append(
            TimingEvent(timestamp=start_time, duration=time - start_time)
        )
        return time

    @contextlib.contextmanager
    def time(self):
        with self.context:
            start = current_time()
            try:
                yield self
            finally:
                self.log(start)

    def time_iter(self, it: Iterable[T]) -> Generator[T]:
        current_iter = current_time()
        with self.context:
            for value in it:
                yield value
                current_iter = self.log(current_iter)

    def export(self, ctx: ExportContext | None = None) -> Iterable[ScopedEvents]:
        ctx = ctx or ExportContext()
        yield ScopedEvents(call_stack=(), events=[ctx.convert(e) for e in self.events])
        yield from self.context.export(ctx)


class TimerContext:
    def __init__(self):
        self.timers: Mapping[CallerID, Timer] = collections.defaultdict(Timer)
        self.stopwatches: Mapping[CallerID, TimerContext] = collections.defaultdict(
            TimerContext
        )
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

    def export(self, ctx: ExportContext | None = None) -> Iterable[ScopedEvents]:
        ctx = ctx or ExportContext()
        for caller_id, timer in self.timers.items():
            for events in timer.export(ctx):
                yield events.nest_under(caller_id)
        for caller_id, sw_ctx in self.stopwatches.items():
            sw_caller_id = CallerID(
                f":stopwatch: {caller_id.label}",
                caller_id.filename,
                caller_id.lineno,
            )
            for events in sw_ctx.export(ctx):
                yield events.nest_under(sw_caller_id)


_CURRENT_CONTEXT = contextvars.ContextVar[TimerContext]("_CURRENT_CONTEXT")
_CURRENT_CONTEXT.set(TimerContext())


def current_context() -> TimerContext:
    return _CURRENT_CONTEXT.get()


P = ParamSpec("P")
R = typing.TypeVar("R")
T = typing.TypeVar("T")


@typing.overload
def time(label: str) -> GeneratorContextManager[None]: ...


@typing.overload
def time(label: str, it: Iterable[T]) -> Generator[T]: ...


@typing.overload
def time(label: Callable[P, R]) -> Callable[P, R]: ...


def time(label: str | Callable[P, R], it: Optional[Iterable[T]] = None):
    if isinstance(label, str):
        caller_id = CallerID.from_caller(label)
        if it is None:
            return _time(caller_id)
        return current_context()[caller_id].time_iter(it)
    else:
        return _time(CallerID.from_fn(label))(label)


# This is key to correctness of decorators. @contextmanagers can be used
# as context managers _or_ as decorators, so by delaying creation like this
# we allow the decorator to retrieve the _dynamic_ timer context, rather than
# the context at decoration time.
@contextlib.contextmanager
def _time(caller_id: CallerID):
    with current_context()[caller_id].time() as timer:
        yield timer


def stopwatch(name: str):
    return current_context().stopwatch(name)


def report(name: str = "", log: Optional[Log] = None):
    from .reporting import RichReporter

    log = log or Log.from_events(current_context().export())
    reporter = RichReporter(name)
    reporter.report(log)


@contextlib.contextmanager
def time_and_report(label: str):
    try:
        with _time(CallerID.from_caller(label)):
            yield
    finally:
        report(label)
