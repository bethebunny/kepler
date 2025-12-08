from __future__ import annotations

import contextlib
from contextlib import _GeneratorContextManager
import functools
from types import FunctionType
import typing
from typing import Callable, Generator, Generic, Iterable, Protocol, TypeVar

from .event import CallerID, Event
from .scope import Scope


P = typing.ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")
EventType = TypeVar("EventType", bound=Event)
MeasurementManager: typing.TypeAlias = Callable[P, Generator[None, None, EventType]]


class Measurement(Protocol, Generic[EventType]):
    @typing.overload
    def __call__(self, label: str | CallerID) -> _GeneratorContextManager[None]: ...
    @typing.overload
    def __call__(self, label: str | CallerID, it: Iterable[T]) -> Iterable[T]: ...
    @typing.overload
    def __call__(self, fn: Callable[P, R]) -> Callable[P, R]: ...

    stopwatch: Callable[[str], Stopwatch[EventType]]


def measure_iter(
    measure: Callable[typing.Concatenate[str, P], _GeneratorContextManager[None]],
    caller_id: CallerID,
    it: Iterable[T],
) -> Iterable[T]:
    # Always add the scope, even if no iterations
    _ = Scope.current[caller_id]
    it = iter(it)
    while True:
        try:
            v = next(it)
        except StopIteration:
            break
        with measure(caller_id):
            yield v


def _coro_return(coro: typing.Generator[None, None, EventType]) -> EventType:
    try:
        coro.send(None)
    except StopIteration as si:
        return si.value
    else:
        raise RuntimeError("Coroutine yielded")


class Stopwatch(Generic[EventType]):
    measure_raw: MeasurementManager[P, EventType]
    scope: Scope
    coro: typing.Generator[None, None, EventType]
    kwargs: dict[str, object]

    def __init__(self, measure_raw, label: str, **kwargs):
        caller_id = CallerID.from_caller(f":stopwatch: {label}")
        self.measure_raw = measure_raw
        self.scope = Scope.current[caller_id]
        self.kwargs = kwargs
        self.start()

    def start(self):
        self.coro = self.measure_raw(**self.kwargs)
        self.coro.send(None)  # kick

    def __call__(self, label: str):
        event = _coro_return(self.coro)
        if event is not None:
            self.scope[CallerID.from_caller(label)].log(event)
        self.start()


def measurement(f: MeasurementManager[P, EventType]) -> Measurement[EventType]:
    # This is key to correctness of decorators. @contextmanagers can be used
    # as context managers _or_ as decorators, so by delaying creation like this
    # we allow the decorator to retrieve the _dynamic_ timer scope, rather than
    # the scope at decoration time.
    @contextlib.contextmanager
    def measure(caller_id: CallerID, **kwargs):
        with Scope.current[caller_id] as scope:
            event = yield from f(**kwargs)
            if event is not None:
                scope.log(event)

    @functools.wraps(f)
    def wrapped(
        label: str | CallerID | FunctionType,
        it: Iterable[T] | None = None,
        **kwargs,
    ):
        if isinstance(label, str):
            label = CallerID.from_caller(label)
        if isinstance(label, CallerID):
            if it is None:
                return measure(label, **kwargs)
            return measure_iter(measure, label, it, **kwargs)
        else:
            caller_id = CallerID.from_fn(label)
            return measure(caller_id, **kwargs)(label)

    measurement = typing.cast(Measurement[EventType], wrapped)
    measurement.stopwatch = functools.partial(Stopwatch, f)

    return measurement
