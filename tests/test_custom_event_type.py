from dataclasses import dataclass
from kepler.context import Context
from kepler.event import Event
from kepler.log import Log
from kepler.timer import TimingEvent
from kepler import measurement

from .conftest import assert_log_json_roundtrip, log_structure, LogStructure


@dataclass
class TickEvent(Event):
    @property
    def value(self):
        return 0

    def json(self):
        return {}


@measurement
def tick():
    yield
    return TickEvent()


def test_simple_context(context: Context):
    with context:
        with tick("simple"):
            pass
    log = Log.from_context(context)
    assert_log_json_roundtrip(log)

    # TimingEvent will show the same structure, but no rows
    assert log_structure(log, TimingEvent) == LogStructure(
        event_counts=[
            (("simple",), 0),
        ]
    )
    assert log_structure(log, TickEvent) == LogStructure(
        event_counts=[
            (("simple",), 1),
        ]
    )


def test_nested_functions(context: Context):
    @tick("outer")
    def outer():
        inner()

    @tick("inner")
    def inner():
        pass

    with context:
        outer()

    log = Log.from_context(context)
    assert_log_json_roundtrip(log)

    # TimingEvent will show the same structure, but no rows
    assert log_structure(log, TimingEvent) == LogStructure(
        event_counts=[
            (("outer",), 0),
            (("outer", "inner"), 0),
        ]
    )
    assert log_structure(log, TickEvent) == LogStructure(
        event_counts=[
            (("outer",), 1),
            (("outer", "inner"), 1),
        ]
    )


def test_stopwatch_splits(context: Context):
    with context:
        split = tick.stopwatch("watch")
        split("start")
        split("middle")
        split("end")

    log = Log.from_context(context)
    assert_log_json_roundtrip(log)

    # TimingEvent will show the same structure, but no rows
    assert log_structure(log, TimingEvent) == LogStructure(
        event_counts=[
            ((":stopwatch: watch",), 0),
            ((":stopwatch: watch", "start"), 0),
            ((":stopwatch: watch", "middle"), 0),
            ((":stopwatch: watch", "end"), 0),
        ]
    )
    assert log_structure(log, TickEvent) == LogStructure(
        event_counts=[
            ((":stopwatch: watch",), 0),
            ((":stopwatch: watch", "start"), 1),
            ((":stopwatch: watch", "middle"), 1),
            ((":stopwatch: watch", "end"), 1),
        ]
    )


def test_iter(context: Context):
    with context:
        for _ in tick("range", range(20)):
            pass

    log = Log.from_context(context)
    assert_log_json_roundtrip(log)

    # TimingEvent will show the same structure, but no rows
    assert log_structure(log, TimingEvent) == LogStructure(
        event_counts=[
            (("range",), 0),
        ]
    )
    assert log_structure(log, TickEvent) == LogStructure(
        event_counts=[
            (("range",), 20),
        ]
    )
