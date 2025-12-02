import kepler
from kepler.context import Context
from kepler.log import Log
from kepler import measurement

from .conftest import assert_log_json_roundtrip, log_structure, LogStructure


@measurement
def tick():
    yield
    return kepler.timer.TimingEvent(0, 1)


def test_simple_context(context: Context):
    with context:
        with tick("simple"):
            pass
    log = Log.from_context(context)
    assert_log_json_roundtrip(log)

    assert log_structure(log) == LogStructure(
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

    assert log_structure(log) == LogStructure(
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

    assert log_structure(log) == LogStructure(
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

    assert log_structure(log) == LogStructure(
        event_counts=[
            (("range",), 20),
        ]
    )
