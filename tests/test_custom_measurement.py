import kepler
from kepler.scope import Scope
from kepler import TimingEvent, measurement

from .conftest import assert_log_json_roundtrip, event_counts


@measurement
def tick():
    yield
    return kepler.timer.TimingEvent(0, 1)


def test_simple_scope(scope: Scope):
    with scope:
        with tick("simple"):
            pass
    assert_log_json_roundtrip(scope)

    assert event_counts(scope) == {
        TimingEvent: [
            (("simple",), 1),
        ]
    }


def test_nested_functions(scope: Scope):
    @tick("outer")
    def outer():
        inner()

    @tick("inner")
    def inner():
        pass

    with scope:
        outer()

    assert_log_json_roundtrip(scope)

    assert event_counts(scope) == {
        TimingEvent: [
            (("outer",), 1),
            (("outer", "inner"), 1),
        ]
    }


def test_stopwatch_splits(scope: Scope):
    with scope:
        split = tick.stopwatch("watch")
        split("start")
        split("middle")
        split("end")

    assert_log_json_roundtrip(scope)

    assert event_counts(scope) == {
        TimingEvent: [
            ((":stopwatch: watch", "start"), 1),
            ((":stopwatch: watch", "middle"), 1),
            ((":stopwatch: watch", "end"), 1),
        ]
    }


def test_iter(scope: Scope):
    with scope:
        for _ in tick("range", range(20)):
            pass

    assert_log_json_roundtrip(scope)

    assert event_counts(scope) == {
        TimingEvent: [
            (("range",), 20),
        ]
    }
