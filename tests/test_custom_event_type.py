from dataclasses import dataclass
from kepler.event import Event
from kepler.scope import Scope
from kepler import measurement

from .conftest import assert_log_json_roundtrip, event_counts


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


def test_simple_scope(scope: Scope):
    with scope:
        with tick("simple"):
            pass
    assert_log_json_roundtrip(scope)

    assert event_counts(scope) == {
        TickEvent: [
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
        TickEvent: [
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
        TickEvent: [
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
        TickEvent: [
            (("range",), 20),
        ]
    }
