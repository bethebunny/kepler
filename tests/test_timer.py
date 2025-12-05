import kepler
from kepler import TimingEvent
from kepler.scope import Scope

from .conftest import event_counts

# test_json_logs.py has much more thorough tests.
# - It probably makes sense to restructure the test files.
# - One option is to take the examples from test_json_logs as fixtures
#   to run as test cases for various behaviors.


def test_decorator(scope: Scope):
    @kepler.time
    def f():
        pass

    with scope:
        f()
    # Uses fully qualified function name
    assert event_counts(scope) == {TimingEvent: [(("test_decorator.<locals>.f",), 1)]}


def test_split(scope: Scope):
    with scope:
        split = kepler.stopwatch("watch")
        split("1")
        split("2")
    assert event_counts(scope) == {
        TimingEvent: [
            ((":stopwatch: watch", "1"), 1),
            ((":stopwatch: watch", "2"), 1),
        ],
    }


def test_iter(scope: Scope):
    with scope:
        for _ in kepler.time("loop", range(10)):
            pass
    assert event_counts(scope) == {TimingEvent: [(("loop",), 10)]}


def test_empty_iter(scope: Scope):
    with scope:
        for _ in kepler.time("loop", []):
            pass
    assert event_counts(scope) == {}
