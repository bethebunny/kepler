from dataclasses import dataclass

import kepler
from kepler.event import Log
from kepler.timer import TimerContext


# test_json_logs.py has much more thorough tests.
# - It probably makes sense to restructure the test files.
# - One option is to take the examples from test_json_logs as fixtures
#   to run as test cases for various behaviors.


CallStackLabels = tuple[str, ...]


@dataclass
class LogStructure:
    event_counts: list[tuple[CallStackLabels, int]]


def log_structure(log: Log) -> LogStructure:
    return LogStructure(
        event_counts=[
            (tuple(e.label for e in event.call_stack), len(event.events))
            for event in log.events
        ]
    )


def test_decorator(context: TimerContext):
    @kepler.time
    def f():
        pass

    with context:
        f()
    log = Log.from_events(context.export())
    # Uses fully qualified function name
    assert log_structure(log) == LogStructure(
        event_counts=[(("test_decorator.<locals>.f",), 1)]
    )


def test_split(context: TimerContext):
    with context:
        split = kepler.stopwatch("watch")
        split("1")
        split("2")
    log = Log.from_events(context.export())
    assert len(log.events) == 2


def test_iter(context: TimerContext):
    with context:
        for _ in kepler.time("loop", range(10)):
            pass
    log = Log.from_events(context.export())
    assert log_structure(log) == LogStructure(event_counts=[(("loop",), 10)])


def test_empty_iter(context: TimerContext):
    with context:
        for _ in kepler.time("loop", []):
            pass
    log = Log.from_events(context.export())
    assert log_structure(log) == LogStructure(event_counts=[(("loop",), 0)])
