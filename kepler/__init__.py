import contextlib

from .event import CallerID, Event
from .log import Log
from .measurement import measurement
from .report import report
from .timer import TimingEvent, stopwatch, time


@contextlib.contextmanager
def time_and_report(label: str):
    try:
        with time(CallerID.from_caller(label)):
            yield
    finally:
        report(label)


__all__ = [
    "Event",
    "Log",
    "TimingEvent",
    "measurement",
    "report",
    "stopwatch",
    "time",
    "time_and_report",
]
