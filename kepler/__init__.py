import contextlib

from .event import CallerID, Event
from .measurement import measurement

# from .report import report
from .scope import log
from .timer import TimingEvent, stopwatch, time
from .report import report


@contextlib.contextmanager
def time_and_report(label: str):
    try:
        with time(CallerID.from_caller(label)):
            yield
    finally:
        report(label)


__all__ = [
    "CallerID",
    "Event",
    "TimingEvent",
    "log",
    "measurement",
    "report",
    "stopwatch",
    "time",
    "time_and_report",
]
