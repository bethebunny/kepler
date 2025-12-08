import contextlib

from .event import CallerID, Event
from .measurement import measurement

# from .report import report
from .scope import Scope, log
from .timer import TimingEvent, stopwatch, time
from .report import report


@contextlib.contextmanager
def time_and_report(label: str):
    caller_id = CallerID.from_caller(label)
    with time(caller_id):
        yield
    report(label, scope=Scope.current[caller_id])


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
