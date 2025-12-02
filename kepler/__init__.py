import contextlib

from .event import CallerID
from .log import Log
from .measurement import measurement
from .report import report
from .timer import stopwatch, time


@contextlib.contextmanager
def time_and_report(label: str):
    try:
        with time(CallerID.from_caller(label)):
            yield
    finally:
        report(label)


__all__ = ["Log", "measurement", "report", "stopwatch", "time", "time_and_report"]
