from __future__ import annotations

import contextlib
import dataclasses
from dataclasses import dataclass
from time import perf_counter_ns, time_ns

from .event import Event
from .measurement import measurement

GeneratorContextManager = contextlib._GeneratorContextManager  # type: ignore

# Always use `current_time` to get a timestamp.
# - This is the single source of truth for timestamps
# - This time is not guaranteed to map to the system time!


@dataclass
class TimingEvent(Event):
    timestamp: int  # in nanoseconds
    duration: int  # in nanoseconds

    @property
    def value(self) -> float:
        return self.duration

    def json(self):
        return dataclasses.asdict(self)


OFFSET = time_ns() - perf_counter_ns()


def current_time():
    return perf_counter_ns() - OFFSET


@measurement
def time():
    start = current_time()
    yield
    return TimingEvent(start, current_time() - start)


stopwatch = time.stopwatch
