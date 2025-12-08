from __future__ import annotations

import contextlib
import dataclasses
from dataclasses import dataclass, field
from functools import partial
from time import perf_counter_ns, time_ns
from typing import Callable, Sequence

import numpy as np
from rich import text

from .event import Event
from .measurement import measurement
from .reporting.color import HLSColorGradient
from .reporting.format import Metadata, Sparkline
from .reporting.units import Time
from .reporting import Statistic, RichReporter
from .report import register_default_report

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
    return perf_counter_ns() + OFFSET


@measurement
def time():
    start = current_time()
    yield
    return TimingEvent(start, current_time() - start)


stopwatch = time.stopwatch


@dataclass
class TimedeltaFormatter:
    gradient: HLSColorGradient = field(default_factory=HLSColorGradient)

    def format(self, nanos: int, meta: Metadata[int]) -> text.Text:
        color = self.gradient.color(nanos, meta.data_range)
        return text.Text(Time.format_nanos(nanos), style=color)


def p(q: float) -> Callable[[Sequence[float]], float]:
    return partial(np.percentile, q=q)


def duration(event: TimingEvent):
    return event.duration


# fmt: off
TIMING_STATISTICS = (
    Statistic("Count", len),
    Statistic("Total", sum, duration, formatter=TimedeltaFormatter()),
    Statistic("Average", np.mean, duration, formatter=TimedeltaFormatter()),
    Statistic("Min", min, duration, formatter=TimedeltaFormatter()),
    Statistic("Histogram", partial(np.histogram, bins=20), duration, formatter=Sparkline()),
    Statistic("Max", max, duration, formatter=TimedeltaFormatter()),
    Statistic("P50", p(50), duration, formatter=TimedeltaFormatter()),
    Statistic("P90", p(90), duration, formatter=TimedeltaFormatter()),
    Statistic("P99", p(99), duration, formatter=TimedeltaFormatter()),
)
# fmt: on


register_default_report(
    RichReporter(
        title="Timings for [b][blue]{name} :stopwatch:[/blue][/b]",
        event_type=TimingEvent,
        statistics=TIMING_STATISTICS,
    )
)
