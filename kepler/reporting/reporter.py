from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Generic, Iterable, Protocol, TypeVar

import numpy as np
import numpy.typing as npt
from rich import console, table

from .event import CallStack, Event
from .format import (
    Formatter,
    FormatMetadata,
    Pretty,
    Sparkline,
    TimedeltaFormatter,
)
from ..timer import Timer, TimerContext


T = TypeVar("T")


class Reporter(Protocol):
    def report(self, ctx: TimerContext):
        ...


@dataclass
class Metric(Generic[T]):
    name: str
    compute: Callable[[npt.NDArray[np.float64]], T]
    formatter: Formatter[T] = Pretty()
    rich_args: dict[str, Any] = field(default_factory=dict)

    def format(self, event: Event, meta: FormatMetadata):
        value = self.compute(event.times)
        return self.formatter.format(value, meta)


DEFAULT_METRICS = (
    Metric("Count", len),
    Metric("Total", np.sum, formatter=TimedeltaFormatter()),
    Metric("Average", np.mean, formatter=TimedeltaFormatter()),
    Metric("Min", np.min, formatter=TimedeltaFormatter()),
    Metric("Histogram", partial(np.histogram, bins=20), formatter=Sparkline()),
    Metric("Max", np.max, formatter=TimedeltaFormatter()),
    Metric("P50", partial(np.percentile, q=50), formatter=TimedeltaFormatter()),
    Metric("P90", partial(np.percentile, q=90), formatter=TimedeltaFormatter()),
    Metric("P99", partial(np.percentile, q=99), formatter=TimedeltaFormatter()),
)


def flat_timers(
    ctx: TimerContext, call_stack: CallStack = []
) -> Iterable[tuple[CallStack, Timer]]:
    for caller_id, timer in ctx.timers.items():
        stack = call_stack + [caller_id.label]
        yield stack, timer
        yield from flat_timers(timer.context, stack)
    for caller_id, sw_ctx in ctx.stopwatches.items():
        name = f":stopwatch: {caller_id.label}"
        yield from flat_timers(sw_ctx, call_stack + [name])


def flat_events(ctx: TimerContext) -> list[Event]:
    return [
        Event(call_stack, timer.events)
        for call_stack, timer in flat_timers(ctx)
    ]


def common_prefix(l: CallStack, r: CallStack) -> CallStack:
    for i, (lv, rv) in enumerate(zip(l, r)):
        if lv != rv:
            return l[:i]
    return l[: len(r)]


def indent_label(call_stack: CallStack, indent: str = "  ") -> str:
    return indent * (len(call_stack) - 1) + call_stack[-1]


@dataclass
class RichReporter:
    name: str
    metrics: tuple[Metric, ...] = DEFAULT_METRICS

    def report(self, ctx: TimerContext):
        # Report a table with metrics as column names, events as rows
        name = self.name
        title = f"Timings for [b][blue]{name} :stopwatch:[/blue][/b]"
        report = table.Table(
            title=title, row_styles=("", "on black"), title_style="white"
        )

        events = flat_events(ctx)
        # TODO: range upper bound should probably include sums
        meta = FormatMetadata(events)

        summary = None
        if events and not events[0].call_stack:
            # First event is summary
            summary, *events = events
            report.show_footer = True

        # Columns are metrics, plus "Stage" at the beginning for labels
        report.add_column(
            "Stage", footer="Total" if summary else None, style="bold blue"
        )

        for metric in self.metrics:
            kwargs = {"justify": "right", **metric.rich_args}
            footer = metric.format(summary, meta) if summary else None
            report.add_column(metric.name, footer=footer, **kwargs)

        # Rows are events
        for prev_event, event in zip([None, *events], events):
            if prev_event:  # Add context rows if necessary
                prefix = common_prefix(prev_event.call_stack, event.call_stack)
                for i in range(len(prefix) + 1, len(event.call_stack)):
                    report.add_row(indent_label(event.call_stack[:i]))

            cells = [metric.format(event, meta) for metric in self.metrics]
            report.add_row(indent_label(event.call_stack), *cells)

        console.Console().print(report)
