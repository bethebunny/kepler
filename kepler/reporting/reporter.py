from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Generic, Protocol, TypeVar

import numpy as np
from rich import console, table

from ..context import Context
from ..event import CallStack, ScopedEvents
from ..log import Log
from .format import (
    FormatMetadata,
    Formatter,
    Pretty,
    Sparkline,
    TimedeltaFormatter,
)

T = TypeVar("T")


class Reporter(Protocol):
    def report(self, ctx: Context): ...


@dataclass
class Metric(Generic[T]):
    name: str
    compute: Callable[[list[float]], T]
    formatter: Formatter[T] = Pretty()
    rich_args: dict[str, Any] = field(default_factory=dict)

    def format(self, scope: ScopedEvents, meta: FormatMetadata):
        events = [e.value for events in scope.events.values() for e in events]
        if not events:
            return ""
        value = self.compute(events)
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


def common_prefix(l: CallStack, r: CallStack) -> CallStack:
    for i, (lv, rv) in enumerate(zip(l, r)):
        if lv != rv:
            return l[:i]
    return l[: len(r)]


def indent_label(call_stack: CallStack, indent: str = "  ") -> str:
    return indent * (len(call_stack) - 1) + call_stack[-1].label


@dataclass
class RichReporter:
    name: str
    metrics: tuple[Metric, ...] = DEFAULT_METRICS

    def report(self, log: Log):
        # Report a table with metrics as column names, events as rows
        name = self.name
        title = f"Timings for [b][blue]{name} :stopwatch:[/blue][/b]"
        report = table.Table(
            title=title, row_styles=("", "on black"), title_style="white"
        )

        # TODO: range upper bound should probably include sums
        meta = FormatMetadata(log)

        summary = None
        top_level_events = [
            events for events in log.events if len(events.call_stack) == 1
        ]
        if len(top_level_events) == 1:
            assert top_level_events[0] is log.events[0]
            summary, *events = log.events
            events = [e.pop_from_front() for e in events]
            report.show_footer = True
        else:
            events = log.events

        # Columns are metrics, plus "Stage" at the beginning for labels
        report.add_column("Stage", footer="Total" if summary else "", style="bold blue")

        for metric in self.metrics:
            kwargs = {"justify": "right", **metric.rich_args}
            footer = metric.format(summary, meta) if summary else ""
            report.add_column(metric.name, footer=footer, **kwargs)  # type: ignore

        # XXX: transpose the outer dimension from scope to event type
        # Rows are ScopedEvents
        for prev_scope, scope in zip([None, *events], events):
            if prev_scope:  # Add context rows if necessary
                prefix = common_prefix(prev_scope.call_stack, scope.call_stack)
                for i in range(len(prefix) + 1, len(scope.call_stack)):
                    report.add_row(indent_label(scope.call_stack[:i]))

            cells = [metric.format(scope, meta) for metric in self.metrics]
            report.add_row(indent_label(scope.call_stack), *cells)

        console.Console().print(report)
