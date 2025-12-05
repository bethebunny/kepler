from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Protocol

import numpy as np
from rich import console, table

from ..event import CallStack
from ..scope import Scope
from ..timer import TimingEvent
from .format import FormatMetadata, Sparkline, TimedeltaFormatter
from .metric import Metric


class Reporter(Protocol):
    def report(self, scope: Scope): ...


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

    def report(self, scope: Scope):
        # Report a table with metrics as column names, events as rows
        name = self.name
        title = f"Timings for [b][blue]{name} :stopwatch:[/blue][/b]"
        report = table.Table(
            title=title, row_styles=("", "on black"), title_style="white"
        )

        # TODO: range upper bound should probably include sums
        meta = FormatMetadata(list(scope.export()))

        # XXX: remove this :P
        while not scope.events[TimingEvent] and len(scope.scopes) == 1:
            scope = next(iter(scope.scopes.values()))

        # These are TimingEvents, but the type system doesn't know this yet
        summary = [e.duration for e in scope.events[TimingEvent]]  # type: ignore
        report.show_footer = bool(summary)

        # Columns are metrics, plus "Stage" at the beginning for labels
        report.add_column("Stage", footer="Total" if summary else "", style="bold blue")

        for metric in self.metrics:
            kwargs = {"justify": "right", **metric.rich_args}
            footer = metric.format(summary, meta) if summary else ""
            report.add_column(metric.name, footer=footer, **kwargs)  # type: ignore

        def report_scope(label: str, scope: Scope, indent: int = 0):
            # These are TimingEvents, but the type system doesn't know this yet
            values = [e.duration for e in scope.events[TimingEvent]]  # type: ignore
            cells = [metric.format(values, meta) for metric in self.metrics]
            report.add_row("  " * indent + label, *cells)
            for caller_id, subscope in scope.scopes.items():
                report_scope(caller_id.label, subscope, indent + 1)

        for caller_id, subscope in scope.scopes.items():
            report_scope(caller_id.label, subscope)

        console.Console().print(report)
