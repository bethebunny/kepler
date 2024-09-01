from __future__ import annotations
import colorsys
from dataclasses import dataclass, field
import functools
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np
import numpy.typing as npt
from rich import console, pretty, table, text

from .timer import Timer, TimerContext
from .units import format_timedelta_ns


@dataclass
class Metric:
    name: str
    compute: Callable[[npt.NDArray[np.float64]], int | float]
    _format: Callable[[list[Any]], Any] = np.vectorize(pretty.Pretty)
    format: Callable[[Any, FormatMetadata], Any] = lambda v, *_: pretty.Pretty(v)
    rich_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class FormatMetadata:
    metric: Metric
    events: list[Event]
    data_range: tuple[float, float]


def gradient_td(timedelta: float, meta: FormatMetadata):
    color_norm = log_feature_norm(timedelta, *meta.data_range, smoothing=0.5)
    return text.Text(
        format_timedelta_ns(timedelta), style=hls_color_gradient(color_norm)
    )


def feature_norm(
    data: npt.NDArray[np.float64], range: Optional[tuple[float, float]] = None
):
    ymin, ymax = range or (data.min(), data.max())
    return (data - ymin) / (ymax - ymin)


def log_feature_norm(value: float, ymin: float, ymax: float, smoothing: float):
    lmin, lmax = np.log(ymin), np.log(ymax)
    ltd = np.log(np.clip(value, ymin, ymax))
    return feature_norm(ltd, range=(lmin - smoothing, lmax + smoothing))


def hls_color_gradient(
    normed: float,  # in range [0, 1]
    # lower bound is red (higher time), upper bound is bluish green (lower time)
    h_range: tuple[float, float] = (0.6, 0.0),
    l_range: tuple[float, float] = (0.5, 0.5),
    s_range: tuple[float, float] = (1, 1),
) -> list[str]:
    assert 0 <= normed <= 1
    scale = lambda value, ymin, ymax: (ymin + (ymax - ymin) * value)
    # Values are between 0 and 1
    rn, gn, bn = colorsys.hls_to_rgb(
        scale(normed, *h_range), scale(normed, *l_range), scale(normed, *s_range)
    )
    return f"rgb({int(rn * 255)},{int(gn * 255)},{int(bn * 255)})"


def histogram(timedeltas: npt.NDArray[np.float64], bins: int = 20):
    return np.histogram(timedeltas, bins=bins)


def brail(data: npt.NDArray[np.int8]):
    def brail_chr(pair: tuple[int, int]):
        left_offset: int = [0, 0x40, 0x44, 0x46, 0x47][pair[0]]
        right_offset: int = [0, 0x80, 0xA0, 0xB0, 0xB8][pair[1]]
        return chr(0x2800 + left_offset + right_offset)

    if len(data) % 2:
        data = np.append(data, [0])
    return "".join(brail_chr(pair) for pair in data.reshape((-1, 2)))


Histogram = tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]


def sparkline(hist: Histogram):
    counts, _ = hist
    pixel_height = 4
    # To get a reasonable histogram of the data with 4 pixels of y axis:
    # - linearly scale the bin counts
    # - set y limit to pixel_height / bin size
    #   - this puts approximately uniform distributions right on the boundary
    #     between 1 and 2 pixels
    # - always put at least 1 pixel per non-empty bin
    ymax = pixel_height / len(counts)
    normed = pixel_height * counts / (counts.sum() * ymax)
    normed = np.where((normed > 0) & (normed.round() == 0), 1, normed.round())
    return brail(normed.clip(max=pixel_height).astype(np.int8))


def colored_sparkline(hist: Histogram, meta: FormatMetadata):
    _, bins = hist
    bin_means = np.stack((bins[1:], bins[:-1]), axis=1).mean(axis=1)
    line = text.Text()
    for rune, (ymin, ymax) in zip(sparkline(hist), bin_means.reshape(-1, 2)):
        color_norm = log_feature_norm(
            np.mean((ymin, ymax)), *meta.data_range, smoothing=0.5
        )
        line.append(rune, style=hls_color_gradient(color_norm))
    return line


DEFAULT_METRICS = (
    Metric("Count", len),
    Metric("Total", np.sum, format=gradient_td),
    Metric("Average", np.mean, format=gradient_td),
    Metric("Min", np.min, format=gradient_td),
    Metric(
        "Histogram",
        lambda a: np.histogram(a, bins=20),
        format=colored_sparkline,
    ),
    Metric("Max", np.max, format=gradient_td),
    Metric("P50", lambda a: float(np.percentile(a, 50)), format=gradient_td),
    Metric("P90", lambda a: float(np.percentile(a, 90)), format=gradient_td),
    Metric("P99", lambda a: float(np.percentile(a, 99)), format=gradient_td),
)


CallStack = list[str]


@dataclass
class Event:
    call_stack: CallStack
    times: npt.NDArray[np.float64]
    metrics: dict[str, Any]

    @property
    def name(self):
        return self.call_stack[-1]

    @property
    def indented_name(self, indent: str = "  "):
        return indent * (len(self.call_stack) - 1) + self.name


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


def common_prefix(l: CallStack, r: CallStack) -> CallStack:
    for i, (lv, rv) in enumerate(zip(l, r)):
        if lv != rv:
            return l[:i]
    return l[: len(r)]


class Reporter:
    def report(self, ctx: TimerContext):
        pass

    def events(self, ctx: TimerContext, metrics: Sequence[Metric]) -> list[Event]:
        return [
            Event(
                call_stack,
                times := timer.events,
                {metric.name: metric.compute(times) for metric in metrics},
            )
            for call_stack, timer in flat_timers(ctx)
        ]


@dataclass
class RichReporter(Reporter):
    name: str
    metrics: tuple[Metric, ...] = DEFAULT_METRICS

    def report(self, ctx: TimerContext):
        events = self.events(ctx, self.metrics)

        data_range = (
            min(min(event.times) for event in events if event.times),
            max(max(event.times) for event in events if event.times),
        )
        rows = [
            (
                event,
                [
                    metric.format(
                        event.metrics[metric.name],
                        FormatMetadata(metric, events, data_range),
                    )
                    for metric in self.metrics
                ],
            )
            for event in events
        ]

        name = self.name
        prefix = functools.reduce(common_prefix, (e.call_stack for e in events))
        if prefix:
            name = (f"{name}: " if name else "") + " -> ".join(prefix)
            for event in events:
                event.call_stack = event.call_stack[len(prefix) :]

        title = f"Timings for [b][blue]{name} :stopwatch:[/blue][/b]"

        report = table.Table(
            title=title, row_styles=("", "on black"), title_style="white"
        )

        if not events[0].call_stack:
            # First event is summary
            (_, summary), *rows = rows
            report.show_footer = True

            report.add_column("Stage", "Total", style="bold blue")
            for metric, footer in zip(self.metrics, summary):
                kwargs = {"justify": "right", **metric.rich_args}
                report.add_column(metric.name, footer=footer, **kwargs)  # type: ignore
        else:
            report.add_column("Stage", style="bold blue")
            for metric in self.metrics:
                kwargs = {"justify": "right", **metric.rich_args}
                report.add_column(metric.name, **kwargs)  # type: ignore

        prev_stack: CallStack = []
        for event, row in rows:
            stack = event.call_stack
            prefix = common_prefix(prev_stack, stack)
            if len(stack) - len(prefix) > 1:
                new_context = stack[len(prefix) : -1]
                report.add_row("  " * len(prefix) + " ".join(new_context))
            prev_stack = stack

            report.add_row(event.indented_name, *row, end_section=False)

        console.Console().print(report)
