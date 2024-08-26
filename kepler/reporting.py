import colorsys
from dataclasses import dataclass, field
import functools
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import numpy.typing as npt
from rich import console, pretty, table, text

from .timer import Timer, TimerContext


SECONDS_IN_MINUTE = 60
SECONDS_IN_HOUR = 60 * SECONDS_IN_MINUTE
SECONDS_IN_DAY = 24 * SECONDS_IN_HOUR


@dataclass
class Metric:
    name: str
    compute: Callable[[npt.NDArray[np.float64]], int | float]
    format: Callable[[npt.NDArray[Any]], Any] = np.vectorize(pretty.Pretty)
    rich_args: dict[str, Any] = field(default_factory=dict)


def gradient_td(timedeltas: npt.NDArray[np.float64]):
    formatted = np.vectorize(format_timedelta)(timedeltas)
    colors = hls_color_gradient(timedeltas)
    return [text.Text(td, style=color) for td, color in zip(formatted, colors)]


def hls_color_gradient(
    array: npt.NDArray[Any],
    smoothing: float = 1,
    # low is bluish green, high is red
    h_range: tuple[float, float] = (0, 0.6),
    l_range: tuple[float, float] = (0.5, 0.5),
    s_range: tuple[float, float] = (1, 1),
    reversed: bool = False,
) -> list[str]:
    log = np.log(array)
    min = log.min() - smoothing
    max = log.max() + smoothing
    log_normed = (log - min) / (max - min)
    if not reversed:
        log_normed = 1 - log_normed
    h = h_range[0] + (h_range[1] - h_range[0]) * log_normed
    l = l_range[0] + (l_range[1] - l_range[0]) * log_normed
    s = s_range[0] + (s_range[1] - s_range[0]) * log_normed

    def hue_to_color(h: float, l: float, s: float):
        rn, gn, bn = colorsys.hls_to_rgb(h, l, s)
        color = f"rgb({int(rn * 255)},{int(gn * 255)},{int(bn * 255)})"
        return color

    return [hue_to_color(*hls) for hls in zip(h, l, s)]


def format_timedelta(seconds: float):
    if seconds > 60:
        days, seconds = int(seconds // SECONDS_IN_DAY), seconds % SECONDS_IN_DAY
        hours, seconds = (
            int(seconds // SECONDS_IN_HOUR),
            seconds % SECONDS_IN_HOUR,
        )
        minutes, seconds = (
            int(seconds // SECONDS_IN_MINUTE),
            seconds % SECONDS_IN_MINUTE,
        )
        if days:
            if minutes >= 30:
                hours += 1
            return f"{days}d{hours}h"
        elif hours:
            if seconds >= 30:
                minutes += 1
            return f"{hours}h{minutes}m"
        else:
            seconds = int(round(seconds))
            return f"{minutes}m{seconds}s"
    elif seconds > 1:
        return f"{seconds:.1f}s"
    elif seconds > 0.001:
        return f"{seconds * 1000:.1f}ms"
    elif seconds > 1e-6:
        return f"{seconds * 1000000:.1f}us"
    else:
        return f"{seconds * 1e9:.1f}ns"


DEFAULT_METRICS = (
    Metric("Count", len),
    Metric("Total", np.sum, format=gradient_td),
    Metric("Average", np.mean, format=gradient_td),
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
        name = f":stopwatch: {caller_id}"
        yield from flat_timers(sw_ctx, call_stack + [name])


def common_prefix(l: CallStack, r: CallStack) -> CallStack:
    for i, (lv, rv) in enumerate(zip(l, r)):
        if lv != rv:
            return l[:i]
    return l[: len(r)]


class Reporter:
    def report(self, ctx: TimerContext):
        pass

    def events(
        self, ctx: TimerContext, metrics: Sequence[Metric]
    ) -> list[Event]:
        return [
            Event(
                call_stack,
                (times := np.array(timer.events)),
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
        prefix = functools.reduce(common_prefix, (e.call_stack for e in events))

        columns = [
            (metric, np.array([event.metrics[metric.name] for event in events]))
            for metric in self.metrics
        ]
        formatted_columns = [metric.format(col) for metric, col in columns]
        rows = list(zip(events, zip(*formatted_columns)))

        name = self.name
        if prefix:
            name = (f"{name}: " if name else "") + " -> ".join(prefix)
            for event in events:
                event.call_stack = event.call_stack[len(prefix) :]

        title = f"Timings for [b][blue]{name} :stopwatch:[/blue][/b]"
        # TODO: figure out how to format stopwatch regions

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
                report.add_column(metric.name, footer=footer, **kwargs)
        else:
            report.add_column("Stage", style="bold blue")
            for metric in self.metrics:
                kwargs = {"justify": "right", **metric.rich_args}
                report.add_column(metric.name, **kwargs)

        for event, row in rows:
            report.add_row(event.indented_name, *row, end_section=False)

        console.Console().print(report)
