import colorsys
from dataclasses import dataclass, field
import typing
from typing import Any, Callable, Iterable

import numpy as np
import numpy.typing as npt
from rich import console, pretty, table, text

from .timer import Timer


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

    def hue_to_color(hls: npt.NDArray[np.float64]):
        rn, gn, bn = colorsys.hls_to_rgb(*hls)
        return f"rgb({int(rn * 255)},{int(gn * 255)},{int(bn * 255)})"

    return list(
        np.apply_along_axis(hue_to_color, 1, np.stack([h, l, s], axis=1))
    )


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
    Metric("Count", len, rich_args={"justify": "right"}),
    Metric("Total", np.sum, format=gradient_td),
    Metric("Average", np.mean, format=gradient_td),
    Metric("Max", np.max, format=gradient_td),
    Metric("P50", lambda a: float(np.percentile(a, 50)), format=gradient_td),
    Metric("P90", lambda a: float(np.percentile(a, 90)), format=gradient_td),
    Metric("P99", lambda a: float(np.percentile(a, 99)), format=gradient_td),
)


@dataclass
class Event:
    call_path: list[str]
    times: npt.NDArray[np.float64]

    @property
    def name(self):
        return self.call_path[-1]

    @property
    def indented_name(self, indent: str = "  "):
        return indent * (len(self.call_path) - 1) + self.name


def flat_events(timer: Timer, call_stack: list[str] = []) -> Iterable[Event]:
    yield Event(call_stack, np.array(timer.events))
    for caller_id, timer in timer.timers.items():
        yield from flat_events(timer, call_stack + [caller_id.label])


def report(
    timer: Timer, name: str, metrics: typing.Iterable[Metric] = DEFAULT_METRICS
):
    events = list(flat_events(timer, []))
    metric_rows = [
        [metric.compute(e.times) for metric in metrics] for e in events
    ]
    formatted_columns = [
        metric.format(np.array(column))
        for metric, column in zip(metrics, zip(*metric_rows))
    ]
    summary, *rows = list(zip(*formatted_columns))

    report = table.Table(
        title=f"Timings for [b][blue]{name} :stopwatch:[/blue][/b]",
        show_footer=True,
        row_styles=("", "on black"),
        title_style="white",
    )

    report.add_column("Stage", "Total", style="bold blue")
    for metric, footer in zip(metrics, summary):
        report.add_column(metric.name, footer=footer, **metric.rich_args)

    # First event is summary
    for event, row in zip(events[1:], rows):
        report.add_row(event.indented_name, *row, end_section=False)

    console.Console().print(report)
