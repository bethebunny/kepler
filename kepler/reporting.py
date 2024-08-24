import colorsys
from dataclasses import dataclass, field
import functools
import timeit
import typing
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
from rich import console, pretty, table, text

from .types import Snapshot, TimingKey


SECONDS_IN_MINUTE = 60
SECONDS_IN_HOUR = 60 * SECONDS_IN_MINUTE
SECONDS_IN_DAY = 24 * SECONDS_IN_HOUR


def format_timing_key(key: TimingKey):
    return " -> ".join(c.label for c in key)


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


def compare_timing_keys(k1: TimingKey, k2: TimingKey):
    if len(k1) < len(k2) and k2[: len(k1)] == k1:
        return -1
    elif len(k1) > len(k2) and k1[: len(k2)] == k2:
        return 1
    return 0


def report(
    snapshot: Snapshot, metrics: typing.Iterable[Metric] = DEFAULT_METRICS
):
    snapshot_time = timeit.default_timer() - snapshot.start
    keys = sorted(snapshot.times, key=functools.cmp_to_key(compare_timing_keys))
    times = [np.array(snapshot.times[key]) for key in keys]
    times.append(np.array([snapshot_time]))
    metric_rows = [[metric.compute(a) for metric in metrics] for a in times]
    formatted_columns = [
        metric.format(np.array(column))
        for metric, column in zip(metrics, zip(*metric_rows))
    ]
    formatted_rows = list(zip(*formatted_columns))
    summary = formatted_rows.pop()

    report = table.Table(
        title=f"{snapshot.name} Timings",
        show_footer=True,
        row_styles=("", "on black"),
    )

    report.add_column("Stage", "Total", style="bold blue")
    for metric, footer in zip(metrics, summary):
        report.add_column(metric.name, footer=footer, **metric.rich_args)

    for key, row in zip(keys, formatted_rows):
        report.add_row(format_timing_key(key), *row)

    console.Console().print(report)
