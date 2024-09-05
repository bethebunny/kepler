from __future__ import annotations

from dataclasses import dataclass, field
import functools
import itertools
from typing import Any, Generic, Protocol, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt
from rich import pretty, text

from .color import HLSColorGradient
from .brail import brail_bars
from .event import Event
from .units import format_timedelta_ns

flatten = itertools.chain.from_iterable
Histogram: TypeAlias = tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
T = TypeVar("T")


class Formatter(Protocol, Generic[T]):
    def format(self, value: T, meta: FormatMetadata) -> Any: ...


@dataclass
class FormatMetadata:
    events: list[Event]

    @functools.cached_property
    def data_range(self) -> tuple[float, float]:
        return (
            min(flatten(event.times for event in self.events)),
            max(flatten(event.times for event in self.events)),
        )


class Pretty(Formatter[Any]):
    def format(self, value: Any, meta: FormatMetadata):
        return pretty.Pretty(value)


@dataclass
class TimedeltaFormatter:
    gradient: HLSColorGradient = field(default_factory=HLSColorGradient)

    def format(self, timedelta: float, meta: FormatMetadata) -> text.Text:
        color = self.gradient.color(timedelta, meta.data_range)
        return text.Text(format_timedelta_ns(timedelta), style=color)


@dataclass
class Sparkline:
    gradient: HLSColorGradient = field(default_factory=HLSColorGradient)

    def format(self, hist: Histogram, meta: FormatMetadata) -> text.Text:
        runes = self.brail_sparkline(hist)

        _, bins = hist
        # TODO: mean min and max, not mean of means
        bin_means = np.stack((bins[1:], bins[:-1]), axis=1).mean(axis=1)
        line = text.Text()
        for rune, (ymin, ymax) in zip(runes, bin_means.reshape(-1, 2)):
            color = self.gradient.color(np.mean((ymin, ymax)), meta.data_range)
            line.append(rune, style=color)
        return line

    def brail_sparkline(self, hist: Histogram):
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
        return brail_bars(normed.clip(max=pixel_height).astype(np.int8))
