from __future__ import annotations

from dataclasses import dataclass, field
import functools
import itertools
from typing import Any, Generic, Iterable, Protocol, TypeVar

try:
    from typing import TypeAlias
except ImportError:  # python 3.9
    from typing_extensions import TypeAlias

import numpy as np
import numpy.typing as npt
from rich import pretty, text

from .color import HLSColorGradient
from .brail import brail_bars
from ..event import Event, Log
from .units import Time

flatten = itertools.chain.from_iterable
Histogram: TypeAlias = tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
T = TypeVar("T", contravariant=True)


class Formatter(Protocol, Generic[T]):
    def format(self, value: T, /, meta: FormatMetadata) -> Any: ...


@dataclass
class FormatMetadata:
    log: Log

    @property
    def all_events(self) -> Iterable[Event]:
        for event in self.log.events:
            yield from event.events

    @functools.cached_property
    def data_range(self) -> tuple[float, float]:
        return (
            min(event.value for event in self.all_events),
            max(event.value for event in self.all_events),
        )


class Pretty(Formatter[Any]):
    def format(self, value: Any, meta: FormatMetadata):
        return pretty.Pretty(value)


@dataclass
class TimedeltaFormatter:
    gradient: HLSColorGradient = field(default_factory=HLSColorGradient)

    def format(self, nanos: int, meta: FormatMetadata) -> text.Text:
        color = self.gradient.color(nanos, meta.data_range)
        return text.Text(Time.format_nanos(nanos), style=color)


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
        # - normalize as a ratio of the mean
        #   - this puts approximately uniform distributions at 1 pixel height
        # - always put at least 1 pixel per non-empty bin
        bins = (counts / counts.mean()).round()
        bins = np.maximum(bins, counts > 0)
        return brail_bars(bins.clip(max=pixel_height).astype(np.int8))
