from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt
from rich import pretty, text

from .brail import brail_bars
from .color import HLSColorGradient
from .units import Bytes


Histogram: TypeAlias = tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
Stat = TypeVar("Stat")
Measurement = TypeVar("Measurement")


@dataclass
class Metadata(Generic[Stat]):
    all_stats: list[Stat]
    all_stats: list[Stat]

    @functools.cached_property
    def data_range(self) -> tuple[Stat, Stat]:
        return min(self.all_stats), max(self.all_stats)  # type: ignore

    @functools.cached_property
    def hist_data_range(self) -> tuple[float, float]:
        return (
            min(a.min() for _, a in self.all_stats),  # type: ignore
            max(a.max() for _, a in self.all_stats),  # type: ignore
        )


class Formatter(Protocol, Generic[Stat]):
    def format(self, value: Stat, /, meta: Metadata[Stat]) -> Any: ...


class Pretty:
    def format(self, value: Any, meta: Metadata[Any]):
        return pretty.Pretty(value)


@dataclass
class BytesFormatter:
    gradient: HLSColorGradient = field(default_factory=HLSColorGradient)

    def format(self, bytes: int, meta: Metadata[int]) -> text.Text:
        color = self.gradient.color(bytes, meta.data_range)
        return text.Text(Bytes.format(bytes), style=color)


@dataclass
class Sparkline:
    gradient: HLSColorGradient = field(default_factory=HLSColorGradient)

    def format(self, hist: Histogram, meta: Metadata[float]) -> text.Text:
        runes = self.brail_sparkline(hist)

        _, bins = hist
        # TODO: mean min and max, not mean of means
        bin_means = np.stack((bins[1:], bins[:-1]), axis=1).mean(axis=1)
        line = text.Text()
        for rune, (ymin, ymax) in zip(runes, bin_means.reshape(-1, 2)):
            color = self.gradient.color(np.mean((ymin, ymax)), meta.hist_data_range)
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
