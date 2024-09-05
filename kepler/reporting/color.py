from __future__ import annotations

import colorsys
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

Range: TypeAlias = tuple[float, float]


@dataclass
class HLSColorGradient:
    smoothing: float = 0.5
    # lower bound is red (higher time), upper bound is bluish green (lower time)
    h_range: Range = (0.6, 0.0)
    l_range: Range = (0.5, 0.5)
    s_range: Range = (1, 1)

    def color(self, value: float, range: Range):
        hls = self.hls(self.log_feature_norm(value, range))
        return self.rich_color(*hls)

    def feature_norm(self, value: float, range: Range):
        ymin, ymax = range[0] - self.smoothing, range[1] + self.smoothing
        return (value - ymin) / (ymax - ymin)

    def log_feature_norm(self, value: float, range: Range):
        clipped = np.clip(value, *range)
        return self.feature_norm(np.log(clipped), range=np.log(range))

    def hls(self, value: float) -> tuple[float, float, float]:
        assert 0 <= value <= 1
        scale = lambda value, ymin, ymax: (ymin + (ymax - ymin) * value)
        return (
            scale(value, *self.h_range),
            scale(value, *self.l_range),
            scale(value, *self.s_range),
        )

    def rich_color(self, h: float, l: float, s: float) -> str:
        r, g, b = colorsys.hls_to_rgb(h, l, s)  # in range [0, 1], not [0, 255]
        color_short = lambda normed: int(np.round(normed * 255))
        return f"rgb({color_short(r)},{color_short(g)},{color_short(b)})"
