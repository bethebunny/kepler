from __future__ import annotations

import enum
import functools
import math


@functools.total_ordering
class Unit(enum.Enum):
    shortname: str
    base: int

    def __init__(self, shortname, base):
        self.shortname = shortname
        self.base = base

    NANOS = ("ns", 1000)
    MICROS = ("us", 1000)
    MILLIS = ("ms", 1000)
    SECONDS = ("s", 60)
    MINUTES = ("m", 60)
    HOURS = ("h", 24)
    DAYS = ("d", 365)
    YEARS = ("y", None)

    @functools.cached_property
    def in_nanos(self):
        return math.prod(unit.base for unit in Unit if unit < self)

    @functools.cached_property
    def _sort_key(self):
        return next(i for i, unit in enumerate(Unit) if self is unit)

    def __lt__(self, other: Unit):
        return self._sort_key < other._sort_key

    def format(self, units):
        return f"{units}{self.shortname}"


def format_timedelta_ns(nanos: int, precision: int = 3) -> str:
    unit = max((unit for unit in Unit if (nanos // unit.in_nanos)), default=Unit.NANOS)
    if unit >= Unit.MINUTES:
        # Format rounding to the top two units, eg. 2y186d
        second = max(second for second in Unit if second < unit)
        units = nanos // unit.in_nanos
        seconds = (nanos % unit.in_nanos) // second.in_nanos
        return f"{unit.format(units)}{second.format(seconds)}"
    # Format as a fractional number of the unit, to 3 digits of precision
    units = nanos / unit.in_nanos
    return f"{units:.3g}{unit.shortname}"
