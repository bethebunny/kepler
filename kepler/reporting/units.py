from __future__ import annotations

import enum
import functools


# Strong opinions taken here:
# - 3 significant digits plus a unit is more than enough for most use cases
#    - most uses of this library trying to gauge order-of-magnitude
#    - more than 1e-3 variance in most macroscopic timing things
# - Present the most immediately semantically meaningful unit possible
#    - Give a standard unit if available
#    - If not available within 3 significant digits, report a sensible _base_ unit
#    - For time


@functools.total_ordering
class Unit(enum.Enum):
    shortname: str
    order: float

    def __init__(self, shortname: str, order: float):
        self.shortname = shortname
        self.order = order

    def __lt__(self, other: Unit):
        if not (isinstance(other, type(self)) or isinstance(self, type(other))):
            raise TypeError(f"Incomparable unit types: {self=}, {other=}")
        return self.order < other.order

    def _format(self, units: float, precision: int = 3):
        return f"{units/self.order:.{precision}g} {self.shortname}"

    @classmethod
    def best_unit(cls, units: float):
        return max(
            (unit for unit in cls if 1 <= (units // unit.order) < 1000),
            default=cls.default(),
        )

    @classmethod
    def default(cls):
        return min(cls)

    @classmethod
    def format(cls, units: float, precision: int = 3):
        # It makes sense to format a fractional number of bytes because eg.
        # this might be used to represent a statistic or derived quantity, like
        # the mean of a distribution of bytes, or bytes/second.
        return cls.best_unit(units)._format(units, precision=precision)


class MetricUnit(Unit):
    QUECTO = ("q", 1e-30)
    RONTO = ("r", 1e-27)
    YOCTO = ("y", 1e-24)
    ZEPTO = ("z", 1e-21)
    ATTO = ("a", 1e-18)
    FEMTO = ("f", 1e-15)
    PICO = ("p", 1e-12)
    NANO = ("n", 1e-9)
    MICRO = ("μ", 1e-6)
    MILLI = ("m", 1e-3)
    UNIT = ("", 1)
    KILO = ("k", 1e3)
    MEGA = ("M", 1e6)
    GIGA = ("G", 1e9)
    TERA = ("T", 1e12)
    PETA = ("P", 1e15)
    EXA = ("E", 1e18)
    ZETA = ("Z", 1e21)
    YOTTA = ("Y", 1e24)
    RONNA = ("R", 1e27)
    QUETTA = ("Q", 1e30)

    @classmethod
    def default(cls):
        return cls.UNIT


class Bytes(Unit):
    BYTE = ("B", 1)
    K = ("KiB", 2**10)  # I respectfully refuse to type "****byte"
    M = ("MiB", 2**20)
    G = ("GiB", 2**30)
    T = ("TiB", 2**40)
    P = ("PiB", 2**50)
    E = ("EiB", 2**60)
    Z = ("ZiB", 2**70)
    Y = ("YiB", 2**80)


class Time(Unit):
    SECOND = ("s", 1)
    MINUTE = ("m", 60)
    HOUR = ("h", 60 * 60)
    DAY = ("d", 24 * 60 * 60)
    YEAR = ("y", 365 * 24 * 60 * 60)

    @classmethod
    def format_nanos(cls, nanos: float, precision: int = 3) -> str:
        return cls.format(nanos * 1e-9, precision=precision)

    @classmethod
    def format(cls, units: float, precision: int = 3) -> str:
        seconds = units
        unit = cls.best_unit(seconds)
        if unit < cls.MINUTE:
            # Format as a fractional number of the unit, to 3 digits of precision
            return f"{MetricUnit.format(seconds, precision=precision)}s"

        # Format rounding to the top two units, eg. 2y186d
        units = int(seconds) // unit.order
        subunit = max(subunit for subunit in Time if subunit < unit)
        subunits = (int(seconds) % unit.order) // subunit.order
        return f"{units}{unit.shortname}{subunits}{subunit.shortname}"
