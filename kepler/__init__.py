import functools

from . import reporting
from .timer import Timer


TIMER = Timer()
SNAPSHOT = TIMER.snapshot()


# functions operating on the static context
time = TIMER.time
report = functools.partial(reporting.report, SNAPSHOT)
report_snapshot = TIMER.report_snapshot
snapshot = TIMER.snapshot
split = TIMER.split
