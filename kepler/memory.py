import collections
import gc
import psutil
import sys
from dataclasses import dataclass


from .reporting import Statistic, RichReporter
from .reporting.format import BytesFormatter
from .report import register_default_report
from .scope import scope
from . import Event, measurement, log


def _memory_snapshot():
    gc.collect()
    objs = list(gc.get_objects())
    counts = collections.Counter(type(o) for o in objs)
    usage = collections.defaultdict(list)
    for o in objs:
        usage[type(o)].append(sys.getsizeof(o))
    return {t: (counts[t], sum(usage[t])) for t in counts}


def _snapshot_delta(s1, s2):
    snapshot = {}
    for t in set(s1.keys()) | set(s2.keys()):
        pc, pt = s1.get(t, (0, 0))
        nc, nt = s2.get(t, (0, 0))
        if (count := nc - pc) or (total := nt - pt):
            snapshot[t] = count, total
    return snapshot


@dataclass
class GCDeltaEvent(Event):
    d_count: int
    d_bytes: int
    count: int
    bytes: int


def _log_delta(s1, s2):
    delta = _snapshot_delta(s1, s2)
    for type, (d_count, d_total) in sorted(
        delta.items(), key=lambda p: p[1][0], reverse=True
    ):
        count, total = s2.get(type, (0, 0))
        with scope(type.__name__):
            log(GCDeltaEvent(d_count, d_total, count, total))


@dataclass
class RSSMemoryEvent(Event):
    bytes: int


PROCESS = psutil.Process()


@measurement
def rss():
    yield
    return RSSMemoryEvent(PROCESS.memory_info().rss)


register_default_report(
    RichReporter(
        title="RSS memory in [b][blue]{name}[/blue][/b]",
        event_type=RSSMemoryEvent,
        statistics=(
            Statistic("Min", max, lambda e: e.bytes, formatter=BytesFormatter()),
            Statistic("Max", max, lambda e: e.bytes, formatter=BytesFormatter()),
        ),
    )
)


@measurement
def objects():
    # Janky way to (almost) discount overhead of memory logging
    snapshot = _memory_snapshot()
    # snapshot = _memory_snapshot()
    yield
    new_snapshot = _memory_snapshot()
    _log_delta(snapshot, new_snapshot)


GCDELTA_STATISTICS = (
    Statistic("Δ Objects", sum, lambda e: e.d_count),
    Statistic("# Objects", max, lambda e: e.count),
    Statistic("Δ Bytes", sum, lambda e: e.d_bytes, formatter=BytesFormatter()),
    Statistic("# Bytes", max, lambda e: e.bytes, formatter=BytesFormatter()),
)


register_default_report(
    RichReporter(
        title="Tracked objects in [b][blue]{name}[/blue][/b]",
        event_type=GCDeltaEvent,
        statistics=GCDELTA_STATISTICS,
    )
)


if __name__ == "__main__":
    from .report import report

    foo = []
    for i in objects("memory", range(20)):
        foo.append({i})

    report("main")
