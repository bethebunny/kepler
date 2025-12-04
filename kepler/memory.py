import collections
import gc
import sys
import weakref
from dataclasses import dataclass

from . import Event, measurement


def _memory_snapshot():
    gc.collect()
    objs = list(gc.get_objects())
    counts = collections.Counter(type(o) for o in objs)
    usage = collections.defaultdict(list)
    for o in objs:
        usage[type(o)].append(sys.getsizeof(o))
    return weakref.WeakKeyDictionary((t, (counts[t], sum(usage[t]))) for t in counts)


def _snapshot_delta(s1, s2):
    snapshot = weakref.WeakKeyDictionary()
    for t in set(s1.keys()) | set(s2.keys()):
        pc, pt = s1.get(t, (0, 0))
        nc, nt = s2.get(t, (0, 0))
        if (count := nc - pc) or (total := nt - pt):
            snapshot[t] = count, total
    return snapshot


def _print_delta(s1, s2):
    delta = _snapshot_delta(s1, s2)
    print("              type               |  Δcount  |  Δbytes  |  count  |    bytes")
    print(
        "---------------------------------+----------+----------+---------+--------------"
    )
    for type, (d_count, d_total) in sorted(
        delta.items(), key=lambda p: p[1][0], reverse=True
    ):
        count, total = s2.get(type, (0, 0))
        print(
            f" {type.__qualname__:<31} | {d_count:<8} | {d_total:<8} | {count:<7} | {total}"
        )


@dataclass
class LiveObjectsDeltaEvent(Event):
    delta: dict[type, tuple[int, int]]

    @property
    def value(self):
        return 1

    def json(self):
        return {}


@measurement
def log_memory():
    # Janky way to (almost) discount overhead of memory logging
    snapshot = _memory_snapshot()
    snapshot = _memory_snapshot()
    yield
    new_snapshot = _memory_snapshot()
    _print_delta(snapshot, new_snapshot)
    return LiveObjectsDeltaEvent({})


if __name__ == "__main__":
    from . import report

    foo = []
    for i in log_memory("memory", range(20)):
        foo.append({i})

    report()
