import contextlib
import inspect
import timeit
import weakref
from typing import overload, Any, Callable, ContextManager

from .types import TimingEntry, TimingKey, CallerID, Snapshot
from .reporting import report


_TIMER = timeit.default_timer


class Timer:
    def __init__(self):
        self.current_stack: list[TimingEntry] = []
        self.current_split: float = _TIMER()
        self.snapshots = weakref.WeakSet[Snapshot]()

    def snapshot(self, name: str = ""):
        self.snapshots.add(snapshot := Snapshot(name=name, start=_TIMER()))
        return snapshot

    def _publish_time(self, key: TimingKey, time: float):
        for snapshot in self.snapshots:
            snapshot.times[key].append(time)

    @property
    def _current_key(self) -> TimingKey:
        return tuple(e.caller_id for e in self.current_stack)

    def push(self, caller_id: CallerID):
        self.current_stack.append(TimingEntry(caller_id, _TIMER()))

    def pop(self):
        key = tuple(e.caller_id for e in self.current_stack)
        entry = self.current_stack.pop()
        self._publish_time(key, _TIMER() - entry.start_time)

    def split(self, label: str):
        """Publish a split time based on the timer's current split, and reset it."""
        time = _TIMER()
        self._publish_time(
            self._current_key + (CallerID.from_caller(label),),
            time - self.current_split,
        )
        self.current_split = time

    def tare(self):
        """Resets the split timer without marking a split."""
        self.current_split = timeit.default_timer()

    @overload
    def time(self, label_or_fn: str) -> ContextManager[None]:
        ...

    @overload
    def time(self, label_or_fn: Callable[..., Any]) -> Callable[..., Any]:
        ...

    def time(self, label_or_fn: str | Callable[..., Any]):
        if isinstance(label_or_fn, str):
            caller_id = CallerID.from_frame(label_or_fn, inspect.stack()[-1])
            return self._time(caller_id)
        return self._time(CallerID.from_fn(label_or_fn))(label_or_fn)

    @contextlib.contextmanager
    def _time(self, caller_id: CallerID):
        self.push(caller_id or CallerID.from_frame(label, inspect.stack()[-1]))
        try:
            yield
        finally:
            self.pop()

    @contextlib.contextmanager
    def report_snapshot(self, name: str):
        snapshot = self.snapshot(name=name)
        try:
            yield
        finally:
            report(snapshot)
