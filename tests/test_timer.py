import time

from kepler import Timer
from kepler.event import Log
from kepler.timer import current_context


def test_split(timer: Timer):
    split = timer.stopwatch("watch")
    split("1")
    time.sleep(0.001)
    split("2")
    time.sleep(0.001)
    log = Log.from_events(current_context().export())
    assert len(log.events) == 2
    assert all("watch" in event.call_stack[0].label for event in log.events)
