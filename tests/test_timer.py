import time
import pytest

from kepler import Timer
from kepler.timer import current_context
from kepler.reporting.reporter import flat_events


@pytest.fixture
def timer():
    with (timer := Timer()).context:
        yield timer


def test_split(timer: Timer):
    split = timer.stopwatch("watch")
    split("1")
    time.sleep(0.001)
    split("2")
    time.sleep(0.001)
    events = flat_events(current_context())
    assert len(events) == 2
    assert all("watch" in event.call_stack[0] for event in events)
