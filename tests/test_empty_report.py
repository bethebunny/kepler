import pytest

import kepler
from kepler import timer


def test_empty_report():
    ctx = timer.TimerContext()
    assert not ctx.timers
    assert not ctx.stopwatches
    with pytest.raises(ValueError):
        kepler.report()
