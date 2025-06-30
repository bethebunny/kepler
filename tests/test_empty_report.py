import kepler
from kepler import timer
from kepler.event import Log


def test_empty_report():
    ctx = timer.TimerContext()
    assert not ctx.timers
    assert not ctx.stopwatches

    # Test empty log creation and reporting
    empty_log = Log.from_events(ctx.export())
    kepler.report("Empty Test", empty_log)
