import kepler
from kepler.context import Context
from kepler.log import Log


def test_empty_report():
    ctx = Context()
    assert not ctx.scopes

    # Test empty log creation and reporting
    empty_log = Log.from_events(ctx.export())
    kepler.report("Empty Test", empty_log)
