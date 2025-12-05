import kepler
from kepler.scope import Scope


def test_empty_report():
    scope = Scope()
    assert not scope.scopes
    assert not scope.events

    # Test empty log creation and reporting
    kepler.report("Empty Test", scope)
